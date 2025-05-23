# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing entity_extract methods."""

import logging
from enum import Enum
from typing import Any, cast

import pandas as pd
from datashaper import (
    AsyncType,
    TableContainer,
    VerbCallbacks,
    VerbInput,
    derive_from_rows,
    verb,
)

from graphrag.index.bootstrap import bootstrap
from graphrag.index.cache import PipelineCache

from .strategies.typing import Document, EntityExtractStrategy

log = logging.getLogger(__name__)


class ExtractEntityStrategyType(str, Enum):
    """ExtractEntityStrategyType class definition."""

    graph_intelligence = "graph_intelligence"
    graph_intelligence_json = "graph_intelligence_json"
    nltk = "nltk"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]


@verb(name="entity_extract")
async def entity_extract(
    input: VerbInput,
    cache: PipelineCache,
    callbacks: VerbCallbacks,
    column: str,
    id_column: str,
    to: str,
    strategy: dict[str, Any] | None,
    graph_to: str | None = None,
    async_mode: AsyncType = AsyncType.AsyncIO,
    entity_types=DEFAULT_ENTITY_TYPES,
    **kwargs,
) -> TableContainer:
    """
    Extract entities from a piece of text.

    ## Usage
    ### json
    ```json
    {
        "verb": "entity_extract",
        "args": {
            "column": "the_document_text_column_to_extract_entities_from", /* In general this will be your document text column */
            "id_column": "the_column_with_the_unique_id_for_each_row", /* In general this will be your document id */
            "to": "the_column_to_output_the_entities_to", /* This will be a list[dict[str, Any]] a list of entities, with a name, and additional attributes */
            "graph_to": "the_column_to_output_the_graphml_to", /* Optional: This will be a graphml graph in string form which represents the entities and their relationships */
            "strategy": {...} <strategy_config>, see strategies section below
            "entity_types": ["list", "of", "entity", "types", "to", "extract"] /* Optional: This will limit the entity types extracted, default: ["organization", "person", "geo", "event"] */
            "summarize_descriptions" : true | false /* Optional: This will summarize the descriptions of the entities and relationships, default: true */
        }
    }
    ```
    ### yaml
    ```yaml
    verb: entity_extract
    args:
        column: the_document_text_column_to_extract_entities_from
        id_column: the_column_with_the_unique_id_for_each_row
        to: the_column_to_output_the_entities_to
        graph_to: the_column_to_output_the_graphml_to
        strategy: <strategy_config>, see strategies section below
        summarize_descriptions: true | false /* Optional: This will summarize the descriptions of the entities and relationships, default: true */
        entity_types:
            - list
            - of
            - entity
            - types
            - to
            - extract
    ```

    ## Strategies
    The entity extract verb uses a strategy to extract entities from a document. The strategy is a json object which defines the strategy to use. The following strategies are available:

    ### graph_intelligence
    This strategy uses the [graph_intelligence] library to extract entities from a document. In particular it uses a LLM to extract entities from a piece of text. The strategy config is as follows:

    ```yml
    strategy:
        type: graph_intelligence
        extraction_prompt: !include ./entity_extraction_prompt.txt # Optional, the prompt to use for extraction
        completion_delimiter: "<|COMPLETE|>" # Optional, the delimiter to use for the LLM to mark completion
        tuple_delimiter: "<|>" # Optional, the delimiter to use for the LLM to mark a tuple
        record_delimiter: "##" # Optional, the delimiter to use for the LLM to mark a record

        prechunked: true | false # Optional, If the document is already chunked beforehand, otherwise this will chunk the document into smaller bits. default: false
        encoding_name: cl100k_base # Optional, The encoding to use for the LLM, if not already prechunked, default: cl100k_base
        chunk_size: 1000 # Optional ,The chunk size to use for the LLM, if not already prechunked, default: 1200
        chunk_overlap: 100 # Optional, The chunk overlap to use for the LLM, if not already prechunked, default: 100

        llm: # The configuration for the LLM
            type: openai # the type of llm to use, available options are: openai, azure, openai_chat, azure_openai_chat.  The last two being chat based LLMs.
            api_key: !ENV ${GRAPHRAG_OPENAI_API_KEY} # The api key to use for openai
            model: !ENV ${GRAPHRAG_OPENAI_MODEL:gpt-4-turbo-preview} # The model to use for openai
            max_tokens: !ENV ${GRAPHRAG_MAX_TOKENS:6000} # The max tokens to use for openai
            organization: !ENV ${GRAPHRAG_OPENAI_ORGANIZATION} # The organization to use for openai

            # if using azure flavor
            api_base: !ENV ${GRAPHRAG_OPENAI_API_BASE} # The api base to use for azure
            api_version: !ENV ${GRAPHRAG_OPENAI_API_VERSION} # The api version to use for azure
            proxy: !ENV ${GRAPHRAG_OPENAI_PROXY} # The proxy to use for azure

    ```

    ### nltk
    This strategy uses the [nltk] library to extract entities from a document. In particular it uses a nltk to extract entities from a piece of text. The strategy config is as follows:
    ```yml
    strategy:
        type: nltk
    ```
    """
    log.debug("entity_extract strategy=%s", strategy)
    if entity_types is None:
        entity_types = DEFAULT_ENTITY_TYPES
    # 核心逻辑
    # output是拓扑前序依赖的输出，在这里作为输入
    # 具体内容是切分完成的文本片段
    output = cast(pd.DataFrame, input.get_input())
    strategy = strategy or {}

    # 获取需要被执行的函数
    # 本教程中，会得到一个名为run_fi的函数
    # run_fi的源码见 ./strategies/graph_intelligence/run_graph_intelligence.py
    # 实体提取的核心源码也在那个文件之中
    strategy_exec = _load_strategy(
        strategy.get("type", ExtractEntityStrategyType.graph_intelligence)
    )
    strategy_config = {**strategy}

    num_started = 0

    async def run_strategy(row):
        nonlocal num_started
        text = row[column]
        id = row[id_column]
        # __call__(), -> execute_llm()
        result = await strategy_exec(
            [Document(text=text, id=id)],
            entity_types,
            callbacks,
            cache,
            strategy_config,
        )
        num_started += 1
        return [result.entities, result.graphml_graph]

    # 通过datashaper提供的协程调度函数，并发地提取output df的每一行，即每个文本切片的实体
    results = await derive_from_rows(
        output,
        run_strategy,
        callbacks,
        scheduling_type=async_mode,
        num_threads=kwargs.get("num_threads", 4),
    )

    # 汇总并对齐每个文本切片的实体提取结果与图谱
    to_result = []
    graph_to_result = []
    for result in results:
        if result:
            to_result.append(result[0])
            graph_to_result.append(result[1])
        else:
            to_result.append(None)
            graph_to_result.append(None)

    # to："entities"
    # graph_to："entity_graph"
    # 在output中添加两个列，分别对应实体提取结果和图谱
    output[to] = to_result
    if graph_to is not None:
        output[graph_to] = graph_to_result

    return TableContainer(table=output.reset_index(drop=True))


def _load_strategy(strategy_type: ExtractEntityStrategyType) -> EntityExtractStrategy:
    """Load strategy method definition."""
    match strategy_type:
        case ExtractEntityStrategyType.graph_intelligence:
            from .strategies.graph_intelligence import run_gi

            return run_gi

        case ExtractEntityStrategyType.nltk:
            bootstrap()
            # dynamically import nltk strategy to avoid dependency if not used
            from .strategies.nltk import run as run_nltk

            return run_nltk
        case _:
            msg = f"Unknown strategy: {strategy_type}"
            raise ValueError(msg)
