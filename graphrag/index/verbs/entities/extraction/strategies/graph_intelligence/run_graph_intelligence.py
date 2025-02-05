# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_gi,  run_extract_entities and _create_text_splitter methods to run graph intelligence."""

import networkx as nx
from datashaper import VerbCallbacks

import graphrag.config.defaults as defs
from graphrag.config.enums import LLMType
from graphrag.index.cache import PipelineCache
from graphrag.index.graph.extractors.graph import GraphExtractor
from graphrag.index.llm import load_llm
from graphrag.index.text_splitting import (
    NoopTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from graphrag.index.verbs.entities.extraction.strategies.typing import (
    Document,
    EntityExtractionResult,
    EntityTypes,
    StrategyConfig,
)
from graphrag.llm import CompletionLLM

from .defaults import DEFAULT_LLM_CONFIG


async def run_gi(
    docs: list[Document],
    entity_types: EntityTypes,
    reporter: VerbCallbacks,
    pipeline_cache: PipelineCache,
    args: StrategyConfig,
) -> EntityExtractionResult:
    """Run the graph intelligence entity extraction strategy."""
    llm_config = args.get("llm", DEFAULT_LLM_CONFIG)
    llm_type = llm_config.get("type", LLMType.StaticResponse)
    llm = load_llm("entity_extraction", llm_type, reporter, pipeline_cache, llm_config)

    # llm() -> __call__ -> execute_llm() -> 调用大模型接口
    # execute_llm函数源码在graphrag/llm/openai/openai_chat_llm.py
    return await run_extract_entities(llm, docs, entity_types, reporter, args)


async def run_extract_entities(
    llm: CompletionLLM,
    docs: list[Document],
    entity_types: EntityTypes,
    reporter: VerbCallbacks | None,
    args: StrategyConfig,
) -> EntityExtractionResult:
    """Run the entity extraction chain."""
    encoding_name = args.get("encoding_name", "cl100k_base")

    # Chunking Arguments
    prechunked = args.get("prechunked", False)
    chunk_size = args.get("chunk_size", defs.CHUNK_SIZE)
    chunk_overlap = args.get("chunk_overlap", defs.CHUNK_OVERLAP)

    # Extraction Arguments
    tuple_delimiter = args.get("tuple_delimiter", None)
    record_delimiter = args.get("record_delimiter", None)
    completion_delimiter = args.get("completion_delimiter", None)
    extraction_prompt = args.get("extraction_prompt", None)
    encoding_model = args.get("encoding_name", None)
    max_gleanings = args.get("max_gleanings", defs.ENTITY_EXTRACTION_MAX_GLEANINGS)

    # note: We're not using UnipartiteGraphChain.from_params
    # because we want to pass "timeout" to the llm_kwargs
    # 基于tiktoken的tokenizer的文本切分器
    # 如果输入的docs未被切分过（prechunked==False），就会被调用
    # 在本教程中不会被调用
    text_splitter = _create_text_splitter(
        prechunked, chunk_size, chunk_overlap, encoding_name
    )

    # 核心逻辑
    # 实例化实体提取器
    # 实体提取的源码在其__call__方法中
    extractor = GraphExtractor(
        llm_invoker=llm,
        prompt=extraction_prompt,
        encoding_model=encoding_model,
        max_gleanings=max_gleanings,
        on_error=lambda e, s, d: (
            reporter.error("Entity Extraction Error", e, s, d) if reporter else None
        ),
    )

    # 获取输入文本列表
    text_list = [doc.text.strip() for doc in docs]

    # If it's not pre-chunked, then re-chunk the input
    # 本教程中不会被调用
    if not prechunked:
        text_list = text_splitter.split_text("\n".join(text_list))

    # 执行实体提取
    # delimiter是prompt中定义的分割符
    results = await extractor(
        list(text_list),
        {
            "entity_types": entity_types,
            "tuple_delimiter": tuple_delimiter,
            "record_delimiter": record_delimiter,
            "completion_delimiter": completion_delimiter,
        },
    )

    # 获取实体提取结果，并为各个节点和边添加来源文本段的id
    graph = results.output
    # Map the "source_id" back to the "id" field
    for _, node in graph.nodes(data=True):  # type: ignore
        if node is not None:
            node["source_id"] = ",".join(
                docs[int(id)].id for id in node["source_id"].split(",")
            )

    for _, _, edge in graph.edges(data=True):  # type: ignore
        if edge is not None:
            edge["source_id"] = ",".join(
                docs[int(id)].id for id in edge["source_id"].split(",")
            )

    # 提取得到的所有实体
    entities = [
        ({"name": item[0], **(item[1] or {})})
        for item in graph.nodes(data=True)
        if item is not None
    ]
    # 转换为graphml格式的图数据
    graph_data = "".join(nx.generate_graphml(graph))
    # 返回实体提取结果
    return EntityExtractionResult(entities, graph_data)


def _create_text_splitter(
    prechunked: bool, chunk_size: int, chunk_overlap: int, encoding_name: str
) -> TextSplitter:
    """Create a text splitter for the extraction chain.

    Args:
        - prechunked - Whether the text is already chunked
        - chunk_size - The size of each chunk
        - chunk_overlap - The overlap between chunks
        - encoding_name - The name of the encoding to use
    Returns:
        - output - A text splitter
    """
    if prechunked:
        return NoopTextSplitter()

    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name,
    )
