# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing cluster_graph, apply_clustering and run_layout methods definition."""

import logging
from enum import Enum
from random import Random
from typing import Any, cast

import networkx as nx
import pandas as pd
from datashaper import TableContainer, VerbCallbacks, VerbInput, progress_iterable, verb

from graphrag.index.utils import gen_uuid, load_graph

from .typing import Communities

log = logging.getLogger(__name__)


@verb(name="cluster_graph")
def cluster_graph(
    input: VerbInput,
    callbacks: VerbCallbacks,
    strategy: dict[str, Any],
    column: str,
    to: str,
    level_to: str | None = None,
    **_kwargs,
) -> TableContainer:
    """
    Apply a hierarchical clustering algorithm to a graph. The graph is expected to be in graphml format. The verb outputs a new column containing the clustered graph, and a new column containing the level of the graph.

    ## Usage
    ```yaml
    verb: cluster_graph
    args:
        column: entity_graph # The name of the column containing the graph, should be a graphml graph
        to: clustered_graph # The name of the column to output the clustered graph to
        level_to: level # The name of the column to output the level to
        strategy: <strategy config> # See strategies section below
    ```

    ## Strategies
    The cluster graph verb uses a strategy to cluster the graph. The strategy is a json object which defines the strategy to use. The following strategies are available:

    ### leiden
    This strategy uses the leiden algorithm to cluster a graph. The strategy config is as follows:
    ```yaml
    strategy:
        type: leiden
        max_cluster_size: 10 # Optional, The max cluster size to use, default: 10
        use_lcc: true # Optional, if the largest connected component should be used with the leiden algorithm, default: true
        seed: 0xDEADBEEF # Optional, the seed to use for the leiden algorithm, default: 0xDEADBEEF
        levels: [0, 1] # Optional, the levels to output, default: all the levels detected

    ```
    """
    output_df = cast(pd.DataFrame, input.get_input())
    # 核心逻辑
    # 调用分层莱顿算法，对输入图进行分层和社区划分
    # results格式：pd.Series 仅一个元素
    # results[0]格式：[(层次号,社区号,[包含的实体序列]), (层次号,社区号,[包含的实体序列]), ...]
    # 高层次中的社区，是对上一层社区的进一步划分，例如：
    # 层次0包含社区0和社区1，层次1包含社区2和社区3
    # 层次1社区2 和 层次1社区3 中的实体都来自 层次0社区1
    results = output_df[column].apply(lambda graph: run_layout(strategy, graph))

    community_map_to = "communities"
    # 保存社区划分结果到输出表
    output_df[community_map_to] = results

    level_to = level_to or f"{to}_level"
    output_df[level_to] = output_df.apply(
        lambda x: list({level for level, _, _ in x[community_map_to]}), axis=1
    )
    output_df[to] = [None] * len(output_df)

    num_total = len(output_df)

    # 核心逻辑
    # 添加层次、社区，以及度等关键信息到图中
    # Go through each of the rows
    graph_level_pairs_column: list[list[tuple[int, str]]] = []
    for _, row in progress_iterable(
        output_df.iterrows(), callbacks.progress, num_total
    ):
        levels = row[level_to]
        graph_level_pairs: list[tuple[int, str]] = []

        # 对于划分的每一层，都生成其对应的图，并将其添加到列表中
        # For each of the levels, get the graph and add it to the list
        for level in levels:
            graph = "\n".join(
                nx.generate_graphml(
                    apply_clustering(
                        cast(str, row[column]),
                        cast(Communities, row[community_map_to]),
                        level,
                    )
                )
            )
            graph_level_pairs.append((level, graph))
        graph_level_pairs_column.append(graph_level_pairs)
    output_df[to] = graph_level_pairs_column

    # explode the list of (level, graph) pairs into separate rows
    output_df = output_df.explode(to, ignore_index=True)

    # split the (level, graph) pairs into separate columns
    # TODO: There is probably a better way to do this
    output_df[[level_to, to]] = pd.DataFrame(
        output_df[to].tolist(), index=output_df.index
    )

    # clean up the community map
    output_df.drop(columns=[community_map_to], inplace=True)

    return TableContainer(table=output_df)


# TODO: This should support str | nx.Graph as a graphml param
def apply_clustering(
    graphml: str, communities: Communities, level=0, seed=0xF001
) -> nx.Graph:
    """Apply clustering to a graphml string."""
    random = Random(seed)  # noqa S311
    graph = nx.parse_graphml(graphml)
    for community_level, community_id, nodes in communities:
        if level == community_level:
            for node in nodes:
                graph.nodes[node]["cluster"] = community_id
                graph.nodes[node]["level"] = level

    # add node degree
    for node_degree in graph.degree:
        graph.nodes[str(node_degree[0])]["degree"] = int(node_degree[1])

    # add node uuid and incremental record id (a human readable id used as reference in the final report)
    for index, node in enumerate(graph.nodes()):
        graph.nodes[node]["human_readable_id"] = index
        graph.nodes[node]["id"] = str(gen_uuid(random))

    # add ids to edges
    for index, edge in enumerate(graph.edges()):
        graph.edges[edge]["id"] = str(gen_uuid(random))
        graph.edges[edge]["human_readable_id"] = index
        graph.edges[edge]["level"] = level
    return graph


class GraphCommunityStrategyType(str, Enum):
    """GraphCommunityStrategyType class definition."""

    leiden = "leiden"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


def run_layout(
    strategy: dict[str, Any], graphml_or_graph: str | nx.Graph
) -> Communities:
    """Run layout method definition."""
    graph = load_graph(graphml_or_graph)
    if len(graph.nodes) == 0:
        log.warning("Graph has no nodes")
        return []

    clusters: dict[int, dict[str, list[str]]] = {}
    strategy_type = strategy.get("type", GraphCommunityStrategyType.leiden)
    match strategy_type:
        case GraphCommunityStrategyType.leiden:
            from .strategies.leiden import run as run_leiden
            # 核心逻辑
            # 调用分层莱顿算法，对输入图进行分层和社区划分
            clusters = run_leiden(graph, strategy)
        case _:
            msg = f"Unknown clustering strategy {strategy_type}"
            raise ValueError(msg)

    # clusters格式：
    # {"层次 0" : {"社区 0" : [本社区实体列表], "社区 1" : [本社区实体列表]}}
    # {"层次 1" : {"社区 2" : [本社区实体列表], "社区 3" : [本社区实体列表]}}
    # 高层次中的社区，是对上一层社区的进一步划分，例如：
    # 社区2 和 社区3 中的实体都来自 社区1
    results: Communities = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level, cluster_id, nodes))
    return results
