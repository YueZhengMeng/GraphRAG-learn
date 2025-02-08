# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run and _compute_leiden_communities methods definitions."""

import logging
from typing import Any

import networkx as nx
from graspologic.partition import hierarchical_leiden

from graphrag.index.graph.utils import stable_largest_connected_component

log = logging.getLogger(__name__)


def run(graph: nx.Graph, args: dict[str, Any]) -> dict[int, dict[str, list[str]]]:
    """Run method definition."""
    max_cluster_size = args.get("max_cluster_size", 10)
    use_lcc = args.get("use_lcc", True)
    if args.get("verbose", False):
        log.info(
            "Running leiden with max_cluster_size=%s, lcc=%s", max_cluster_size, use_lcc
        )
    # 核心逻辑
    # 调用层次化莱顿算法
    node_id_to_community_map = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=args.get("seed", 0xDEADBEEF),
    )
    levels = args.get("levels")

    # If they don't pass in levels, use them all
    if levels is None:
        levels = sorted(node_id_to_community_map.keys())

    # 返回分层与社区划分结果
    # 格式：
    # {"层次 0" : {"社区 0" : [本社区实体列表], "社区 1" : [本社区实体列表]}}
    # {"层次 1" : {"社区 2" : [本社区实体列表], "社区 3" : [本社区实体列表]}}
    # 高层次中的社区，是对上一层社区的进一步划分，例如：
    # 社区2 和 社区3 中的实体都来自 社区1
    results_by_level: dict[int, dict[str, list[str]]] = {}
    for level in levels:
        result = {}
        results_by_level[level] = result
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            community_id = str(raw_community_id)
            if community_id not in result:
                result[community_id] = []
            result[community_id].append(node_id)
    return results_by_level


# Taken from graph_intelligence & adapted
def _compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph,
    max_cluster_size: int,
    use_lcc: bool,
    seed=0xDEADBEEF,
) -> dict[int, dict[str, int]]:
    """Return Leiden root communities."""
    # 最大连通分量
    if use_lcc:
        graph = stable_largest_connected_component(graph)
    # 调库，执行层次化莱顿算法
    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed
    )
    results: dict[int, dict[str, int]] = {}
    # 取出分层与社区划分结果
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster

    return results
