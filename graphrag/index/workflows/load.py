# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing load_workflows, create_workflow, _get_steps_for_workflow and _remove_disabled_steps methods definition."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from datashaper import Workflow

from graphrag.index.errors import (
    NoWorkflowsDefinedError,
    UndefinedWorkflowError,
    UnknownWorkflowError,
)
from graphrag.index.utils import topological_sort

from .default_workflows import default_workflows as _default_workflows
from .typing import VerbDefinitions, WorkflowDefinitions, WorkflowToRun

if TYPE_CHECKING:
    from graphrag.index.config import (
        PipelineWorkflowConfig,
        PipelineWorkflowReference,
        PipelineWorkflowStep,
    )

anonymous_workflow_count = 0

VerbFn = Callable[..., Any]
log = logging.getLogger(__name__)


class LoadWorkflowResult(NamedTuple):
    """A workflow loading result object."""

    workflows: list[WorkflowToRun]
    """The loaded workflow names in the order they should be run."""

    dependencies: dict[str, list[str]]
    """A dictionary of workflow name to workflow dependencies."""


def load_workflows(
    workflows_to_load: list[PipelineWorkflowReference],
    additional_verbs: VerbDefinitions | None = None,
    additional_workflows: WorkflowDefinitions | None = None,
    memory_profile: bool = False,
) -> LoadWorkflowResult:
    """Load the given workflows.

    Args:
        - workflows_to_load - The workflows to load
        - additional_verbs - The list of custom verbs available to the workflows
        - additional_workflows - The list of custom workflows
    Returns:
        - output[0] - The loaded workflow names in the order they should be run
        - output[1] - A dictionary of workflow name to workflow dependencies
    """
    workflow_graph: dict[str, WorkflowToRun] = {}

    global anonymous_workflow_count
    for reference in workflows_to_load:
        name = reference.name
        is_anonymous = name is None or name.strip() == ""
        if is_anonymous:
            name = f"Anonymous Workflow {anonymous_workflow_count}"
            anonymous_workflow_count += 1
        name = cast(str, name)

        # reference: name, config
        config = reference.config

        # 核心逻辑
        # 根据workflow name创建工作流
        # workflow对象的核心是workflow._schema['name']和workflow._schema['steps']
        # 详细注释见create_workflow函数内容
        workflow = create_workflow(
            name or "MISSING NAME!",
            reference.steps,
            config,
            additional_verbs,
            additional_workflows,
        )
        workflow_graph[name] = WorkflowToRun(workflow, config=config or {})

    # Backfill any missing workflows
    # 检查每个workflow的拓扑前置依赖，如果前置依赖workflow不存在，则创建该前置依赖workflow
    for name in list(workflow_graph.keys()):
        workflow = workflow_graph[name]
        deps = [
            d.replace("workflow:", "")
            for d in workflow.workflow.dependencies
            if d.startswith("workflow:")
        ]
        for dependency in deps:
            if dependency not in workflow_graph:
                reference = {"name": dependency, **workflow.config}
                workflow_graph[dependency] = WorkflowToRun(
                    workflow=create_workflow(
                        dependency,
                        config=reference,
                        additional_verbs=additional_verbs,
                        additional_workflows=additional_workflows,
                        memory_profile=memory_profile,
                    ),
                    config=reference,
                )

    # Run workflows in order of dependencies
    # 如果某个workflow的前置依赖workflow不存在，则删除该前置依赖workflow
    def filter_wf_dependencies(name: str) -> list[str]:
        externals = [
            e.replace("workflow:", "")
            for e in workflow_graph[name].workflow.dependencies
        ]
        return [e for e in externals if e in workflow_graph]

    task_graph = {name: filter_wf_dependencies(name) for name in workflow_graph}
    # 拓扑排序
    workflow_run_order = topological_sort(task_graph)
    # 根据拓扑排序结果，获取workflow的运行顺序
    workflows = [workflow_graph[name] for name in workflow_run_order]
    log.info("Workflow Run Order: %s", workflow_run_order)
    return LoadWorkflowResult(workflows=workflows, dependencies=task_graph)


def create_workflow(
    name: str,
    steps: list[PipelineWorkflowStep] | None = None,
    config: PipelineWorkflowConfig | None = None,
    additional_verbs: VerbDefinitions | None = None,
    additional_workflows: WorkflowDefinitions | None = None,
    memory_profile: bool = False,
) -> Workflow:
    """Create a workflow from the given config."""
    # 核心逻辑
    # additional_workflows是所有的workflow，包括默认的和用户自定义的
    # 默认的workflow汇总见./default_workflows.py
    # 具体每个默认workflow的源码见./v1/xxx.py
    additional_workflows = {
        **_default_workflows,
        **(additional_workflows or {}),
    }
    # 核心逻辑
    # 每个具体workflow的源码中，都用字典类型定义了一系列步骤（step），保存在一个列表中
    # step的格式为{"verb": "xxx", "args": {"xxx": "xxx"}, "input": {"xxx": "xxx"},"enabled": "xxx", ......}
    # 其中verb是datashaper库中的注册的函数名。该库会根据verb字段，自动调度@verb装饰器注册的函数。
    # 具体函数源码见graphrag/index/verbs/xxx.py
    # _get_steps_for_workflow函数中，会根据workflow name，取出该workflow的steps
    # 并根据config装填参数。其中部分step会根据条件被设置为disabled
    steps = steps or _get_steps_for_workflow(name, config, additional_workflows)
    # _remove_disabled_steps会过滤掉该workflow中disabled的步骤
    steps = _remove_disabled_steps(steps)
    return Workflow(
        verbs=additional_verbs or {},
        schema={
            "name": name,
            "steps": steps,
        },
        validate=False,
        memory_profile=memory_profile,
    )


def _get_steps_for_workflow(
    name: str | None,
    config: PipelineWorkflowConfig | None,
    workflows: dict[str, Callable] | None,
) -> list[PipelineWorkflowStep]:
    """Get the steps for the given workflow config."""
    if config is not None and "steps" in config:
        return config["steps"]

    if workflows is None:
        raise NoWorkflowsDefinedError

    if name is None:
        raise UndefinedWorkflowError

    if name not in workflows:
        raise UnknownWorkflowError(name)

    return workflows[name](config or {})


def _remove_disabled_steps(
    steps: list[PipelineWorkflowStep],
) -> list[PipelineWorkflowStep]:
    return [step for step in steps if step.get("enabled", True)]
