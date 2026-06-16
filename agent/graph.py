from langgraph.graph import END, StateGraph

from agent.nodes import (
    executor_node,
    generator_node,
    planner_node,
    reflector_node,
    router_node,
    verifier_node,
)
from agent.state import AgentState


def _route_after_router(state: AgentState) -> str:
    query_type = state.get("query_type", "fact")
    return "planner" if query_type in {"fact", "search", "causal"} else "executor"


def _route_after_planner(state: AgentState) -> str:
    action = (state.get("current_action") or {}).get("action")
    if action == "search":
        return "executor"
    if (not state.get("should_continue", True)) or state.get("step_count", 0) >= state.get("max_steps", 4):
        return "generator"
    if state.get("no_gain_rounds", 0) >= state.get("max_no_gain_rounds", 2):
        return "generator"
    return "reflector"


def _route_after_reflector(state: AgentState) -> str:
    if not state.get("should_continue"):
        return "generator"
    if state.get("no_gain_rounds", 0) >= state.get("max_no_gain_rounds", 2):
        return "generator"
    if state.get("step_count", 0) >= state.get("max_steps", 4):
        return "generator"
    return "planner"


def _route_after_verifier(state: AgentState) -> str:
    # 校验通过则结束；校验失败且还能继续则回 planner
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 4)
    verify_pass = state.get("verify_pass", False)

    if verify_pass:
        return END
    if step_count >= max_steps:
        return END
    return "planner"


def build_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reflector", reflector_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("verifier", verifier_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges("router", _route_after_router)
    workflow.add_conditional_edges("planner", _route_after_planner)
    workflow.add_edge("executor", "reflector")

    workflow.add_conditional_edges("reflector", _route_after_reflector)
    workflow.add_edge("generator", "verifier")
    workflow.add_conditional_edges("verifier", _route_after_verifier)

    return workflow.compile()
