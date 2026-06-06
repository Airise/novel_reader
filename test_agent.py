import argparse

from agent.graph import build_agent_graph
from agent.state import AgentState


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_agent_graph()
    return _GRAPH


def run_agent(
    question: str,
    max_steps: int = 4,
    fast_mode: bool = False,
    source_filter=None,
):
    graph = _get_graph()
    initial_state: AgentState = {
        "question": question,
        "contexts": [],
        "steps": [],
        "step_count": 0,
        "max_steps": max_steps,
        "current_action": None,
        "answer": None,
        "should_continue": True,
        "searched_queries": [],
        "no_gain_rounds": 0,
        "planner_raw": "",
        "reflector_raw": "",
        "verifier_raw": "",
        "verifier_feedback": "",
        "verifier_feedback_tag": "unclear",
        "verifier_tag_history": [],
        "verify_pass": False,
        "fast_mode": fast_mode,
        "node_timings_ms": {
            "planner": [],
            "executor": [],
            "reflector": [],
            "generator": [],
            "verifier": [],
        },
        "retrieval_trace": [],
        "source_filter": source_filter or [],
        "errors": [],
    }
    final_state = graph.invoke(initial_state, config={"recursion_limit": 60})
    return final_state


def print_debug_info(result: AgentState):
    print("\n[DEBUG] retrieval_trace:")
    for item in result.get("retrieval_trace", []):
        print(f"  - {item}")

    if result.get("planner_raw"):
        print("\n[DEBUG] planner_raw:")
        print(result["planner_raw"])

    if result.get("reflector_raw"):
        print("\n[DEBUG] reflector_raw:")
        print(result["reflector_raw"])

    if result.get("verifier_raw"):
        print("\n[DEBUG] verifier_raw:")
        print(result["verifier_raw"])

    if result.get("verifier_feedback"):
        print("\n[DEBUG] verifier_feedback:")
        print(result["verifier_feedback"])

    if result.get("verifier_feedback_tag"):
        print("\n[DEBUG] verifier_feedback_tag:")
        print(result["verifier_feedback_tag"])

    if result.get("verifier_tag_history"):
        print("\n[DEBUG] verifier_tag_history:")
        print(result["verifier_tag_history"])

    if result.get("errors"):
        print("\n[DEBUG] errors:")
        for err in result["errors"]:
            print(f"  - {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    while True:
        question = input("\n请输入问题（输入 q 退出）：").strip()
        if question.lower() == "q":
            break

        result = run_agent(question, max_steps=args.max_steps, fast_mode=args.fast_mode)
        print("\n最终答案：", result["answer"])
        print("\n检索步骤：")
        for step in result["steps"]:
            print("  -", step)

        if args.debug:
            print_debug_info(result)