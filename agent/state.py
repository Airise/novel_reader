from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    """Agent 运行状态"""

    question: str
    contexts: List[str]
    steps: List[str]
    step_count: int
    max_steps: int

    current_action: Optional[Dict[str, Any]]
    should_continue: bool
    answer: Optional[str]

    searched_queries: List[str]
    no_gain_rounds: int
    max_no_gain_rounds: int
    planner_raw: str
    reflector_raw: str
    verifier_raw: str
    verifier_feedback: str
    verifier_feedback_tag: str
    verifier_tag_history: List[str]
    verify_pass: bool
    fast_mode: bool
    node_timings_ms: Dict[str, List[float]]
    retrieval_trace: List[Dict[str, Any]]
    source_filter: List[str]
    errors: List[str]

    query_type: str
    sub_questions: List[str]
    sub_question_results: List[Dict[str, Any]]
    retriever_status: Dict[str, Any]
    degraded_mode: bool
    chapter_candidates: List[str]