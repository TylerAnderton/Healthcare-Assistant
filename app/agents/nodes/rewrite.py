from typing import Callable
from langchain_core.messages import HumanMessage


def build_rewrite_node(llm) -> Callable[[dict], dict]:
    def rewrite_node(state: dict) -> dict:
        feedback_msg = HumanMessage(
            content=(
                f"Your previous answer was rejected by a quality check.\n"
                f"Feedback: {state['grader_feedback']}\n"
                f"Rewrite the answer addressing the feedback. "
                f"Use only information already retrieved — do not request new data."
            )
        )
        messages = list(state["messages"]) + [feedback_msg]
        response = llm.invoke(messages)
        return {
            "messages": [response],
            "rewrite_count": state["rewrite_count"] + 1,
        }

    return rewrite_node
