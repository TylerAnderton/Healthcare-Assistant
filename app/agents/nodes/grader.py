from typing import Callable, Literal
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

_GRADER_SYSTEM_PROMPT = (
    "You are a strict answer quality judge for a healthcare data assistant.\n"
    "Evaluate the candidate answer against the original user query on two criteria:\n"
    "1. Relevance: Does the answer directly address what was asked?\n"
    "2. Completeness: Does the answer cover all parts of the question without major gaps?\n\n"
    "Respond with verdict='pass' only if BOTH criteria are met.\n"
    "Respond with verdict='fail' and a short reason (≤2 sentences) if either criterion fails.\n"
    "Do not evaluate factual accuracy — only relevance and completeness."
)


class GraderOutput(BaseModel):
    verdict: Literal["pass", "fail"]
    reason: str


def build_grader_node(llm) -> Callable[[dict], dict]:
    grader_llm = llm.with_structured_output(GraderOutput)

    def grader_node(state: dict) -> dict:
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        ai_msgs = [m for m in state["messages"] if isinstance(m, AIMessage)]
        query = human_msgs[-1].content if human_msgs else ""
        answer = ai_msgs[-1].content if ai_msgs else ""

        messages = [
            SystemMessage(content=_GRADER_SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {query}\n\nAnswer: {answer}"),
        ]
        result = grader_llm.invoke(messages)
        return {"grader_verdict": result.verdict, "grader_feedback": result.reason}

    return grader_node
