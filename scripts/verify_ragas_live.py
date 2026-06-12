"""One-off verification: live graph run should populate ragas_scores fields.

Streams node-by-node updates so a hang can be pinpointed to a specific node.
"""
import logging
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.agents.react_agent import build_react_agent
from app.chains.prompts import SYSTEM_BASE
from app.constants import AGENT_MAX_ITERATIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.INFO)

QUERY = "What is my most recent ALT (SGPT) value?"

print("building graph...", flush=True)
graph = build_react_agent(ChatOllama(model="qwen3:4b"))
print("graph built, streaming...", flush=True)

final_state = {}
for update in graph.stream(
    {
        "messages": [SystemMessage(content=SYSTEM_BASE), HumanMessage(content=QUERY)],
        "query": QUERY,
        "sources": [],
        "retrieved_docs": [],
        "tool_outputs": [],
        "tool_validation_errors": [],
        "grader_verdict": "",
        "grader_feedback": "",
        "rewrite_count": 0,
        "iteration_count": 0,
        "ragas_scores": None,
        "ragas_rewrite_count": 0,
    },
    config={"recursion_limit": AGENT_MAX_ITERATIONS * 3},
    stream_mode="values",
):
    last = update["messages"][-1]
    print(
        f">>> step: last_msg={type(last).__name__} "
        f"iter={update.get('iteration_count')} "
        f"ragas_scores={update.get('ragas_scores')}",
        flush=True,
    )
    final_state = update

print("ragas_scores:", final_state.get("ragas_scores"))
print("ragas_rewrite_count:", final_state.get("ragas_rewrite_count"))
print("answer:", final_state["messages"][-1].content[:300])
