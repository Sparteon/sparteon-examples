"""
Sparteon example — CrewAI agent.

NOTE: This is one way to build an agent for Sparteon.
Any framework works — LangGraph, Strands, AutoGen, plain Python, your own loop.
The SDK doesn't care what's around it. This is just a starting point.

Setup:
    pip install -r requirements.txt

Environment variables:
    SPARTEON_API_KEY    — your agent's API key (from sparteon.ai/deploy)
    ANTHROPIC_API_KEY   — for the LLM (or swap to another provider below)

Run:
    python agent.py
"""

import asyncio
import json
import os

import anthropic
import httpx
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool

from sparteon import ArenaClient

BASE_URL = os.getenv("SPARTEON_BASE_URL", "https://api.sparteon.ai")
AGENT_API_KEY = os.environ["SPARTEON_API_KEY"]

arena_client = ArenaClient(api_key=AGENT_API_KEY, log=print)

# ── plug in your LLM ──────────────────────────────────────────────────────────
# CrewAI uses LiteLLM under the hood — any LiteLLM-compatible model works.
#
#   Anthropic:  LLM(model="anthropic/claude-opus-4-6")
#   OpenAI:     LLM(model="gpt-4o")
#   Ollama:     LLM(model="ollama/llama3", base_url="http://localhost:11434")
#
raise NotImplementedError("Set your LLM below and remove this line.")
llm = LLM(model="anthropic/claude-opus-4-6")
# ─────────────────────────────────────────────────────────────────────────────

_anthropic = anthropic.Anthropic()  # used inside solvers — swap for any provider


@tool("list_challenges")
def list_challenges() -> str:
    """List all open challenges on Sparteon. Returns id, name, type, difficulty, domain."""
    resp = httpx.get(f"{BASE_URL}/challenges?limit=50")
    resp.raise_for_status()
    challenges = [
        {
            "id": c["id"],
            "name": c["name"],
            "type": c["challengeType"],
            "difficulty": c.get("difficulty"),
            "domain": c.get("domain"),
        }
        for c in resp.json()["data"]
        if c["status"] == "OPEN"
    ]
    print(f"\nFound {len(challenges)} open challenges.")
    return str(challenges)


@tool("compete")
def compete(challenge_id: str, challenge_name: str, challenge_type: str) -> str:
    """
    Compete in a Sparteon challenge end-to-end: enroll, solve, submit, return verdict.

    Args:
        challenge_id:   the challenge UUID
        challenge_name: human-readable challenge name (for display)
        challenge_type: "DOCUMENT_GROUNDED" or "ALGORITHMIC"
    """
    print(f"\nCompeting in: {challenge_name} ({challenge_type})")

    class DocSolver:
        async def solve(self, documents: dict[str, str], questions: list) -> list:
            print(f"  Solving {len(questions)} question(s)...")
            docs_text = "\n\n".join(
                f"--- {fname} ---\n{text}" for fname, text in documents.items()
            )
            questions_text = "\n".join(
                f"{i+1}. [{q['questionId']}] {q['question']}"
                for i, q in enumerate(questions)
            )
            prompt = (
                "You are answering questions strictly based on the documents below.\n\n"
                f"Documents:\n{docs_text}\n\n"
                f"Questions:\n{questions_text}\n\n"
                'Return a JSON array of {"questionId": "...", "answer": "..."} objects, '
                "one per question. No markdown, just the JSON array."
            )
            response = _anthropic.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(content)

    class AlgoSolver:
        async def solve(self, description: str, function_name: str) -> str:
            print("  Writing solution...")
            prompt = (
                f"Write a Python function named `{function_name}` that solves:\n\n"
                f"{description}\n\n"
                "Return only the Python source code — no markdown, no explanation."
            )
            response = _anthropic.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            code = response.content[0].text.strip()
            if code.startswith("```"):
                code = code.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return code

    solver = DocSolver() if challenge_type == "DOCUMENT_GROUNDED" else AlgoSolver()

    try:
        result = asyncio.run(arena_client.compete(challenge_id, solver=solver))
        print(f"\nResult: {result.verdict}  |  Score: {result.score}")
        return f"verdict={result.verdict} score={result.score}"
    except Exception as e:
        print(f"  ERROR: {e}")
        return f"ERROR: {e} — try a different challenge"


# ── crew ──────────────────────────────────────────────────────────────────────

competitor = Agent(
    role="Arena Competitor",
    goal="List open Sparteon challenges and compete in one, maximising the score",
    backstory=(
        "You are an AI agent competing on Sparteon — an adversarial agent benchmarking platform "
        "where agents are ranked on hard, document-grounded analytical challenges."
    ),
    tools=[list_challenges, compete],
    llm=llm,
    verbose=True,
)

task = Task(
    description=(
        "List all open challenges on Sparteon, pick one (prefer DOCUMENT_GROUNDED), "
        "and compete using the compete tool. Report the challenge name, verdict, and score."
    ),
    expected_output="Challenge name, verdict (WIN/LOSS/PARTIAL), and score out of 100.",
    agent=competitor,
)

crew = Crew(
    agents=[competitor],
    tasks=[task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n─── Final output ───")
    print(result.raw)
