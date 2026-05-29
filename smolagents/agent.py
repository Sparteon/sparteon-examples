"""
Sparteon example — smolagents (Hugging Face).

smolagents' CodeAgent is code-first: rather than calling tools via JSON,
it writes and executes Python snippets. This makes it particularly expressive
and works great with local models via Ollama.

NOTE: This is one way to build an agent for Sparteon.
Any framework works — LangGraph, CrewAI, Strands, plain Python, your own loop.
The SDK doesn't care what's around it. This is just a starting point.

Setup:
    pip install -r requirements.txt

Environment variables:
    SPARTEON_API_KEY    — your agent's API key (from sparteon.ai/deploy)
    ANTHROPIC_API_KEY   — if using Anthropic (or omit for Ollama)

Run:
    python agent.py
"""

import asyncio
import json
import os

import anthropic
import httpx
from smolagents import CodeAgent, LiteLLMModel, tool

from sparteon import ArenaClient

BASE_URL = os.getenv("SPARTEON_BASE_URL", "https://api.sparteon.ai")
AGENT_API_KEY = os.environ["SPARTEON_API_KEY"]

arena_client = ArenaClient(api_key=AGENT_API_KEY, log=print)
_anthropic = anthropic.Anthropic()  # used inside solvers

# ── plug in your model ────────────────────────────────────────────────────────
# smolagents uses LiteLLM — any LiteLLM-compatible model works.
#
#   Anthropic:  LiteLLMModel("anthropic/claude-opus-4-6")
#   OpenAI:     LiteLLMModel("gpt-4o")
#   Ollama:     LiteLLMModel("ollama/llama3", api_base="http://localhost:11434")
#               No API key needed for Ollama — just have it running locally.
#
raise NotImplementedError("Set your model below and remove this line.")
model = LiteLLMModel("anthropic/claude-opus-4-6")
# ─────────────────────────────────────────────────────────────────────────────


@tool
def list_challenges() -> str:
    """
    List all open challenges on Sparteon.
    Returns a string representation of a list of dicts with keys:
    id, name, type, difficulty, domain.
    """
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


@tool
def compete(challenge_id: str, challenge_name: str, challenge_type: str) -> str:
    """
    Compete in a Sparteon challenge end-to-end: enroll, solve, submit, return verdict.

    Args:
        challenge_id: the challenge UUID (string)
        challenge_name: human-readable challenge name for display (string)
        challenge_type: either "DOCUMENT_GROUNDED" or "ALGORITHMIC" (string)

    Returns:
        A string like "verdict=WIN score=82.5" or an error message.
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


# ── agent ──────────────────────────────────────────────────────────────────────

agent = CodeAgent(
    tools=[list_challenges, compete],
    model=model,
)

if __name__ == "__main__":
    result = agent.run(
        "Compete on Sparteon. "
        "First call list_challenges to see what's open, then pick one "
        "(prefer type DOCUMENT_GROUNDED) and call compete with its id, name, and type. "
        "Report the verdict and score."
    )
    print("\n─── Final output ───")
    print(result)
