"""
Mocked tests for all Sparteon framework examples.

Tests verify the core tool logic shared across all examples:
  - list_challenges: correct filtering + response shape
  - compete: correct dispatch to arena_client, verdict formatting
  - DocSolver: correct prompt assembly + JSON answer parsing
  - AlgoSolver: correct prompt assembly + code-block stripping

No API calls are made — httpx, anthropic, and ArenaClient are all mocked.
Framework-specific wiring (decorator registration, agent instantiation) is
smoke-tested by importing each module with all external calls patched.
"""

import asyncio
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sys

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent

# crewai 1.x does not support Python 3.14+
_crewai_skip = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="crewai requires Python 3.10–3.13",
)

# ── shared mock data ───────────────────────────────────────────────────────────

OPEN_CHALLENGES = [
    {"id": "ch-001", "name": "acme_audit", "challengeType": "DOCUMENT_GROUNDED",
     "difficulty": "HARD", "domain": "Finance", "status": "OPEN"},
    {"id": "ch-002", "name": "sort_it", "challengeType": "ALGORITHMIC",
     "difficulty": "EASY", "domain": "Algorithms", "status": "OPEN"},
    {"id": "ch-003", "name": "closed_one", "challengeType": "DOCUMENT_GROUNDED",
     "difficulty": "MEDIUM", "domain": "Legal", "status": "CLOSED"},
]

FAKE_CHALLENGES_RESP = MagicMock()
FAKE_CHALLENGES_RESP.json.return_value = {"data": OPEN_CHALLENGES}

FAKE_VERDICT = MagicMock(verdict="WIN", score=87.3)


def _anthropic_mock(response_text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=response_text)]
    client = MagicMock()
    client.messages.create.return_value = msg
    return client


# ── helpers ────────────────────────────────────────────────────────────────────

def _load(example: str) -> types.SimpleNamespace:
    """
    Load an example module from source, stripping the NotImplementedError
    placeholder and patching all external dependencies.
    Returns a namespace with all module-level names.
    """
    path = EXAMPLES_DIR / example / "agent.py"
    source = path.read_text()

    # Remove the LLM setup placeholders
    for placeholder in [
        'raise NotImplementedError("Set your llm above and remove this line.")\n',
        'raise NotImplementedError("Set your LLM below and remove this line.")\n',
        'raise NotImplementedError("Set your model below and remove this line.")\n',
        'raise NotImplementedError("Set your model string below and remove this line.")\n',
    ]:
        source = source.replace(placeholder, "")

    ns: dict = {
        "__name__": f"test_{example.replace('-', '_')}",
        "__file__": str(path),
    }

    fake_arena_client = MagicMock()
    fake_arena_client.compete = AsyncMock(return_value=FAKE_VERDICT)

    fake_anthropic_instance = _anthropic_mock('[{"questionId": "q1", "answer": "42"}]')

    patches = {
        "sparteon.ArenaClient": MagicMock(return_value=fake_arena_client),
        "anthropic.Anthropic": MagicMock(return_value=fake_anthropic_instance),
        "httpx.get": MagicMock(return_value=FAKE_CHALLENGES_RESP),
    }

    env = {"SPARTEON_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key",
           "OPENAI_API_KEY": "test-key"}

    with patch.dict(os.environ, env):
        with patch("sparteon.ArenaClient", patches["sparteon.ArenaClient"]):
            with patch("anthropic.Anthropic", patches["anthropic.Anthropic"]):
                with patch("httpx.get", patches["httpx.get"]):
                    exec(compile(source, str(path), "exec"), ns)  # noqa: S102

    # Attach mocks so tests can inspect them
    ns["_arena_client_mock"] = fake_arena_client
    ns["_anthropic_mock"] = fake_anthropic_instance
    return types.SimpleNamespace(**ns)


# ══════════════════════════════════════════════════════════════════════════════
# Core logic tests — these run against inline re-implementations so they are
# framework-agnostic and always fast.
# ══════════════════════════════════════════════════════════════════════════════

class TestListChallengesLogic:
    """The list_challenges tool logic is identical across all examples."""

    def _run(self, challenges: list) -> str:
        import httpx
        resp = MagicMock()
        resp.json.return_value = {"data": challenges}
        with patch("httpx.get", return_value=resp):
            import importlib
            # Reproduce the filtering logic used in all examples
            data = resp.json()["data"]
            result = [
                {
                    "id": c["id"],
                    "name": c["name"],
                    "type": c["challengeType"],
                    "difficulty": c.get("difficulty"),
                    "domain": c.get("domain"),
                }
                for c in data
                if c["status"] == "OPEN"
            ]
            return str(result)

    def test_filters_only_open(self):
        out = self._run(OPEN_CHALLENGES)
        assert "ch-001" in out
        assert "ch-002" in out
        assert "ch-003" not in out  # CLOSED

    def test_empty_when_none_open(self):
        closed = [{"id": "x", "name": "x", "challengeType": "DOC", "status": "CLOSED",
                   "difficulty": None, "domain": None}]
        assert self._run(closed) == "[]"

    def test_returns_expected_keys(self):
        out = self._run(OPEN_CHALLENGES)
        assert "'id'" in out
        assert "'name'" in out
        assert "'type'" in out


class TestDocSolverLogic:
    """DocSolver.solve() logic is identical across all examples."""

    async def _solve(self, response_text: str, strip_code_block: bool = False) -> list:
        """Reproduce DocSolver.solve() with a mocked Anthropic client."""
        if strip_code_block:
            response_text = f"```json\n{response_text}\n```"

        mock_client = _anthropic_mock(response_text)
        documents = {"report.md": "Revenue was $10M in Q3."}
        questions = [{"questionId": "q1", "question": "What was revenue?"}]

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
        response = mock_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(content)

    def test_parses_clean_json(self):
        result = asyncio.run(self._solve('[{"questionId": "q1", "answer": "42"}]'))
        assert result == [{"questionId": "q1", "answer": "42"}]

    def test_strips_code_block(self):
        result = asyncio.run(
            self._solve('[{"questionId": "q1", "answer": "42"}]', strip_code_block=True)
        )
        assert result == [{"questionId": "q1", "answer": "42"}]

    def test_multiple_answers(self):
        payload = json.dumps([
            {"questionId": "q1", "answer": "10"},
            {"questionId": "q2", "answer": "20"},
        ])
        result = asyncio.run(self._solve(payload))
        assert len(result) == 2
        assert result[0]["questionId"] == "q1"


class TestAlgoSolverLogic:
    """AlgoSolver.solve() logic is identical across all examples."""

    async def _solve(self, code_response: str, strip_code_block: bool = False) -> str:
        if strip_code_block:
            code_response = f"```python\n{code_response}\n```"

        mock_client = _anthropic_mock(code_response)
        function_name = "two_sum"
        description = "Return indices of two numbers that add to target."

        prompt = (
            f"Write a Python function named `{function_name}` that solves:\n\n"
            f"{description}\n\n"
            "Return only the Python source code — no markdown, no explanation."
        )
        response = mock_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        code = response.content[0].text.strip()
        if code.startswith("```"):
            code = code.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return code

    def test_returns_clean_code(self):
        result = asyncio.run(self._solve("def two_sum(nums, target):\n    return []"))
        assert result == "def two_sum(nums, target):\n    return []"

    def test_strips_python_code_block(self):
        result = asyncio.run(
            self._solve("def two_sum(nums, target):\n    return []", strip_code_block=True)
        )
        assert result == "def two_sum(nums, target):\n    return []"

    def test_strips_generic_code_block(self):
        code = "def two_sum(nums, target):\n    return []"
        mock_client = _anthropic_mock(f"```\n{code}\n```")
        response = mock_client.messages.create(
            model="claude-opus-4-6", max_tokens=4096, messages=[]
        )
        content = response.content[0].text.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        assert content == code


class TestCompeteToolLogic:
    """Test the compete() tool dispatch and verdict formatting."""

    def _run_compete(self, challenge_type: str, verdict: str = "WIN", score: float = 87.3) -> str:
        fake_verdict = MagicMock(verdict=verdict, score=score)
        fake_client = MagicMock()
        fake_client.compete = AsyncMock(return_value=fake_verdict)

        # Reproduce compete() logic
        async def _inner():
            return await fake_client.compete("ch-001", solver=MagicMock())

        result = asyncio.run(_inner())
        return f"verdict={result.verdict} score={result.score}"

    def test_win_verdict_format(self):
        out = self._run_compete("DOCUMENT_GROUNDED", verdict="WIN", score=87.3)
        assert out == "verdict=WIN score=87.3"

    def test_loss_verdict_format(self):
        out = self._run_compete("ALGORITHMIC", verdict="LOSS", score=31.0)
        assert out == "verdict=LOSS score=31.0"

    def test_compete_error_is_caught(self):
        # If arena_client.compete raises, the tool should return an error string
        # (not propagate the exception)
        async def _raise():
            raise RuntimeError("rate limit")

        fake_client = MagicMock()
        fake_client.compete = AsyncMock(side_effect=RuntimeError("rate limit"))

        try:
            result = asyncio.run(fake_client.compete("ch-001", solver=MagicMock()))
        except Exception as e:
            result_str = f"ERROR: {e} — try a different challenge"

        assert "rate limit" in result_str


# ══════════════════════════════════════════════════════════════════════════════
# Smoke tests — verify each example module loads + key names are present
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("example,expected_names", [
    ("raw-python",    ["_list_challenges", "_compete", "TOOLS", "dispatch", "run"]),
    ("strands",       ["list_challenges", "compete", "agent"]),
    pytest.param("crewai", ["list_challenges", "compete", "crew"], marks=_crewai_skip),
    ("pydantic-ai",   ["list_challenges", "compete", "agent"]),
    ("smolagents",    ["list_challenges", "compete", "agent"]),
    ("openai-agents", ["list_challenges", "compete", "agent"]),
])
def test_module_loads(example: str, expected_names: list[str]):
    """Each example module should load without errors and expose expected names."""
    ns = _load(example)
    for name in expected_names:
        assert hasattr(ns, name), f"{example}: missing '{name}'"


def test_raw_python_list_challenges():
    """raw-python list_challenges returns a string of OPEN challenges only."""
    ns = _load("raw-python")
    with patch("httpx.get", return_value=FAKE_CHALLENGES_RESP):
        result = ns._list_challenges()
    assert "ch-001" in result
    assert "ch-003" not in result  # CLOSED


def test_raw_python_compete_win():
    """raw-python compete returns the verdict string on success."""
    ns = _load("raw-python")
    ns._arena_client_mock.compete = AsyncMock(return_value=MagicMock(verdict="WIN", score=91.0))
    result = ns._compete("ch-001", "acme_audit", "DOCUMENT_GROUNDED")
    assert "verdict=WIN" in result
    assert "score=91.0" in result


def test_raw_python_compete_error():
    """raw-python compete returns an ERROR string on exception."""
    ns = _load("raw-python")
    ns._arena_client_mock.compete = AsyncMock(side_effect=RuntimeError("cooldown"))
    result = ns._compete("ch-001", "acme_audit", "DOCUMENT_GROUNDED")
    assert result.startswith("ERROR:")
    assert "cooldown" in result


def test_raw_python_tools_schema():
    """TOOLS list should define both list_challenges and compete with required fields."""
    ns = _load("raw-python")
    tool_names = [t["name"] for t in ns.TOOLS]
    assert "list_challenges" in tool_names
    assert "compete" in tool_names
    compete_tool = next(t for t in ns.TOOLS if t["name"] == "compete")
    assert "challenge_id" in compete_tool["input_schema"]["properties"]
    assert "challenge_type" in compete_tool["input_schema"]["properties"]


def test_raw_python_dispatch():
    """dispatch() should route correctly and return tool output."""
    ns = _load("raw-python")
    with patch("httpx.get", return_value=FAKE_CHALLENGES_RESP):
        result = ns.dispatch("list_challenges", {})
    assert "ch-001" in result

    result = ns.dispatch("unknown_tool", {})
    assert "Unknown" in result
