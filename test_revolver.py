"""
Unit tests for Revolver System:
- Phase 1: Staggered Hedged Racing, EWMA Adaptive Delay, Telemetry
- Phase 2: Native Async Engine (`AsyncRevolver` via asyncio + httpx)
- Phase 3: Streaming Race by TTFT (Time-To-First-Token)
- Phase 4: Complexity-Aware Model Tiering (sprint/heavy)
- Phase 5: JSON Schema Judge + Prometheus Metrics Export
"""

import asyncio
import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock

from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from revolver import (
    DEFAULT_HEDGE_DELAY,
    MAX_HEDGE_DELAY,
    MIN_HEDGE_DELAY,
    AsyncRevolver,
    Revolver,
    TIER_HEAVY,
    TIER_SPRINT,
)


def _mock_response(status_code=200, content="ok", usage=None, text="", raw_json=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if raw_json is not None:
        payload = raw_json
    elif status_code == 200:
        payload = {"choices": [{"message": {"content": content}}]}
        if usage is not None:
            payload["usage"] = usage
    else:
        payload = {}
    resp.json.return_value = payload
    return resp


class TestRevolverPhase1(unittest.TestCase):
    def setUp(self):
        self.rev = Revolver(timeout=5.0, hedge_delay=0.25)
        self.rev.add_chamber("C1", "https://example.com/c1", "k1", "model-1")
        self.rev.add_chamber("C2", "https://example.com/c2", "k2", "model-2")

    def tearDown(self):
        self.rev.close()

    def test_fast_primary_saves_backup_call(self):
        """Primary finishing before hedge_delay never fires the backup chamber."""
        called = []

        def fake_post(url, headers=None, json=None, timeout=None):
            called.append(url)
            time.sleep(0.02)
            return _mock_response(
                200,
                content="Fast primary answer",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )

        self.rev._session.post = fake_post
        reply = self.rev.ask("Hello")

        self.assertEqual(reply, "Fast primary answer")
        self.assertEqual(called, ["https://example.com/c1"])

        metrics = self.rev.get_metrics()
        self.assertEqual(metrics["hedged_saved_calls"], 1)
        self.assertEqual(metrics["hedge_triggers"], 0)
        self.assertEqual(metrics["calls_saved"], 1)
        self.assertEqual(metrics["hedge_efficiency_pct"], 100.0)
        self.assertEqual(metrics["total_prompt_tokens"], 10)
        self.assertEqual(metrics["total_completion_tokens"], 5)

    def test_slow_primary_triggers_backup_and_kills_tail_latency(self):
        """Sluggish primary past hedge_delay triggers backup, which wins fast."""
        called = []
        lock = threading.Lock()

        def fake_post(url, headers=None, json=None, timeout=None):
            with lock:
                called.append(url)
            if url.endswith("/c1"):
                time.sleep(0.7)
                return _mock_response(200, content="Slow primary")
            time.sleep(0.03)
            return _mock_response(200, content="Fast backup winner")

        self.rev._session.post = fake_post
        started = time.monotonic()
        reply = self.rev.ask("Hello")
        elapsed = time.monotonic() - started

        self.assertEqual(reply, "Fast backup winner")
        self.assertEqual(len(called), 2)
        self.assertLess(elapsed, 0.5, "Hedged backup should prevent waiting for slow primary")

        metrics = self.rev.get_metrics()
        self.assertEqual(metrics["hedge_triggers"], 1)
        self.assertEqual(metrics["hedged_saved_calls"], 0)

        time.sleep(0.55)
        statuses = {s["name"]: s for s in self.rev.get_status()}
        self.assertEqual(statuses["C1"]["successes"], 1)
        self.assertEqual(statuses["C2"]["successes"], 1)

    def test_fast_failing_primary_immediately_fires_backup(self):
        """If primary fails early (429), backup fires immediately instead of waiting hedge_delay."""
        self.rev.hedge_delay = 1.5
        called = []

        def fake_post(url, headers=None, json=None, timeout=None):
            called.append(url)
            if url.endswith("/c1"):
                time.sleep(0.02)
                return _mock_response(429, text="rate limit")
            time.sleep(0.02)
            return _mock_response(200, content="Backup saved the day")

        self.rev._session.post = fake_post
        started = time.monotonic()
        reply = self.rev.ask("Hello")
        elapsed = time.monotonic() - started

        self.assertEqual(reply, "Backup saved the day")
        self.assertEqual(len(called), 2)
        self.assertLess(elapsed, 0.4, "Should not wait full 1.5s hedge_delay after fast 429 failure")

        statuses = {s["name"]: s for s in self.rev.get_status()}
        self.assertEqual(statuses["C1"]["status"], "cooling")
        self.assertEqual(statuses["C1"]["failures"], 1)
        self.assertLess(statuses["C1"]["health"], 60.0)

    def test_concurrent_mode_fires_both_immediately(self):
        """Explicit concurrent mode launches both chambers at t=0."""
        called = []
        lock = threading.Lock()

        def fake_post(url, headers=None, json=None, timeout=None):
            with lock:
                called.append(url)
            time.sleep(0.05)
            return _mock_response(200, content="Concurrent reply")

        self.rev._session.post = fake_post
        reply = self.rev.ask("Race now", race_mode="concurrent")

        self.assertEqual(reply, "Concurrent reply")
        self.assertEqual(len(called), 2)
        self.assertEqual(self.rev.get_metrics()["concurrent_races"], 1)

    def test_adaptive_hedge_delay_ewma(self):
        """Adaptive Revolver learns primary EWMA latency and clamps hedge delay cleanly."""
        adaptive_rev = Revolver(
            timeout=5.0,
            hedge_delay=None,
            min_hedge_delay=MIN_HEDGE_DELAY,
            max_hedge_delay=MAX_HEDGE_DELAY,
            hedge_multiplier=1.15,
        )
        try:
            adaptive_rev.add_chamber("FastA", "https://example.com/a", "k", "m1")
            c1 = adaptive_rev._chambers[0]

            self.assertEqual(adaptive_rev.get_hedge_delay(c1), DEFAULT_HEDGE_DELAY)

            adaptive_rev._apply_outcome(c1, ("ok", 0.2))
            self.assertEqual(adaptive_rev.get_hedge_delay(c1), MIN_HEDGE_DELAY)

            for _ in range(10):
                adaptive_rev._apply_outcome(c1, ("ok", 1.5))
            delay = adaptive_rev.get_hedge_delay(c1)
            self.assertGreater(delay, 1.0)
            self.assertLessEqual(delay, MAX_HEDGE_DELAY)

            for _ in range(15):
                adaptive_rev._apply_outcome(c1, ("ok", 10.0))
            self.assertEqual(adaptive_rev.get_hedge_delay(c1), MAX_HEDGE_DELAY)
        finally:
            adaptive_rev.close()

    def test_unfired_backup_runs_if_fast_primary_unrepairable(self):
        """If fast primary returns unrepairable garbage and Judge fails, unfired C2 must run in Tier 2."""
        called = []

        def fake_post(url, headers=None, json=None, timeout=None):
            called.append(url)
            if url.endswith("/c1"):
                return _mock_response(200, content="[truncated]")
            msgs = json.get("messages", [])
            if any("Draft to repair:" in m.get("content", "") for m in msgs):
                return _mock_response(500, text="judge error")
            return _mock_response(200, content="Clean Tier 2 response from C2")

        self.rev._session.post = fake_post
        reply = self.rev.ask("Give clean answer")

        self.assertEqual(reply, "Clean Tier 2 response from C2")
        self.assertIn("https://example.com/c2", called)

    def test_sanitizer_and_judge_repair(self):
        """Sanitizer strips tables/glitches and Judge repairs corrupted drafts."""
        dirty = "Range 1?2\u20113\u00a0ok\n| A | B |\n|---|---|\n| 1 | 2 |\nFinal line."
        cleaned = self.rev.sanitize_draft(dirty)
        self.assertEqual(cleaned, "Range 1-2-3 ok\nFinal line.")

        def fake_post(url, headers=None, json=None, timeout=None):
            msgs = json.get("messages", [])
            if any("Draft to repair:" in m.get("content", "") for m in msgs):
                return _mock_response(200, content="Repaired clean response")
            return _mock_response(200, content="As an AI language model I cannot answer [truncated]")

        self.rev._session.post = fake_post
        reply = self.rev.ask("Test judge")
        self.assertEqual(reply, "Repaired clean response")
        self.assertEqual(self.rev.get_metrics()["judge_repairs"], 1)

    def test_invalid_race_mode_raises(self):
        with self.assertRaises(ValueError):
            Revolver(race_mode="invalid_mode")

    def test_concurrent_ask_thread_safety(self):
        """Multiple concurrent ask() calls do not corrupt health or metrics."""
        self.rev.hedge_delay = 0.2

        def fake_post(url, headers=None, json=None, timeout=None):
            time.sleep(0.01)
            return _mock_response(200, content="Concurrent ok", usage={"prompt_tokens": 2, "completion_tokens": 3})

        self.rev._session.post = fake_post

        results = []
        errors = []

        def worker():
            try:
                results.append(self.rev.ask("Ping"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 10)
        self.assertTrue(all(r == "Concurrent ok" for r in results))
        metrics = self.rev.get_metrics()
        self.assertEqual(metrics["total_requests"], 10)
        self.assertEqual(metrics["hedged_saved_calls"], 10)


class TestAsyncRevolverPhase2(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.rev = AsyncRevolver(timeout=5.0, hedge_delay=0.15)
        self.rev.add_chamber("C1", "https://example.com/c1", "k1", "model-1")
        self.rev.add_chamber("C2", "https://example.com/c2", "k2", "model-2")

    async def asyncTearDown(self):
        await self.rev.aclose()

    def _inject_mock_client(self, post_fn, get_fn=None):
        mock_client = MagicMock()
        mock_client.is_closed = False
        mock_client.post = post_fn
        if get_fn is not None:
            mock_client.get = get_fn
        async def fake_aclose():
            mock_client.is_closed = True
        mock_client.aclose = fake_aclose
        self.rev._client = mock_client
        self.rev._client_loop = None

    async def test_async_fast_primary_saves_backup(self):
        called = []

        async def fake_post(url, headers=None, json=None, timeout=None):
            called.append(url)
            await asyncio.sleep(0.01)
            return _mock_response(
                200,
                content="Async fast answer",
                usage={"prompt_tokens": 12, "completion_tokens": 8},
            )

        self._inject_mock_client(fake_post)
        reply = await self.rev.ask("Hello async")

        self.assertEqual(reply, "Async fast answer")
        self.assertEqual(called, ["https://example.com/c1"])
        metrics = self.rev.get_metrics()
        self.assertEqual(metrics["hedged_saved_calls"], 1)
        self.assertEqual(metrics["total_prompt_tokens"], 12)
        self.assertEqual(metrics["total_completion_tokens"], 8)

    async def test_async_slow_primary_cancels_loser_coroutine(self):
        """Sluggish primary triggers backup; once backup wins, slow primary coroutine is cancelled."""
        cancelled_primary = False

        async def fake_post(url, headers=None, json=None, timeout=None):
            nonlocal cancelled_primary
            if url.endswith("/c1"):
                try:
                    await asyncio.sleep(2.0)
                    return _mock_response(200, content="Too slow")
                except asyncio.CancelledError:
                    cancelled_primary = True
                    raise
            await asyncio.sleep(0.01)
            return _mock_response(200, content="Async backup winner")

        self._inject_mock_client(fake_post)
        started = time.monotonic()
        reply = await self.rev.ask("Race async")
        elapsed = time.monotonic() - started

        self.assertEqual(reply, "Async backup winner")
        self.assertLess(elapsed, 0.4)
        self.assertTrue(cancelled_primary, "Losing asyncio task must be cancelled immediately")
        self.assertEqual(self.rev.get_metrics()["hedge_triggers"], 1)

    async def test_async_fast_fail_fires_backup_immediately(self):
        self.rev.hedge_delay = 1.5
        called = []

        async def fake_post(url, headers=None, json=None, timeout=None):
            called.append(url)
            if url.endswith("/c1"):
                await asyncio.sleep(0.01)
                return _mock_response(429, text="rate limit")
            await asyncio.sleep(0.01)
            return _mock_response(200, content="Async backup rescued")

        self._inject_mock_client(fake_post)
        started = time.monotonic()
        reply = await self.rev.ask("Fast fail test")
        elapsed = time.monotonic() - started

        self.assertEqual(reply, "Async backup rescued")
        self.assertEqual(len(called), 2)
        self.assertLess(elapsed, 0.3)

    async def test_async_concurrent_mode_cancels_loser(self):
        cancelled = []

        async def fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/c1"):
                try:
                    await asyncio.sleep(2.0)
                    return _mock_response(200, content="Slow c1")
                except asyncio.CancelledError:
                    cancelled.append("c1")
                    raise
            await asyncio.sleep(0.01)
            return _mock_response(200, content="Fast c2")

        self._inject_mock_client(fake_post)
        reply = await self.rev.ask("Go", race_mode="concurrent")

        self.assertEqual(reply, "Fast c2")
        self.assertEqual(cancelled, ["c1"])
        self.assertEqual(self.rev.get_metrics()["concurrent_races"], 1)

    async def test_async_unfired_backup_fallback_and_judge(self):
        """Unfired C2 runs in Tier 2 when C1 fails validation and Judge cannot repair."""
        called = []

        async def fake_post(url, headers=None, json=None, timeout=None):
            called.append(url)
            if url.endswith("/c1"):
                return _mock_response(200, content="[truncated]")
            msgs = json.get("messages", [])
            if any("Draft to repair:" in m.get("content", "") for m in msgs):
                return _mock_response(500, text="judge fail")
            return _mock_response(200, content="Clean async Tier 2")

        self._inject_mock_client(fake_post)
        reply = await self.rev.ask("Fallback test")
        self.assertEqual(reply, "Clean async Tier 2")
        self.assertIn("https://example.com/c2", called)

    async def test_async_judge_repair(self):
        async def fake_post(url, headers=None, json=None, timeout=None):
            msgs = json.get("messages", [])
            if any("Draft to repair:" in m.get("content", "") for m in msgs):
                return _mock_response(200, content="Async repaired answer")
            return _mock_response(200, content="system prompt leak")

        self._inject_mock_client(fake_post)
        reply = await self.rev.ask("Judge async")
        self.assertEqual(reply, "Async repaired answer")
        self.assertEqual(self.rev.get_metrics()["judge_repairs"], 1)

    async def test_async_provider_discovery(self):
        async def fake_get(url, headers=None, timeout=None):
            return _mock_response(
                200,
                raw_json={
                    "data": [
                        {"id": "llama-3.3-70b-versatile"},
                        {"id": "whisper-large-v3"},
                        {"id": "openai/gpt-oss-120b"},
                    ]
                },
            )

        self._inject_mock_client(MagicMock(), get_fn=fake_get)
        fresh = AsyncRevolver()
        try:
            fresh._client = self.rev._client
            fresh._client_loop = None
            chambers = await fresh.aadd_provider("groq", api_key="k", discover_models=True)
            self.assertEqual(chambers, ["Groq llama-3.3-70b-versatile", "Groq openai/gpt-oss-120b"])
        finally:
            fresh._client = None  # shared mock client is closed by tearDown

    async def test_async_high_concurrency_stress(self):
        """50 simultaneous async ask() calls run cleanly without contention or dropped results."""
        self.rev.hedge_delay = 0.3

        async def fake_post(url, headers=None, json=None, timeout=None):
            await asyncio.sleep(0.01)
            return _mock_response(200, content="High concurrency ok")

        self._inject_mock_client(fake_post)
        replies = await asyncio.gather(*(self.rev.ask(f"Prompt {i}") for i in range(50)))

        self.assertEqual(len(replies), 50)
        self.assertTrue(all(r == "High concurrency ok" for r in replies))
        metrics = self.rev.get_metrics()
        self.assertEqual(metrics["total_requests"], 50)
        self.assertEqual(metrics["hedged_saved_calls"], 50)

    async def test_async_context_manager_lifecycle(self):
        async with AsyncRevolver(max_keepalive_connections=15, max_connections=60) as rev:
            self.assertIsNotNone(rev._client)
            self.assertFalse(rev._client.is_closed)
        self.assertIsNone(rev._client)


def _sse(content: str) -> str:
    return 'data: {"choices":[{"delta":{"content":"%s"}}]}' % content


_SSE_DONE = "data: [DONE]"


class _TimedSseResponse:
    def __init__(self, items):
        self._items = items
        self.closed = False

    def iter_lines(self, decode_unicode=True):
        for delay, line in self._items:
            if delay:
                time.sleep(delay)
            yield line

    def close(self):
        self.closed = True


class TestPhase4Tiering(unittest.TestCase):
    def test_classify_prompt(self):
        self.assertEqual(Revolver.classify_prompt("say hi"), TIER_SPRINT)
        self.assertEqual(Revolver.classify_prompt(""), TIER_SPRINT)
        self.assertEqual(Revolver.classify_prompt("x" * 600), TIER_HEAVY)
        self.assertEqual(Revolver.classify_prompt("```python\nprint(1)\n```"), TIER_HEAVY)
        self.assertEqual(Revolver.classify_prompt("prove this theorem step by step"), TIER_HEAVY)

    def test_select_racers_tiered(self):
        rev = Revolver()
        rev.add_chamber("Sprint1", "u1", "k", "m1", tier=TIER_SPRINT)
        rev.add_chamber("Sprint2", "u2", "k", "m2", tier=TIER_SPRINT)
        rev.add_chamber("Heavy1", "u3", "k", "m3", tier=TIER_HEAVY)
        healthy = list(rev._chambers)
        sprint = rev._select_racers(healthy, [], TIER_SPRINT)
        self.assertEqual([r["name"] for r in sprint], ["Sprint1", "Sprint2"])
        heavy = rev._select_racers(healthy, [], TIER_HEAVY)
        self.assertEqual([r["name"] for r in heavy], ["Heavy1"])
        rev.close()

    def test_add_provider_tier(self):
        rev = Revolver()
        rev.add_provider("groq", api_key="k", models=["llama-3.3-70b-versatile"], tier=TIER_HEAVY)
        self.assertTrue(all(c["tier"] == TIER_HEAVY for c in rev._chambers))
        rev.close()


class TestPhase3StreamingSync(unittest.TestCase):
    def setUp(self):
        self.rev = Revolver(timeout=1.0, hedge_delay=0.05)
        self.rev.add_chamber("C1", "https://example.com/c1", "k1", "m1")
        self.rev.add_chamber("C2", "https://example.com/c2", "k2", "m2")

    def tearDown(self):
        self.rev.close()

    def test_ttft_winner_claims_stream(self):
        c1_items = [
            (0.01, _sse("A")),
            (0.01, _sse("B")),
            (0, _SSE_DONE),
        ]
        c2_items = [
            (0.2, _sse("C")),
            (0, _SSE_DONE),
        ]

        def fake_post(url, headers=None, json=None, timeout=None, stream=None):
            items = c1_items if url.endswith("/c1") else c2_items
            return _TimedSseResponse(items)

        self.rev._session.post = fake_post
        gen = self.rev.ask("Hello", stream=True)
        chunks = list(gen)
        self.assertEqual(chunks, ["A", "B"])
        self.assertNotIn("C", chunks)

    def test_single_chamber_streams(self):
        self.rev._chambers = self.rev._chambers[:1]
        items = [(0.01, _sse("Solo")), (0, _SSE_DONE)]

        def fake_post(url, headers=None, json=None, timeout=None, stream=None):
            return _TimedSseResponse(items)

        self.rev._session.post = fake_post
        gen = self.rev.ask("Hi", stream=True)
        self.assertEqual(list(gen), ["Solo"])

    def test_parse_sse_delta(self):
        self.assertEqual(Revolver._parse_sse_delta(_sse("hello")), "hello")
        self.assertIsNone(Revolver._parse_sse_delta(_SSE_DONE))
        self.assertIsNone(Revolver._parse_sse_delta(": comment line"))
        self.assertIsNone(Revolver._parse_sse_delta("data: not-json"))


class TestPhase3StreamingAsync(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.rev = AsyncRevolver(timeout=1.0, hedge_delay=0.05)
        self.rev.add_chamber("C1", "https://example.com/c1", "k1", "m1")
        self.rev.add_chamber("C2", "https://example.com/c2", "k2", "m2")

    async def asyncTearDown(self):
        await self.rev.aclose()

    def _inject_stream_client(self, c1_items, c2_items):
        def fake_stream(method, url, headers=None, json=None, timeout=None):
            items = c1_items if url.endswith("/c1") else c2_items
            return _AsyncSseCM(items)

        mock_client = MagicMock()
        mock_client.is_closed = False
        mock_client.stream = fake_stream

        async def fake_aclose():
            mock_client.is_closed = True

        mock_client.aclose = fake_aclose
        self.rev._client = mock_client
        self.rev._client_loop = None

    async def test_async_ttft_winner_claims_stream(self):
        c1_items = [(0.01, _sse("A")), (0.01, _sse("B")), (0, _SSE_DONE)]
        c2_items = [(0.2, _sse("C")), (0, _SSE_DONE)]
        self._inject_stream_client(c1_items, c2_items)
        gen = await self.rev.ask("Hello", stream=True)
        chunks = []
        async for chunk in gen:
            chunks.append(chunk)
        self.assertEqual(chunks, ["A", "B"])
        self.assertNotIn("C", chunks)


class _AsyncSseCM:
    def __init__(self, items):
        self._items = items

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def aiter_lines(self):
        for delay, line in self._items:
            if delay:
                await asyncio.sleep(delay)
            yield line


class TestPhase5Schema(unittest.TestCase):
    class Reply(BaseModel):
        answer: str
        score: int

    def setUp(self):
        self.rev = Revolver()
        self.rev.add_chamber("C1", "https://example.com/c1", "k1", "m1")
        self.rev.add_chamber("C2", "https://example.com/c2", "k2", "m2")

    def tearDown(self):
        self.rev.close()

    def test_validate_schema_valid(self):
        obj, err = self.rev._validate_schema('{"answer": "ok", "score": 5}', self.Reply)
        self.assertIsNone(err)
        self.assertEqual(obj.answer, "ok")
        self.assertEqual(obj.score, 5)

    def test_validate_schema_invalid(self):
        obj, err = self.rev._validate_schema("{bad json", self.Reply)
        self.assertIsNone(obj)
        self.assertIsNotNone(err)

    def test_schema_judge_repair(self):
        def fake_post(url, headers=None, json=None, timeout=None):
            msgs = json.get("messages", [])
            joined = " ".join(m.get("content", "") for m in msgs)
            if "Invalid draft to repair:" in joined:
                return _mock_response(
                    200,
                    content='{"answer": "repaired", "score": 9}',
                    usage={"prompt_tokens": 3, "completion_tokens": 2},
                )
            return _mock_response(200, content="{invalid json")

        self.rev._session.post = fake_post
        result = self.rev._apply_schema_validation(self.rev._chambers, "{invalid json", self.Reply)
        self.assertEqual(result, '{"answer": "repaired", "score": 9}')
        self.assertEqual(self.rev.get_metrics()["schema_repairs"], 1)

    def test_ask_response_format(self):
        def fake_post(url, headers=None, json=None, timeout=None):
            msgs = json.get("messages", [])
            joined = " ".join(m.get("content", "") for m in msgs)
            if "Invalid draft to repair:" in joined:
                return _mock_response(
                    200,
                    content='{"answer": "ok", "score": 1}',
                    usage={"prompt_tokens": 2, "completion_tokens": 2},
                )
            return _mock_response(200, content="{bad")

        self.rev._session.post = fake_post
        reply = self.rev.ask("x", response_format=self.Reply)
        self.assertEqual(reply, '{"answer": "ok", "score": 1}')

    def test_cost_accounting(self):
        self.rev.cost_rates = {"m1": (1.0, 2.0), "m2": (0.5, 0.5)}
        c1 = self.rev._chambers[0]
        self.rev._apply_outcome(c1, ("ok", 0.5, {"prompt_tokens": 1000, "completion_tokens": 1000}))
        self.assertEqual(self.rev._chamber_state[c1["name"]]["cost_usd"], 3.0)
        self.assertEqual(self.rev.get_metrics()["total_cost_usd"], 3.0)

    def test_get_status_includes_cost(self):
        self.rev.cost_rates = {"m1": (1.0, 2.0)}
        c1 = self.rev._chambers[0]
        self.rev._apply_outcome(c1, ("ok", 0.5, {"prompt_tokens": 1000, "completion_tokens": 1000}))
        st = {s["name"]: s for s in self.rev.get_status()}
        self.assertIn("cost_usd", st["C1"])
        self.assertEqual(st["C1"]["cost_usd"], 3.0)

    def test_export_prometheus(self):
        out = self.rev.export_prometheus()
        self.assertIn("revolver_requests_total", out)
        self.assertIn("revolver_requests_saved_total", out)
        self.assertIn("revolver_engine_cost_usd_total", out)
        self.assertIn("revolver_engine_prompt_tokens_total", out)
        self.assertIn("revolver_chamber_health{", out)
        self.assertIn('revolver_chamber_cost_usd_total{', out)


class TestPhase5SchemaAsync(unittest.IsolatedAsyncioTestCase):
    class Reply(BaseModel):
        answer: str

    async def asyncSetUp(self):
        self.rev = AsyncRevolver()
        self.rev.add_chamber("C1", "https://example.com/c1", "k1", "m1")
        self.rev.add_chamber("C2", "https://example.com/c2", "k2", "m2")

    async def asyncTearDown(self):
        await self.rev.aclose()

    async def test_async_schema_repair(self):
        async def fake_post(url, headers=None, json=None, timeout=None):
            msgs = json.get("messages", [])
            joined = " ".join(m.get("content", "") for m in msgs)
            if "Invalid draft to repair:" in joined:
                return _mock_response(200, content='{"answer": "ok"}', usage={"prompt_tokens": 1, "completion_tokens": 1})
            return _mock_response(200, content="{bad")

        mock_client = MagicMock()
        mock_client.is_closed = False
        mock_client.post = fake_post

        async def fake_aclose():
            mock_client.is_closed = True

        mock_client.aclose = fake_aclose
        self.rev._client = mock_client
        self.rev._client_loop = None

        result = await self.rev._aapply_schema_validation(self.rev._chambers, "{bad", self.Reply)
        self.assertEqual(result, '{"answer": "ok"}')


if __name__ == "__main__":
    unittest.main(verbosity=2)
