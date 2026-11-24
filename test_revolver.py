"""
Unit tests for Revolver System (Phase 1: Staggered Hedged Racing + Core Engine).
"""

import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from revolver import (
    DEFAULT_HEDGE_DELAY,
    MAX_HEDGE_DELAY,
    MIN_HEDGE_DELAY,
    Revolver,
)


def _mock_response(status_code=200, content="ok", usage=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    payload = {}
    if status_code == 200:
        payload = {"choices": [{"message": {"content": content}}]}
        if usage is not None:
            payload["usage"] = usage
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

        # Wait briefly so slow loser's done_callback completes and verify single-apply
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

            # Cold start
            self.assertEqual(adaptive_rev.get_hedge_delay(c1), DEFAULT_HEDGE_DELAY)

            # Fast 0.2s completion -> 0.2 * 1.15 = 0.23s -> clamped to MIN_HEDGE_DELAY (0.4s)
            adaptive_rev._apply_outcome(c1, ("ok", 0.2))
            self.assertEqual(adaptive_rev.get_hedge_delay(c1), MIN_HEDGE_DELAY)

            # Slow completions pull EWMA up toward ~1.5s
            for _ in range(10):
                adaptive_rev._apply_outcome(c1, ("ok", 1.5))
            delay = adaptive_rev.get_hedge_delay(c1)
            self.assertGreater(delay, 1.0)
            self.assertLessEqual(delay, MAX_HEDGE_DELAY)

            # Huge spike clamped to MAX_HEDGE_DELAY
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
                # Return corrupted output both for primary call and any judge call hitting C1
                return _mock_response(200, content="[truncated]")
            # If judge call hits C2, fail it so Tier 2 runs C2 cleanly for original prompt
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
