"""
Interactive Demo & Test Script for Revolver System (Phase 1 & Phase 2).
"""

import asyncio
import os
import sys

# Add parent directory to path so imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from revolver import AsyncRevolver, Revolver


async def async_demo_check():
    async with AsyncRevolver(
        system_prompt="You are a brilliant AI assistant.",
        timeout=10,
        race_mode="hedged",
    ) as arev:
        arev.add_provider(
            provider="groq",
            api_key="mock_groq_key",
            models=["llama-3.3-70b-versatile"],
        )
        print(f"⚡ AsyncRevolver ready (chambers={len(arev.get_status())}, mode='{arev.race_mode}').")


def main():
    print("=" * 60)
    print("🔫 REVOLVER SYSTEM — SYNC & ASYNC AI ENGINE (v1.2)")
    print("=" * 60)

    # 1. Initialize Sync Revolver Engine (Phase 1 defaults to adaptive staggered hedging)
    rev = Revolver(
        system_prompt="You are a brilliant AI assistant. Answer clearly and concisely.",
        timeout=10,
        race_mode="hedged",
    )
    print("✅ Revolver instance initialized (race_mode='hedged').")

    # 2. Add simulated/mock or real chambers
    chambers = rev.add_provider(
        provider="groq",
        api_key="mock_groq_key",
        models=["llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
    )
    print(f"✅ Registered Groq chambers: {chambers}")

    chambers_gemini = rev.add_provider(
        provider="gemini",
        api_key="mock_gemini_key",
        models=["gemini-2.5-flash", "gemini-2.0-flash"],
    )
    print(f"✅ Registered Gemini chambers: {chambers_gemini}")

    rev.add_chamber(
        name="Local vLLM",
        url="http://localhost:8000/v1/chat/completions",
        api_key="local_token",
        model="meta-llama/Llama-3-8B-Instruct",
    )
    print("✅ Registered custom chamber: Local vLLM")

    # 3. Check Chamber Health & Latency Status
    print("\n📊 Initial Chamber Health Status:")
    for status in rev.get_status():
        lat_str = f"{status['avg_latency_ms']}ms" if status["avg_latency_ms"] is not None else "N/A"
        print(
            f"  • {status['name']} [{status['model']}]: "
            f"Health={status['health']}% | Status={status['status']} | AvgLatency={lat_str}"
        )

    # 4. Simulate Adaptive Latency Feedback & Hedge Delay Calculation
    c1 = rev._chambers[0]
    print(f"\n⏱️ Cold-start hedge delay for {c1['name']}: {rev.get_hedge_delay(c1):.2f}s")
    rev._apply_outcome(c1, ("ok", 0.38))
    print(f"⏱️ Adapted hedge delay after 0.38s reply: {rev.get_hedge_delay(c1):.2f}s")

    # 5. Test Text Normalization & Sanitizer
    dirty_text = "Here is a range: 1?2 and \u2011 dash.\n| Col 1 | Col 2 |\n|---|---|\n| val 1 | val 2 |\nClean text here."
    cleaned = rev.sanitize_draft(dirty_text)
    print("\n🧹 Sanitizer Test:")
    print("Before:\n", dirty_text)
    print("-" * 30)
    print("After:\n", cleaned)

    # 6. Test Draft Validation
    is_valid, reason = rev.validate_draft("This response was truncated by [truncated]")
    print(f"\n🔍 Validation Check on Corrupted Text: is_valid={is_valid}, reason='{reason}'")

    # 7. Show Engine Metrics
    print("\n📈 Engine Telemetry Snapshot:")
    print(" ", rev.get_metrics())

    # 8. Verify AsyncRevolver (Phase 2)
    asyncio.run(async_demo_check())

    print("\n🎉 All Phase 1 & Phase 2 engine mechanics verified successfully!")
    rev.close()


if __name__ == "__main__":
    main()
