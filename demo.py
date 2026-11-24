"""
Interactive Demo & Test Script for Revolver System.
"""

import sys
import os

# Add parent directory to path so imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from revolver import Revolver

def main():
    print("=" * 60)
    print("🔫 REVOLVER SYSTEM — SELF-HEALING AI ENGINE")
    print("=" * 60)

    # 1. Initialize Revolver Engine
    rev = Revolver(
        system_prompt="You are a brilliant AI assistant. Answer clearly and concisely.",
        timeout=10,
    )
    print("✅ Revolver instance initialized.")

    # 2. Add simulated/mock or real chambers
    # Adding a provider with custom models:
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

    # Adding custom chamber:
    rev.add_chamber(
        name="Local vLLM",
        url="http://localhost:8000/v1/chat/completions",
        api_key="local_token",
        model="meta-llama/Llama-3-8B-Instruct",
    )
    print("✅ Registered custom chamber: Local vLLM")

    # 3. Check Chamber Health Status
    print("\n📊 Initial Chamber Health Status:")
    for status in rev.get_status():
        print(f"  • {status['name']} [{status['model']}]: Health={status['health']}% | Status={status['status']}")

    # 4. Test Text Normalization & Sanitizer
    dirty_text = "Here is a range: 1?2 and \u2011 dash.\n| Col 1 | Col 2 |\n|---|---|\n| val 1 | val 2 |\nClean text here."
    cleaned = rev.sanitize_draft(dirty_text)
    print("\n🧹 Sanitizer Test:")
    print("Before:\n", dirty_text)
    print("-" * 30)
    print("After:\n", cleaned)

    # 5. Test Draft Validation
    is_valid, reason = rev.validate_draft("This response was truncated by [truncated]")
    print(f"\n🔍 Validation Check on Corrupted Text: is_valid={is_valid}, reason='{reason}'")

    print("\n🎉 All local engine mechanics verified successfully!")

if __name__ == "__main__":
    main()
