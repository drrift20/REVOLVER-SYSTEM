# 🔫 Revolver System

> **Self-Healing, Staggered-Hedged, Judge Revolver Engine for Large Language Models.**

`revolver_system` is a high-availability, fault-tolerant Python AI orchestration library. It brings **adaptive staggered hedged racing**, **dynamic health scoring**, **circuit-breaker auto-cooldowns**, and **automated draft repair (The Judge)** to any Python application, bot, or web service.

---

## 🚀 Key Features

- **Zero Hardcoded Keys:** Bring your own API keys. Supports any OpenAI-compatible provider (Groq, Gemini, OpenRouter, OpenAI, Mistral, Cerebras, vLLM, Ollama, etc.).
- **Dynamic Chamber Generation:** Register providers with your preferred models, or auto-discover available models via the provider's `/models` endpoint.
- **Staggered Hedged Racing (Phase 1):** Fires the healthiest chamber first and waits an adaptive delay ($\Delta t$, computed from that chamber's EWMA latency) before launching a backup racer. Eliminates P99 tail-latency spikes while saving redundant API quota when the primary responds fast. Fast failures trigger the backup *immediately*. Also supports classic dual-fire via `race_mode="concurrent"`.
- **Adaptive Health & Auto-Cooldowns:** Each chamber maintains a live health score (0–100) and EWMA latency. Fast responses gain health boosts; HTTP 429, 5xx, timeouts, or deprecated model IDs (400/404) trigger penalties and cooldowns.
- **The Judge Revolver:** Automatically detects broken drafts (markdown tables, prompt leakage, cutoff text) and prompts a healthy chamber to repair the draft before showing it to users.
- **Default English Judge:** Clean, concise, general-purpose repair logic that users can customize with their own instructions.
- **Connection Pooling:** Built on `requests.Session` with `HTTPAdapter` connection pooling for blazing-fast socket reuse.

---

## 📦 Installation & Setup

Place `revolver_system` in your project or install it in your Python path:

```python
from revolver import Revolver

# 1. Initialize Revolver (defaults to adaptive staggered hedging)
rev = Revolver(
    system_prompt="You are a helpful and intelligent AI assistant.",
    timeout=12,
    race_mode="hedged",       # or "concurrent" for immediate dual-trigger
)

# 2. Add your providers & API keys (as many as you want!)
rev.add_provider("groq", api_key="YOUR_GROQ_API_KEY")
rev.add_provider("gemini", api_key="YOUR_GEMINI_API_KEY")

# Or add specific models:
rev.add_provider(
    provider="openrouter",
    api_key="YOUR_OPENROUTER_KEY",
    models=["meta-llama/llama-3.3-70b-instruct:free", "google/gemma-2-9b-it:free"],
)

# Or add custom self-hosted OpenAI-compatible endpoints:
rev.add_chamber(
    name="Local vLLM",
    url="http://localhost:8000/v1/chat/completions",
    api_key="token",
    model="meta-llama/Llama-3-8B-Instruct",
)

# 3. Ask anything!
response = rev.ask("Explain the theory of relativity simply.")
print(response)

# Per-call overrides (force concurrent race or custom hedge delay):
urgent = rev.ask("Urgent query!", race_mode="concurrent")
custom_wait = rev.ask("Heavy reasoning query", hedge_delay=1.5)
```

---

## 🩺 Live Chamber Telemetry & Quota Metrics

Monitor both your chamber fleet and Phase 1 quota savings in real time:

```python
for chamber in rev.get_status():
    print(
        f"• {chamber['name']}: {chamber['health']}% HP | "
        f"Avg Latency: {chamber['avg_latency_ms']}ms | Status: {chamber['status']}"
    )

metrics = rev.get_metrics()
print(
    f"Quota saved: {metrics['calls_saved']} backup calls "
    f"({metrics['hedge_efficiency_pct']}% efficiency)"
)
```

---

## ⚖️ Customizing The Judge

You can pass a custom repair instruction to Revolver:

```python
custom_judge = (
    "A draft broke rule: {reason}. "
    "Rewrite it in a friendly, witty pirate tone. "
    "Output ONLY the corrected text."
)

rev = Revolver(judge_instruction=custom_judge)
```

---

## 🧪 Running Tests & Demo

```bash
python3 demo.py
python3 -m unittest test_revolver.py -v
```

---

## 📄 License
MIT License. Created by Jakwan.
