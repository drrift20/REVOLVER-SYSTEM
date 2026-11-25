# 🔫 Revolver System

> **Self-Healing, Staggered-Hedged, Sync & Async Judge Revolver Engine for Large Language Models.**

`revolver_system` is a high-availability, fault-tolerant Python AI orchestration library. It brings **adaptive staggered hedged racing**, **native async high-concurrency execution**, **dynamic health scoring**, **circuit-breaker auto-cooldowns**, and **automated draft repair (The Judge)** to any Python application, bot, or web service.

---

## 🚀 Key Features

- **Zero Hardcoded Keys:** Bring your own API keys. Supports any OpenAI-compatible provider (Groq, Gemini, OpenRouter, OpenAI, Mistral, Cerebras, vLLM, Ollama, etc.).
- **Dynamic Chamber Generation:** Register providers with your preferred models, or auto-discover available models via the provider's `/models` endpoint (`discover_models=True` / `await aadd_provider(..., discover_models=True)`).
- **Staggered Hedged Racing (Phase 1):** Fires the healthiest chamber first and waits an adaptive delay ($\Delta t$, computed from that chamber's EWMA latency) before launching a backup racer. Eliminates P99 tail-latency spikes while saving redundant API quota when the primary responds fast. Fast failures trigger the backup *immediately*. Also supports classic dual-fire via `race_mode="concurrent"`.
- **Native Async Engine (Phase 2):** `AsyncRevolver` powered by `asyncio` and pooled `httpx.AsyncClient` (`max_keepalive_connections=20`, `max_connections=100`). Handles massive concurrent workloads on a single event loop and immediately cancels losing HTTP coroutines mid-flight.
- **Adaptive Health & Auto-Cooldowns:** Each chamber maintains a live health score (0–100) and EWMA latency. Fast responses gain health boosts; HTTP 429, 5xx, timeouts, or deprecated model IDs (400/404) trigger penalties and cooldowns.
- **The Judge Revolver:** Automatically detects broken drafts (markdown tables, prompt leakage, cutoff text) and prompts a healthy chamber to repair the draft before showing it to users.
- **Default English Judge:** Clean, concise, general-purpose repair logic that users can customize with their own instructions.
- **Connection Pooling:** Sync `Revolver` uses `requests.Session` + `HTTPAdapter`; `AsyncRevolver` uses `httpx.AsyncClient` with configurable `httpx.Limits`.

---

## 📦 Installation & Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### Synchronous Usage (`Revolver`)

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

### Asynchronous High-Concurrency Usage (`AsyncRevolver`)

```python
import asyncio
from revolver import AsyncRevolver

async def main():
    async with AsyncRevolver(
        system_prompt="You are a helpful and intelligent AI assistant.",
        timeout=12,
        race_mode="hedged",
    ) as rev:
        rev.add_provider("groq", api_key="YOUR_GROQ_API_KEY")
        rev.add_provider("gemini", api_key="YOUR_GEMINI_API_KEY")

        # Optional async model discovery from provider API:
        # await rev.aadd_provider("openrouter", api_key="...", discover_models=True)

        replies = await asyncio.gather(
            rev.ask("What is quantum entanglement?"),
            rev.ask("Summarize how TCP works in two sentences."),
        )
        for reply in replies:
            print(reply)

asyncio.run(main())
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

You can pass a custom repair instruction to Revolver or AsyncRevolver:

```python
custom_judge = (
    "A draft broke rule: {reason}. "
    "Rewrite it in a friendly, witty pirate tone. "
    "Output ONLY the corrected text."
)

rev = Revolver(judge_instruction=custom_judge)
```

---

## 💰 On-Device Budget Guard (Paid API Spending Controls)

Revolver includes a **local, on-device budget guard** that prevents surprise API bills. Configuration lives in a JSON file on the user's device — it never phones home, works offline, and supports both paid and free API users.

### Setup

```python
from revolver import Revolver

rev = Revolver()

# Add providers & chambers
rev.add_provider("groq", api_key="YOUR_GROQ_KEY")
rev.add_chamber(
    name="Local vLLM",
    url="http://localhost:8000/v1/chat/completions",
    api_key="token",
    model="meta-llama/Llama-3-8B-Instruct",
)

# Set a monthly spending cap
rev.set_budget(limit_usd=10.0, cycle_type="monthly", warning_threshold_pct=80.0)

# Tag chambers: paid (with per-1k-token rates) vs free
rev.mark_chamber_paid("Groq llama-3.3-70b-versatile", rate_prompt=0.59, rate_completion=0.79)
rev.mark_chamber_free("Local vLLM")

# Check status anytime
print(rev.get_budget_status())
# {'cycle_type': 'monthly', 'limit_usd': 10.0, 'spent_usd': 0.0,
#  'remaining_usd': 10.0, 'warning_threshold_pct': 80.0, ...}
```

### Behavior

| Event | What happens |
| :--- | :--- |
| **Spending approaches limit** | Warning threshold (default 80%) is tracked; you can surface it in your UI |
| **Limit exhausted** | Paid chambers are skipped automatically; requests fall back to free-only chambers |
| **No free chambers available** | Returns a busy message indicating budget exhaustion |
| **Ignore once** | Bypass the cap for the next cycle hour |
| **Ignore forever** | Permanently disable budget enforcement |
| **Raise budget** | Increase the limit and continue |
| **New cycle starts** | Spend resets to 0 automatically (monthly/yearly/custom) |

### Module-level shortcuts

```python
from revolver import set_budget, mark_chamber_paid, mark_chamber_free, get_budget_status

set_budget(20.0, cycle_type="monthly")
mark_chamber_paid("Groq llama-3.3-70b-versatile", 0.59, 0.79)
mark_chamber_free("Local vLLM")
print(get_budget_status())
```

---

## 🧪 Running Tests & Demo

```bash
python3 demo.py
python3 -m unittest test_revolver.py -v
```

---

## 📄 License
MIT License — see [LICENSE](LICENSE). Created by [drrift20](https://github.com/drrift20).
