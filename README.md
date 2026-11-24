# 🔫 Revolver System

> **Self-Healing, Multi-Trigger, Judge Revolver Engine for Large Language Models.**

`revolver_system` is a high-availability, fault-tolerant Python AI orchestration library. It brings the power of **concurrent model racing**, **adaptive health scoring**, **circuit-breaker auto-cooldowns**, and **automated draft repair (The Judge)** to any Python application, bot, or web service.

---

## 🚀 Key Features

- **Zero Hardcoded Keys:** Bring your own API keys. Supports any OpenAI-compatible provider (Groq, Gemini, OpenRouter, OpenAI, Mistral, Cerebras, vLLM, Ollama, etc.).
- **Dynamic Chamber Generation:** Register providers with your preferred models, or auto-discover available models via the provider's `/models` endpoint.
- **Dual-Trigger Concurrent Racing:** Races top-2 healthiest chambers simultaneously on a thread pool. The first valid non-empty answer wins; slower competitors are instantly cancelled.
- **Adaptive Health & Auto-Cooldowns:** Each chamber maintains a live health score (0–100). Fast responses gain health boosts; HTTP 429, 5xx, timeouts, or deprecated model IDs (400/404) trigger penalties and cooldowns.
- **The Judge Revolver:** Automatically detects broken drafts (markdown tables, prompt leakage, cutoff text) and prompts a healthy chamber to repair the draft before showing it to users.
- **Default English Judge:** Clean, concise, general-purpose repair logic that users can customize with their own instructions.
- **Connection Pooling:** Built on `requests.Session` with `HTTPAdapter` connection pooling for blazing-fast socket reuse.

---

## 📦 Installation & Setup

Place `revolver_system` in your project or install it in your Python path:

```python
from revolver import Revolver

# 1. Initialize Revolver
rev = Revolver(
    system_prompt="You are a helpful and intelligent AI assistant.",
    timeout=12,
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
```

---

## 🩺 Live Chamber Telemetry

Monitor your chamber fleet in real time:

```python
for chamber in rev.get_status():
    print(f"• {chamber['name']}: {chamber['health']}% HP | Status: {chamber['status']}")
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

## 📄 License
MIT License. Created by Jakwan.
