"""
revolver_cli.py — Cross-Platform CLI & Agent Wrapper for Revolver System.

Provides two modes on top of revolver.py (core logic untouched):

  1. Brain / Library Mode
     revolver status        — inspect chamber health, latency, cost
     revolver budget        — view / configure on-device budget guard
     revolver test          — run built-in self-test suite

  2. CLI Agent Mode
     revolver agent         — interactive coding agent (Revolver as brain)
     revolver build         — scaffold new projects (bot, script, web, cli)

Works on Linux and Windows. Uses pathlib.Path everywhere, loads API keys
from environment variables or ~/.revolver/config.json, and never hardcodes keys.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import Revolver core (unchanged).
try:
    from revolver import (
        AsyncRevolver,
        BudgetExceeded,
        BudgetGuard,
        KNOWN_PROVIDERS,
        Revolver,
        TIER_HEAVY,
        TIER_SPRINT,
    )
except ImportError as exc:  # pragma: no cover - helpful CLI error
    print(f"❌ Cannot import revolver module: {exc}")
    print("   Make sure revolver.py is on your PYTHONPATH.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Cross-platform paths & constants
# ---------------------------------------------------------------------------

IS_WINDOWS = os.name == "nt"

CONFIG_DIR = Path.home() / ".revolver"
CONFIG_FILE = CONFIG_DIR / "config.json"
BUDGET_FILE = CONFIG_DIR / "budget.json"

# Provider -> environment variable name for BYOK.
PROVIDER_ENV_KEYS: Dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}

# Reverse map: env var -> provider.
ENV_PROVIDER_MAP = {v: k for k, v in PROVIDER_ENV_KEYS.items()}

# Supported project scaffolds.
PROJECT_TYPES = {
    "python-bot": "A simple Python AI bot using Revolver.",
    "python-script": "A standalone Python script using Revolver.",
    "web-app": "A FastAPI web app powered by Revolver.",
    "cli-app": "A CLI application powered by Revolver.",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise, and intelligent AI assistant. "
    "Answer clearly and accurately."
)


# ---------------------------------------------------------------------------
# Config & BYOK handling (Linux + Windows via pathlib.Path.home())
# ---------------------------------------------------------------------------

def ensure_config_dir() -> None:
    """Create ~/.revolver directory if missing (works on Linux & Windows)."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Load ~/.revolver/config.json. Returns empty dict if missing/invalid."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(config: Dict[str, Any]) -> None:
    """Persist config to ~/.revolver/config.json."""
    ensure_config_dir()
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_api_key(provider: str) -> Optional[str]:
    """
    Resolve an API key for a provider.

    Priority: environment variable > ~/.revolver/config.json.
    Never hardcodes keys.
    """
    env_var = PROVIDER_ENV_KEYS.get(provider.lower())
    if env_var:
        key = os.environ.get(env_var)
        if key:
            return key
    config = load_config()
    keys = config.get("api_keys", {})
    return keys.get(provider.lower())


def list_configured_providers() -> List[Tuple[str, Optional[str]]]:
    """Return [(provider, api_key_or_None)] for all known providers."""
    results = []
    for provider in PROVIDER_ENV_KEYS:
        results.append((provider, get_api_key(provider)))
    return results


# ---------------------------------------------------------------------------
# Revolver engine factory
# ---------------------------------------------------------------------------

def build_revolver(
    providers: Optional[List[str]] = None,
    budget_limit: Optional[float] = None,
    budget_cycle: str = "monthly",
    race_mode: str = "hedged",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Revolver:
    """
    Build a synchronous Revolver engine configured from BYOK sources.

    Args:
        providers: Provider names to register (None = all with keys available).
        budget_limit: Optional monthly/yearly USD cap (None = no budget).
        budget_cycle: Budget cycle type ("monthly", "yearly", "custom").
        race_mode: "hedged" or "concurrent".
        system_prompt: System prompt for the engine.
    """
    rev = Revolver(system_prompt=system_prompt, race_mode=race_mode)

    # Budget guard (on-device, local JSON only).
    if budget_limit is not None:
        rev.set_budget(
            limit_usd=budget_limit,
            cycle_type=budget_cycle,
            config_path=str(BUDGET_FILE),
        )

    # Register providers with available API keys.
    target_providers = providers or list(PROVIDER_ENV_KEYS.keys())
    for provider in target_providers:
        api_key = get_api_key(provider)
        if not api_key:
            continue
        try:
            rev.add_provider(provider=provider, api_key=api_key)
        except Exception as exc:  # pragma: no cover - provider registration
            print(f"⚠️  Could not register provider '{provider}': {exc}")

    if not rev._chambers:
        print("⚠️  No API keys found. Set env vars (e.g. GROQ_API_KEY) or "
              "~/.revolver/config.json, or add chambers manually.")

    return rev


# ---------------------------------------------------------------------------
# Helpers: user confirmation (destructive edits / commands / git push)
# ---------------------------------------------------------------------------

def confirm_action(prompt: str, default: bool = False) -> bool:
    """
    Ask the user for confirmation. Returns True if confirmed.

    Used before destructive file edits, shell commands, and git push.
    """
    suffix = " [y/N] " if not default else " [Y/n] "
    try:
        answer = input(prompt + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n❌ Cancelled.")
        return False
    if not answer:
        return default
    return answer in ("y", "yes")


def print_panel(title: str, body: str) -> None:
    """Print a simple bordered panel (cross-platform, no external deps)."""
    lines = body.splitlines()
    width = max(len(title), max((len(l) for l in lines), default=0)) + 4
    print("+" + "-" * width + "+")
    print(f"| {title:<{width - 2}} |")
    print("+" + "-" * width + "+")
    for line in lines:
        print(f"| {line:<{width - 2}} |")
    print("+" + "-" * width + "+")


# ---------------------------------------------------------------------------
# Brain / Library Mode commands
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    """revolver status — show chamber health, latency, cost, budget."""
    rev = build_revolver(budget_limit=getattr(args, "budget_limit", None))
    chambers = rev.get_status()

    if not chambers:
        print("🔫 Revolver System — no chambers registered.")
        return 0

    print("\n🔫 Revolver System — Chamber Status\n")
    for ch in chambers:
        lat = f"{ch['avg_latency_ms']}ms" if ch["avg_latency_ms"] is not None else "N/A"
        last = f"{ch['last_latency_ms']}ms" if ch["last_latency_ms"] is not None else "N/A"
        print(f"  • {ch['name']} [{ch['model']}]")
        print(f"      Health : {ch['health']}%  ({ch['status']})")
        print(f"      AvgLat : {lat}  |  LastLat : {last}")
        print(f"      Succ/Fail: {ch['successes']}/{ch['failures']}")
        print(f"      Tokens : {ch['prompt_tokens']} prompt / {ch['completion_tokens']} completion")
        print(f"      Cost   : ${ch['cost_usd']:.6f}")
        if ch["cooldown_remaining"] > 0:
            print(f"      Cooldown: {ch['cooldown_remaining']}s remaining")
        print()

    # Engine metrics.
    metrics = rev.get_metrics()
    print("📊 Engine Metrics:")
    print(f"      Total requests : {metrics['total_requests']}")
    print(f"      Hedged saved   : {metrics['calls_saved']} calls")
    print(f"      Hedge efficiency: {metrics['hedge_efficiency_pct']}%")
    print(f"      Total cost     : ${metrics['total_cost_usd']:.6f}")
    print(f"      Race mode      : {metrics['race_mode']}")

    # Budget status.
    budget = rev.get_budget_status()
    if budget.get("budget_configured"):
        print("\n💰 Budget Guard:")
        print(f"      Cycle   : {budget['cycle_type']}")
        print(f"      Limit   : ${budget['limit_usd']}")
        print(f"      Spent   : ${budget['spent_usd']}")
        print(f"      Remaining: ${budget['remaining_usd']}")
        print(f"      Warning : {budget['warning_threshold_pct']}%")

    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    """revolver budget — view or configure the on-device budget guard."""
    rev = build_revolver()

    if args.set_limit is not None:
        rev.set_budget(
            limit_usd=args.set_limit,
            cycle_type=args.cycle,
            config_path=str(BUDGET_FILE),
        )
        print(f"✅ Budget set: ${args.set_limit} / {args.cycle} (stored in {BUDGET_FILE})")

    if args.mark_paid:
        # mark-chamber <name> <prompt_rate> <completion_rate>
        parts = args.mark_paid
        if len(parts) != 3:
            print("❌ --mark-paid requires: <chamber_name> <prompt_rate_usd_per_1k> <completion_rate_usd_per_1k>")
            return 1
        name, rate_p, rate_c = parts
        try:
            rev.mark_chamber_paid(name, float(rate_p), float(rate_c))
            print(f"✅ Marked '{name}' as paid (${rate_p}/1k prompt, ${rate_c}/1k completion)")
        except ValueError:
            print("❌ Invalid rate values. Must be numbers.")
            return 1

    if args.mark_free:
        rev.mark_chamber_free(args.mark_free)
        print(f"✅ Marked '{args.mark_free}' as free")

    budget = rev.get_budget_status()
    print("\n💰 Current Budget Status:")
    if not budget.get("budget_configured"):
        print("      No budget configured. Use --set-limit to set one.")
    else:
        print(f"      Cycle   : {budget['cycle_type']}")
        print(f"      Limit   : ${budget['limit_usd']}")
        print(f"      Spent   : ${budget['spent_usd']}")
        print(f"      Remaining: ${budget['remaining_usd']}")
        print(f"      Warning : {budget['warning_threshold_pct']}%")
        print(f"      Ignore forever: {budget.get('ignore_forever', False)}")

    print("\n🔧 Chambers:")
    for name, info in budget.get("chambers", {}).items():
        tag = "PAID" if info.get("paid") else "FREE"
        print(f"      {name}: {tag}")

    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """revolver test — run the built-in unit test suite."""
    print("🧪 Running Revolver System test suite...\n")
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(Path(__file__).resolve().parent),
        pattern="test_revolver.py",
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


# ---------------------------------------------------------------------------
# Agent core: file system & shell operations (safe, confirmed)
# ---------------------------------------------------------------------------

class AgentWorkspace:
    """Sandboxed workspace for the agent to inspect/create/edit files."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str) -> Path:
        """Resolve a path relative to workspace root, preventing escape."""
        p = (self.root / path).resolve()
        # Guard: do not allow escaping the workspace root.
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"Path escapes workspace: {path}")
        return p

    def list_files(self, path: str = ".") -> List[Dict[str, Any]]:
        """List files/dirs under a workspace-relative path."""
        target = self.resolve(path)
        if not target.exists():
            return []
        entries = []
        for item in sorted(target.iterdir()):
            entries.append({
                "name": item.name,
                "path": str(item.relative_to(self.root)),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else 0,
            })
        return entries

    def read_file(self, path: str) -> str:
        """Read a file from the workspace."""
        target = self.resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str, confirm: bool = True) -> bool:
        """Write a file, with confirmation if it already exists (destructive)."""
        target = self.resolve(path)
        if target.exists() and confirm:
            if not confirm_action(f"Overwrite existing file: {target}"):
                print(f"⏭️  Skipped: {target}")
                return False
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(f"✅ Wrote: {target}")
        return True

    def edit_file(self, path: str, old_string: str, new_string: str, confirm: bool = True) -> bool:
        """Edit a file by replacing old_string with new_string (with confirmation)."""
        target = self.resolve(path)
        if not target.is_file():
            print(f"❌ File not found: {target}")
            return False
        content = target.read_text(encoding="utf-8")
        if old_string not in content:
            print(f"❌ Search string not found in {target}")
            return False
        if confirm:
            preview = new_string[:200] + ("..." if len(new_string) > 200 else "")
            if not confirm_action(f"Edit {target}?\n  New content preview: {preview}"):
                print(f"⏭️  Skipped edit: {target}")
                return False
        new_content = content.replace(old_string, new_string, 1)
        target.write_text(new_content, encoding="utf-8")
        print(f"✅ Edited: {target}")
        return True

    def delete_file(self, path: str, confirm: bool = True) -> bool:
        """Delete a file (with confirmation — destructive)."""
        target = self.resolve(path)
        if not target.exists():
            print(f"❌ Not found: {target}")
            return False
        if confirm:
            if not confirm_action(f"DELETE {target}? This cannot be undone."):
                print(f"⏭️  Skipped deletion: {target}")
                return False
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        print(f"🗑️  Deleted: {target}")
        return True

    def run_command(self, command: str, confirm: bool = True) -> Tuple[int, str, str]:
        """
        Run a shell command cross-platform (Linux bash/sh, Windows cmd/powershell).

        Requires explicit confirmation before execution. git push always requires
        an extra explicit confirmation.
        """
        is_git_push = "git push" in command
        if confirm or is_git_push:
            extra = " (git push)" if is_git_push else ""
            if not confirm_action(f"Run command{extra}: {command}"):
                print(f"⏭️  Skipped command: {command}")
                return -1, "", "Cancelled by user"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as exc:  # pragma: no cover - subprocess edge cases
            return -1, "", str(exc)


# ---------------------------------------------------------------------------
# Project scaffolding
# ---------------------------------------------------------------------------

PROJECT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "python-bot": {
        "main.py": '''"""
{project_name} — AI bot powered by Revolver System.
"""
import os
import sys
from pathlib import Path

# Add project root to path so we can import revolver.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from revolver import Revolver


def build_bot() -> Revolver:
    rev = Revolver(
        system_prompt="You are a helpful AI assistant.",
        race_mode="hedged",
    )
    # Providers are auto-registered from env vars or ~/.revolver/config.json.
    for provider in ("groq", "gemini", "openrouter", "openai"):
        rev.add_provider(provider=provider, api_key=os.environ.get(f"{provider.upper()}_API_KEY", ""))
    return rev


def main():
    rev = build_bot()
    print("🤖 Bot ready. Type your message (Ctrl+C to exit).")
    while True:
        try:
            user = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break
        if not user.strip():
            continue
        try:
            reply = rev.ask(user)
            print(f"🤖 {reply}")
        except Exception as exc:
            print(f"❌ Error: {exc}")


if __name__ == "__main__":
    main()
''',
        "requirements.txt": "revolver-system\nrequests>=2.28.0\nhttpx>=0.25.0\n",
        "README.md": "# {project_name}\n\nAI bot powered by Revolver System.\n",
    },
    "python-script": {
        "script.py": '''"""
{project_name} — Standalone script powered by Revolver System.
"""
from revolver import Revolver


def main():
    rev = Revolver(system_prompt="You are a helpful assistant.")
    rev.add_provider("groq", api_key="YOUR_GROQ_API_KEY")
    reply = rev.ask("Hello from {project_name}!")
    print(reply)


if __name__ == "__main__":
    main()
''',
        "requirements.txt": "revolver-system\nrequests>=2.28.0\nhttpx>=0.25.0\n",
        "README.md": "# {project_name}\n\nStandalone script using Revolver System.\n",
    },
    "web-app": {
        "main.py": '''"""
{project_name} — FastAPI web app powered by Revolver System.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from revolver import Revolver

app = FastAPI(title="{project_name}")
rev = Revolver(system_prompt="You are a helpful AI assistant.")
rev.add_provider("groq", api_key="YOUR_GROQ_API_KEY")


class AskRequest(BaseModel):
    message: str


@app.post("/ask")
def ask(req: AskRequest):
    reply = rev.ask(req.message)
    return {"reply": reply}


@app.get("/status")
def status():
    return {"chambers": rev.get_status()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
''',
        "requirements.txt": "revolver-system\nfastapi\nuvicorn\npydantic\nrequests>=2.28.0\nhttpx>=0.25.0\n",
        "README.md": "# {project_name}\n\nFastAPI web app powered by Revolver System.\n",
    },
    "cli-app": {
        "main.py": '''"""
{project_name} — CLI app powered by Revolver System.
"""
import argparse
from revolver import Revolver


def main():
    parser = argparse.ArgumentParser(description="{project_name}")
    parser.add_argument("message", help="Message to send to Revolver")
    parser.add_argument("--provider", default="groq", help="Provider to use")
    args = parser.parse_args()

    rev = Revolver()
    rev.add_provider(args.provider, api_key="YOUR_{provider.upper()}_API_KEY")
    reply = rev.ask(args.message)
    print(reply)


if __name__ == "__main__":
    main()
''',
        "requirements.txt": "revolver-system\nrequests>=2.28.0\nhttpx>=0.25.0\n",
        "README.md": "# {project_name}\n\nCLI app powered by Revolver System.\n",
    },
}


def render_template(content: str, project_name: str, provider: str = "groq") -> str:
    """Replace {project_name} and {provider} placeholders in template content."""
    return content.replace("{project_name}", project_name).replace("{provider}", provider)


def scaffold_project(
    workspace: AgentWorkspace,
    project_name: str,
    project_type: str,
    wire_revolver: bool,
) -> Path:
    """
    Create a new project directory with scaffold files.

    If wire_revolver is True, copies revolver.py into the project and
    adjusts imports to use a local copy.
    """
    project_dir = workspace.resolve(project_name)
    project_dir.mkdir(parents=True, exist_ok=True)

    templates = PROJECT_TEMPLATES[project_type]

    for filename, template_content in templates.items():
        rendered = render_template(template_content, project_name)
        (project_dir / filename).write_text(rendered, encoding="utf-8")
        print(f"✅ Created: {project_dir / filename}")

    # Copy revolver.py into the project if wiring is requested.
    revolver_source = Path(__file__).resolve().parent / "revolver.py"
    if wire_revolver and revolver_source.exists():
        dest = project_dir / "revolver.py"
        shutil.copy2(revolver_source, dest)
        print(f"✅ Copied revolver.py into project: {dest}")

        # Adjust main.py imports to use local revolver.py instead of installed package.
        main_py = project_dir / "main.py"
        if main_py.exists():
            content = main_py.read_text(encoding="utf-8")
            content = content.replace("from revolver import", "from revolver import")  # local import works since revolver.py is in same dir
            main_py.write_text(content, encoding="utf-8")
        print("✅ Revolver System wired as the project's AI brain.")
    elif wire_revolver:
        print("⚠️  revolver.py not found next to this CLI; skipping auto-wire.")

    return project_dir


# ---------------------------------------------------------------------------
# Agent mode: interactive loop
# ---------------------------------------------------------------------------

def cmd_agent(args: argparse.Namespace) -> int:
    """
    revolver agent — interactive coding agent with Revolver as its brain.

    The agent uses Revolver to generate responses and can inspect/create/edit
    files and run shell commands with explicit user confirmation.
    """
    workspace_root = Path(args.workspace).resolve() if args.workspace else Path.cwd()
    workspace = AgentWorkspace(workspace_root)

    print("\n🤖 Revolver Agent — Interactive Mode")
    print(f"   Workspace: {workspace.root}")
    print("   Commands: /read <path> | /write <path> | /edit <path> | /run <cmd> | /ls <path> | /quit\n")

    rev = build_revolver(race_mode=args.race_mode)
    if not rev._chambers:
        print("⚠️  No chambers registered. Agent will have limited capability without API keys.")

    while True:
        try:
            user_input = input("🔫 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Agent exited.")
            return 0

        if not user_input:
            continue
        if user_input in ("/quit", "/exit", "quit", "exit"):
            print("👋 Agent exited.")
            return 0

        # --- Slash commands for file/system operations ---
        if user_input.startswith("/ls "):
            path = user_input[4:].strip()
            entries = workspace.list_files(path)
            if not entries:
                print("   (empty or not found)")
            for e in entries:
                marker = "📁" if e["is_dir"] else "📄"
                print(f"   {marker} {e['path']}")
            continue

        if user_input.startswith("/read "):
            path = user_input[6:].strip()
            try:
                content = workspace.read_file(path)
                print(f"\n--- {path} ---\n{content}\n--- end ---\n")
            except Exception as exc:
                print(f"❌ {exc}")
            continue

        if user_input.startswith("/write "):
            # /write <path> then prompt for content (multiline until /end)
            path = user_input[7:].strip()
            print(f"   Enter content for {path}. Type /end on its own line to finish.")
            lines = []
            while True:
                try:
                    line = input("   ")
                except (EOFError, KeyboardInterrupt):
                    print("\n❌ Cancelled.")
                    break
                if line.strip() == "/end":
                    break
                lines.append(line)
            content = "\n".join(lines)
            workspace.write_file(path, content)
            continue

        if user_input.startswith("/edit "):
            path = user_input[6:].strip()
            print("   Enter old string (type /end on its own line to finish):")
            old_lines = []
            while True:
                try:
                    line = input("   ")
                except (EOFError, KeyboardInterrupt):
                    print("\n❌ Cancelled.")
                    break
                if line.strip() == "/end":
                    break
                old_lines.append(line)
            old_string = "\n".join(old_lines)

            print("   Enter new string (type /end on its own line to finish):")
            new_lines = []
            while True:
                try:
                    line = input("   ")
                except (EOFError, KeyboardInterrupt):
                    print("\n❌ Cancelled.")
                    break
                if line.strip() == "/end":
                    break
                new_lines.append(line)
            new_string = "\n".join(new_lines)
            workspace.edit_file(path, old_string, new_string)
            continue

        if user_input.startswith("/run "):
            command = user_input[5:].strip()
            code, out, err = workspace.run_command(command)
            if out:
                print(out)
            if err:
                print(err, file=sys.stderr)
            print(f"   (exit code: {code})")
            continue

        if user_input.startswith("/build "):
            parts = user_input[7:].strip().split()
            if not parts:
                print("   Usage: /build <project_name> [python-bot|python-script|web-app|cli-app]")
                continue
            name = parts[0]
            ptype = parts[1] if len(parts) > 1 else "python-bot"
            if ptype not in PROJECT_TEMPLATES:
                print(f"   Unknown project type: {ptype}. Options: {', '.join(PROJECT_TEMPLATES)}")
                continue
            wire = confirm_action(f"Wire Revolver System as the project's AI brain? [y/N]", default=False)
            scaffold_project(workspace, name, ptype, wire)
            continue

        # --- Natural language: use Revolver as the brain ---
        print("   (thinking with Revolver...)")
        try:
            reply = rev.ask(user_input)
        except BudgetExceeded as exc:
            print(f"⚠️  Budget exhausted: {exc}")
            for opt in exc.options:
                print(f"      • {opt['label']} ({opt['action']})")
            continue
        except Exception as exc:
            print(f"❌ Revolver error: {exc}")
            continue

        print(f"\n🤖 {reply}\n")

        # Heuristic: if the reply mentions file paths or commands, offer to act.
        # Keep it simple — just print the reply. The user can use slash commands to act.


def cmd_build(args: argparse.Namespace) -> int:
    """
    revolver build — scaffold a new project powered by Revolver.

    Non-interactive variant of agent project creation. Useful for scripting.
    """
    name = args.name
    ptype = args.type
    if ptype not in PROJECT_TEMPLATES:
        print(f"❌ Unknown project type: {ptype}")
        print(f"   Options: {', '.join(PROJECT_TEMPLATES)}")
        return 1

    workspace = AgentWorkspace(Path.cwd())
    wire = False
    if args.wire_revolver:
        wire = True
    elif not args.yes:
        wire = confirm_action(
            f"Wire Revolver System as the project's AI brain? [y/N]",
            default=False,
        )

    project_dir = scaffold_project(workspace, name, ptype, wire)
    print(f"\n🎉 Project created at: {project_dir}")
    print("   Next steps:")
    print(f"     cd {project_dir}")
    print("     pip install -r requirements.txt")
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="revolver",
        description="Revolver System — CLI & Agent Wrapper (Linux & Windows)",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- status ---
    p_status = sub.add_parser("status", help="Show chamber health, latency, cost")
    p_status.add_argument("--budget-limit", type=float, default=None, help="Optional budget limit for this check")
    p_status.set_defaults(func=cmd_status)

    # --- budget ---
    p_budget = sub.add_parser("budget", help="View / configure on-device budget guard")
    p_budget.add_argument("--set-limit", type=float, default=None, help="Set spending limit (USD)")
    p_budget.add_argument("--cycle", choices=["monthly", "yearly", "custom"], default="monthly", help="Budget cycle")
    p_budget.add_argument("--mark-paid", nargs=3, metavar=("NAME", "PROMPT_RATE", "COMPLETION_RATE"), help="Mark a chamber as paid")
    p_budget.add_argument("--mark-free", metavar="NAME", help="Mark a chamber as free")
    p_budget.set_defaults(func=cmd_budget)

    # --- test ---
    p_test = sub.add_parser("test", help="Run the built-in test suite")
    p_test.set_defaults(func=cmd_test)

    # --- agent ---
    p_agent = sub.add_parser("agent", help="Interactive coding agent (Revolver as brain)")
    p_agent.add_argument("--workspace", default=".", help="Workspace directory (default: current dir)")
    p_agent.add_argument("--race-mode", choices=["hedged", "concurrent"], default="hedged", help="Race mode")
    p_agent.set_defaults(func=cmd_agent)

    # --- build ---
    p_build = sub.add_parser("build", help="Scaffold a new project")
    p_build.add_argument("name", help="Project name")
    p_build.add_argument("--type", choices=list(PROJECT_TYPES.keys()), default="python-bot", help="Project type")
    p_build.add_argument("--wire-revolver", action="store_true", help="Wire Revolver into the project (skip prompt)")
    p_build.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")
    p_build.set_defaults(func=cmd_build)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
