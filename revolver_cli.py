"""
revolver_cli.py — Full Autonomous SWE & Multi-Repo Agent Wrapper for Revolver System.

Provides two modes on top of revolver.py (core logic untouched):

  1. Brain / Library Mode
     revolver status        — inspect chamber health, latency, cost, budget
     revolver budget        — view / configure on-device budget guard
     revolver test          — run built-in self-test suite

  2. CLI Agent Mode
     revolver agent         — interactive coding agent (Revolver as brain)
     revolver swe           — autonomous SWE agent (plan → execute → verify loop)
     revolver build         — scaffold new projects (bot, script, web, cli)

Works on Linux and Windows. Uses pathlib.Path everywhere, loads API keys
from environment variables or ~/.revolver/config.json, and never hardcodes keys.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Import Revolver core (unchanged).
try:
    from revolver import (
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

# Optional Pydantic for structured JSON tool-calling actions.
try:
    from pydantic import BaseModel, Field, ValidationError

    HAS_PYDANTIC = True
except ImportError:  # pragma: no cover - fallback to dict-based actions
    HAS_PYDANTIC = False

    class BaseModel:  # type: ignore[no-redef]
        pass

    class Field:  # type: ignore[no-redef]
        def __init__(self, default=None, description: Optional[str] = None, **kwargs: Any):
            self.default = default
            self.description = description

    class ValidationError(Exception):  # type: ignore[no-redef]
        pass


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

# Directories/files ignored by the codebase indexer.
IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".env", "build", "dist", ".next", ".turbo"}
IGNORE_FILES = {".DS_Store", "Thumbs.db"}

# File extensions treated as source/code for indexing.
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".kt", ".rb",
    ".php", ".cs", ".cpp", ".c", ".h", ".hpp", ".swift", ".m", ".mm",
    ".vue", ".svelte", ".dart", ".lua", ".r", ".mjml",
}
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".txt", ".md", ".rst"}

# SWE agent defaults.
DEFAULT_MAX_STEPS = 15
DEFAULT_SPRINT_PROVIDERS = ("groq", "gemini", "openrouter", "openai")


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
# Revolver engine factory (budget persistence always uses BUDGET_FILE)
# ---------------------------------------------------------------------------

def build_revolver(
    providers: Optional[List[str]] = None,
    budget_limit: Optional[float] = None,
    budget_cycle: str = "monthly",
    race_mode: str = "hedged",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    budget_config_path: Optional[str] = None,
) -> Revolver:
    """
    Build a synchronous Revolver engine configured from BYOK sources.

    Args:
        providers: Provider names to register (None = all with keys available).
        budget_limit: Optional monthly/yearly USD cap (None = no budget).
        budget_cycle: Budget cycle type ("monthly", "yearly", "custom").
        race_mode: "hedged" or "concurrent".
        system_prompt: System prompt for the engine.
        budget_config_path: Path to budget JSON (defaults to ~/.revolver/budget.json).
    """
    rev = Revolver(system_prompt=system_prompt, race_mode=race_mode)

    # Budget guard (on-device, local JSON only). Always persist to BUDGET_FILE
    # so chamber tags, token rates, and cycle limits survive across commands.
    cfg_path = budget_config_path or str(BUDGET_FILE)
    if budget_limit is not None:
        rev.set_budget(
            limit_usd=budget_limit,
            cycle_type=budget_cycle,
            config_path=cfg_path,
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


def tag_chambers_by_tier(rev: Revolver, tier_map: Optional[Dict[str, str]] = None) -> None:
    """
    Tag chambers with sprint/heavy tiers for task-aware routing.

    tier_map: {chamber_name: "sprint"|"heavy"} — if omitted, chambers are
    auto-tagged based on model name heuristics.
    """
    if tier_map:
        for name, tier in tier_map.items():
            # Find chamber by name and set tier.
            for ch in rev._chambers:
                if ch["name"] == name:
                    ch["tier"] = tier
                    break
        return

    # Auto-tag: models with "flash", "lite", "mini", "8b", "20b" → sprint;
    # models with "pro", "70b", "120b", "opus", "o1" → heavy.
    sprint_markers = ("flash", "lite", "mini", "8b", "20b", "small", "fast")
    heavy_markers = ("pro", "70b", "120b", "opus", "o1", "large", "max")
    for ch in rev._chambers:
        model_lower = ch["model"].lower()
        if any(m in model_lower for m in heavy_markers):
            ch["tier"] = TIER_HEAVY
        elif any(m in model_lower for m in sprint_markers):
            ch["tier"] = TIER_SPRINT
        else:
            ch["tier"] = None  # any tier


def select_chambers_by_tier(
    rev: Revolver,
    tier: Optional[str],
    healthy: List[Dict[str, Any]],
    cooled: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Select up to 2 chambers matching the given tier, falling back to any."""
    tiered = [c for c in healthy + cooled if c.get("tier") == tier]
    if tiered:
        return tiered[:2]
    # Fall back to chambers without tier tags, then any.
    any_tier = [c for c in healthy + cooled if c.get("tier") is None]
    if any_tier:
        return any_tier[:2]
    return (healthy + cooled)[:2]


# ---------------------------------------------------------------------------
# Helpers: user confirmation (destructive edits / commands / git push)
# ---------------------------------------------------------------------------

def confirm_action(prompt: str, default: bool = False) -> bool:
    """
    Ask the user for confirmation. Returns True if confirmed.

    The prompt text should NOT include "[y/N]" — it is appended automatically.
    Used before destructive file edits, shell commands, and git push.
    """
    suffix = " [Y/n] " if default else " [y/N] "
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
    # Always use BUDGET_FILE so paid/free tags, token rates, and cycle limits
    # persist and load reliably across all commands.
    rev = build_revolver(budget_config_path=str(BUDGET_FILE))

    if args.set_limit is not None:
        rev.set_budget(
            limit_usd=args.set_limit,
            cycle_type=args.cycle,
            config_path=str(BUDGET_FILE),
        )
        print(f"✅ Budget set: ${args.set_limit} / {args.cycle} (stored in {BUDGET_FILE})")

    if args.mark_paid:
        parts = args.mark_paid
        if len(parts) != 3:
            print("❌ --mark-paid requires: <chamber_name> <prompt_rate_usd_per_1k> <completion_rate_usd_per_1k>")
            return 1
        name, rate_p, rate_c = parts
        try:
            rev.mark_chamber_paid(name, float(rate_p), float(rate_c), config_path=str(BUDGET_FILE))
            print(f"✅ Marked '{name}' as paid (${rate_p}/1k prompt, ${rate_c}/1k completion)")
        except ValueError:
            print("❌ Invalid rate values. Must be numbers.")
            return 1

    if args.mark_free:
        rev.mark_chamber_free(args.mark_free, config_path=str(BUDGET_FILE))
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
# Backup snapshot system for /undo
# ---------------------------------------------------------------------------

class SnapshotManager:
    """In-memory/temp backup snapshots for /undo support."""

    def __init__(self, max_snapshots: int = 50):
        self.max_snapshots = max_snapshots
        self.snapshots: List[Dict[str, Any]] = []
        self._temp_dir: Optional[Path] = None

    @property
    def temp_dir(self) -> Path:
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="revolver_swe_"))
        return self._temp_dir

    def snapshot(self, workspace: "MultiRepoWorkspace", repo_alias: str, rel_path: str) -> str:
        """Save a backup of a file before modification. Returns snapshot id."""
        snap_id = str(uuid.uuid4())
        try:
            repo = workspace.get_repo(repo_alias)
            content = repo.read_file(rel_path)
        except Exception:
            return snap_id  # file didn't exist; nothing to back up

        snap_dir = self.temp_dir / snap_id
        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / "meta.json").write_text(
            json.dumps({"repo": repo_alias, "path": rel_path, "content": content}, indent=2),
            encoding="utf-8",
        )
        self.snapshots.append({"id": snap_id, "repo": repo_alias, "path": rel_path})
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
        return snap_id

    def undo_last(self, workspace: "MultiRepoWorkspace") -> bool:
        """Revert the last snapshot. Returns True if something was undone."""
        if not self.snapshots:
            print("❌ No snapshots to undo.")
            return False
        snap = self.snapshots.pop()
        snap_dir = self.temp_dir / snap["id"]
        meta_file = snap_dir / "meta.json"
        if not meta_file.exists():
            print("❌ Snapshot data missing.")
            return False
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        try:
            repo = workspace.get_repo(meta["repo"])
            repo.write_file(meta["path"], meta["content"], confirm=False)
            print(f"↩️  Undid changes to {meta['repo']}/{meta['path']}")
        except Exception as exc:
            print(f"❌ Undo failed: {exc}")
            return False
        return True

    def cleanup(self) -> None:
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Multi-Repo Workspace + Codebase Indexer
# ---------------------------------------------------------------------------

class RepoHandle:
    """A single registered repository with its own sandboxed workspace."""

    def __init__(self, alias: str, root: Path):
        self.alias = alias
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index: Dict[str, Any] = {}
        self.indexed_at: Optional[float] = None

    def resolve(self, rel_path: str) -> Path:
        p = (self.root / rel_path).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"Path escapes workspace: {rel_path}")
        return p

    def list_files(self, rel_path: str = ".") -> List[Dict[str, Any]]:
        target = self.resolve(rel_path)
        if not target.exists():
            return []
        entries = []
        for item in sorted(target.iterdir()):
            name = item.name
            if name in IGNORE_DIRS or name in IGNORE_FILES:
                continue
            entries.append({
                "name": name,
                "path": str(item.relative_to(self.root)),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else 0,
            })
        return entries

    def read_file(self, rel_path: str) -> str:
        target = self.resolve(rel_path)
        if not target.is_file():
            raise FileNotFoundError(f"Not a file: {rel_path}")
        return target.read_text(encoding="utf-8", errors="replace")

    def write_file(self, rel_path: str, content: str, confirm: bool = True) -> bool:
        target = self.resolve(rel_path)
        if target.exists() and confirm:
            if not confirm_action(f"Overwrite existing file: {target}"):
                print(f"⏭️  Skipped: {target}")
                return False
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(f"✅ Wrote: {target}")
        return True

    def edit_file(self, rel_path: str, old_string: str, new_string: str, confirm: bool = True) -> bool:
        target = self.resolve(rel_path)
        if not target.is_file():
            print(f"❌ File not found: {target}")
            return False
        content = target.read_text(encoding="utf-8", errors="replace")
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

    def delete_file(self, rel_path: str, confirm: bool = True) -> bool:
        target = self.resolve(rel_path)
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

    def run_command(self, command: str, confirm: bool = True, cwd_override: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run a shell command inside this repo's root by default."""
        is_git_push = "git push" in command
        if confirm or is_git_push:
            extra = " (git push)" if is_git_push else ""
            if not confirm_action(f"Run command{extra}: {command}"):
                print(f"⏭️  Skipped command: {command}")
                return -1, "", "Cancelled by user"

        cwd = cwd_override or self.root
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(cwd),
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as exc:  # pragma: no cover - subprocess edge cases
            return -1, "", str(exc)


class MultiRepoWorkspace:
    """Manages a primary workspace plus multiple named repositories."""

    def __init__(self, primary_root: Path):
        self.primary = RepoHandle("primary", primary_root)
        self.repos: Dict[str, RepoHandle] = {"primary": self.primary}
        self.snapshots = SnapshotManager()

    def add_repo(self, alias: str, path: str) -> None:
        root = Path(path).resolve()
        if alias in self.repos:
            print(f"⚠️  Repo alias '{alias}' already exists; updating path.")
        self.repos[alias] = RepoHandle(alias, root)
        print(f"✅ Added repo '{alias}' → {root}")

    def remove_repo(self, alias: str) -> None:
        if alias == "primary":
            print("❌ Cannot remove the primary workspace.")
            return
        if alias not in self.repos:
            print(f"❌ Repo alias '{alias}' not found.")
            return
        del self.repos[alias]
        print(f"✅ Removed repo '{alias}'")

    def get_repo(self, alias: str) -> RepoHandle:
        if alias not in self.repos:
            raise KeyError(f"Repo alias '{alias}' not found. Known: {', '.join(self.repos)}")
        return self.repos[alias]

    def resolve_repo_path(self, spec: str) -> Tuple[str, str]:
        """
        Parse 'alias:path' or just 'path' → (alias, rel_path).
        Defaults to 'primary' if no alias prefix.
        """
        if ":" in spec:
            alias, rel_path = spec.split(":", 1)
            return alias.strip(), rel_path.strip()
        return "primary", spec

    # -- Indexing -----------------------------------------------------------

    def _extract_symbols(self, content: str, ext: str) -> List[str]:
        """Extract key symbols from source text using lightweight regex."""
        symbols: List[str] = []
        if ext == ".py":
            for m in re.finditer(r"^\s*(?:def|class|async def)\s+(\w+)", content, re.MULTILINE):
                symbols.append(m.group(1))
            for m in re.finditer(r"^\s*(?:from|import)\s+([\w.]+)", content, re.MULTILINE):
                symbols.append(m.group(1))
        elif ext in (".js", ".ts", ".jsx", ".tsx"):
            for m in re.finditer(r"(?:function|class|const|let|var)\s+(\w+)", content):
                symbols.append(m.group(1))
            for m in re.finditer(r"(?:import|export).*from\s+['\"]([^'\"]+)['\"]", content):
                symbols.append(m.group(1))
        return symbols[:50]  # cap to avoid bloating the index

    def index_repo(self, alias: str) -> Dict[str, Any]:
        """Build a lightweight index for a repo: file tree + key symbols."""
        repo = self.get_repo(alias)
        index: Dict[str, Any] = {
            "alias": alias,
            "root": str(repo.root),
            "files": {},
            "symbols": {},       # symbol -> [file paths]
            "indexed_at": datetime.now().isoformat(),
        }

        def walk(dir_path: Path, rel_prefix: str = "") -> None:
            try:
                items = list(dir_path.iterdir())
            except PermissionError:
                return
            for item in sorted(items):
                name = item.name
                if name in IGNORE_DIRS or name in IGNORE_FILES:
                    continue
                rel = f"{rel_prefix}/{name}" if rel_prefix else name
                if item.is_dir():
                    walk(item, rel)
                elif item.is_file() and item.stat().st_size < 2_000_000:  # skip huge files
                    ext = item.suffix.lower()
                    index["files"][rel] = {
                        "size": item.stat().st_size,
                        "ext": ext,
                        "is_code": ext in CODE_EXTENSIONS,
                        "is_config": ext in CONFIG_EXTENSIONS,
                    }
                    if ext in CODE_EXTENSIONS or ext in CONFIG_EXTENSIONS:
                        try:
                            content = item.read_text(encoding="utf-8", errors="replace")
                            syms = self._extract_symbols(content, ext)
                            for s in syms:
                                index["symbols"].setdefault(s, []).append(rel)
                        except Exception:
                            pass

        walk(repo.root)
        repo.index = index
        from time import time as _time
        repo.indexed_at = _time()
        return index

    def index_all(self) -> Dict[str, Dict[str, Any]]:
        return {alias: self.index_repo(alias) for alias in self.repos}

    def search_code(self, query: str, alias: Optional[str] = None) -> List[Dict[str, Any]]:
        """Grep-search across repo(s). Returns matches with file/line/context."""
        results: List[Dict[str, Any]] = []
        aliases = [alias] if alias else list(self.repos.keys())
        for a in aliases:
            repo = self.get_repo(a)
            for rel_path, finfo in repo.index.get("files", {}).items():
                if not finfo["is_code"] and not finfo["is_config"]:
                    continue
                abs_path = repo.resolve(rel_path)
                try:
                    lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
                except Exception:
                    continue
                for i, line in enumerate(lines, start=1):
                    if query.lower() in line.lower():
                        results.append({
                            "repo": a,
                            "file": rel_path,
                            "line": i,
                            "match": line.strip()[:200],
                        })
                        if len(results) >= 100:
                            return results
        return results

    def find_files(self, pattern: str, alias: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find files matching a glob pattern across repo(s)."""
        results: List[Dict[str, Any]] = []
        aliases = [alias] if alias else list(self.repos.keys())
        for a in aliases:
            repo = self.get_repo(a)
            for rel_path in repo.index.get("files", {}):
                if re.search(pattern, rel_path, re.IGNORECASE):
                    results.append({"repo": a, "file": rel_path})
                    if len(results) >= 100:
                        return results
        return results


# ---------------------------------------------------------------------------
# Pydantic action model for structured JSON tool-calling
# ---------------------------------------------------------------------------

if HAS_PYDANTIC:

    class AgentAction(BaseModel):
        """Structured action emitted by the SWE agent's reasoning model."""
        tool: str = Field(
            description="Tool to call: list_files, read_file, search_code, write_file, "
                        "edit_file, delete_file, run_command, git_diff, git_status, done"
        )
        args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
        reasoning: str = Field(default="", description="Why this action is being taken")

else:

    class AgentAction:  # type: ignore[no-redef]
        def __init__(self, tool: str, args: Optional[Dict[str, Any]] = None, reasoning: str = ""):
            self.tool = tool
            self.args = args or {}
            self.reasoning = reasoning


# ---------------------------------------------------------------------------
# Autonomous SWE Agent: Planner → Executor → Verifier loop
# ---------------------------------------------------------------------------

class SWEAgent:
    """
    Autonomous software engineering agent powered by Revolver.

    Loop:
      1. Plan: Revolver (TIER_HEAVY) produces a multi-step implementation plan.
      2. Execute & Verify: Revolver (TIER_SPRINT for search, TIER_HEAVY for edits)
         emits JSON tool actions; each is executed, results observed, and
         verification commands run. Self-fix on failure up to max_steps.
    """

    TOOL_DESCRIPTIONS = {
        "list_files": "List files/dirs in a repo path. Args: {repo, path?}",
        "read_file": "Read a file's contents. Args: {repo, path}",
        "search_code": "Grep-search for a query across repo(s). Args: {query, repo?}",
        "find_files": "Find files matching a regex pattern. Args: {pattern, repo?}",
        "write_file": "Write/create a file. Args: {repo, path, content}",
        "edit_file": "Replace old_string with new_string in a file. Args: {repo, path, old_string, new_string}",
        "delete_file": "Delete a file (destructive, requires confirmation unless --auto). Args: {repo, path}",
        "run_command": "Run a shell command inside a repo. Args: {repo, command, confirm?}",
        "git_diff": "Show git diff in a repo. Args: {repo}",
        "git_status": "Show git status in a repo. Args: {repo}",
        "done": "Signal task completion. Args: {summary}",
    }

    def __init__(
        self,
        workspace: MultiRepoWorkspace,
        rev: Revolver,
        max_steps: int = DEFAULT_MAX_STEPS,
        auto: bool = False,
    ):
        self.workspace = workspace
        self.rev = rev
        self.max_steps = max_steps
        self.auto = auto
        self.step = 0
        self.history: List[Dict[str, Any]] = []

    # -- LLM calls with tier-aware routing --------------------------------

    def _llm_plan(self, task: str, context: str) -> str:
        """Planning call using heavy-tier chambers."""
        racers = select_chambers_by_tier(self.rev, TIER_HEAVY, self.rev._chambers[:], [])
        if not racers:
            racers = self.rev._chambers[:2]
        messages = [
            {"role": "system", "content": self.rev.system_prompt},
            {
                "role": "user",
                "content": (
                    "You are an expert software engineer creating an implementation plan.\n"
                    f"TASK: {task}\n\n"
                    f"CONTEXT (codebase index):\n{context}\n\n"
                    "Produce a concise, ordered multi-step plan. Each step should name the "
                    "file(s) to touch and the change to make. Keep it under 200 words."
                ),
            },
        ]
        # Use the heaviest chamber directly for planning.
        chamber = racers[0]
        content, outcome = self.rev._call_chamber(chamber, messages)
        self.rev._apply_outcome(chamber, outcome)
        return content or ""

    def _llm_decide(
        self,
        task: str,
        context: str,
        history: str,
        available_tools: str,
    ) -> Union[AgentAction, Dict[str, Any], None]:
        """
        Ask Revolver (sprint tier for fast inspection, heavy for edits/planning)
        to pick the next tool action as JSON.
        """
        # Decide tier: complex reasoning → heavy, simple lookup → sprint.
        needs_heavy = any(k in task.lower() for k in ("debug", "fix", "architect", "design", "refactor", "plan"))
        tier = TIER_HEAVY if needs_heavy else TIER_SPRINT
        racers = select_chambers_by_tier(self.rev, tier, self.rev._chambers[:], [])
        if not racers:
            racers = self.rev._chambers[:2]

        system_prompt = (
            "You are an autonomous SWE agent. Respond ONLY with a JSON object matching this schema:\n"
            '{"tool": "<tool_name>", "args": {...}, "reasoning": "<why>"}\n\n'
            f"Available tools:\n{available_tools}\n\n"
            "Rules:\n"
            "- If the task is complete, use tool 'done' with a summary.\n"
            "- Prefer read-only tools (list_files, read_file, search_code, find_files) first.\n"
            "- Before editing, read the file. Before running tests, read them.\n"
            "- All paths are relative to a repo. Use 'repo' field to pick a repo alias.\n"
            "- For run_command, set 'confirm': false only for read-only/test commands.\n"
        )

        user_content = (
            f"TASK: {task}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"HISTORY (recent steps):\n{history}\n\n"
            "Choose the next single action. Respond with ONLY the JSON object."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        chamber = racers[0]
        content, outcome = self.rev._call_chamber(chamber, messages)
        self.rev._apply_outcome(chamber, outcome)

        if not content:
            return None

        # Try to parse JSON from the response (strip fences).
        raw = self.rev.sanitize_draft(content)
        raw = self._strip_json_fences(raw)
        try:
            data = json.loads(raw)
            if HAS_PYDANTIC:
                return AgentAction(**data)
            return data
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            print(f"⚠️  Could not parse agent action JSON: {exc}")
            print(f"   Raw response: {raw[:300]}")
            return None

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            nl = text.find("\n")
            if nl != -1:
                text = text[nl + 1 :]
            else:
                text = ""
        if text.rstrip().endswith("```"):
            nl = text.rfind("\n")
            if nl != -1:
                text = text[:nl]
            else:
                text = ""
        return text.strip()

    # -- Tool execution ---------------------------------------------------

    def _execute_action(
        self,
        action: Union[AgentAction, Dict[str, Any]],
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Execute a single agent action. Returns (success, output, new_context_hint).
        """
        if HAS_PYDANTIC and isinstance(action, AgentAction):
            tool = action.tool
            args = action.args
            reasoning = action.reasoning
        else:
            tool = action.get("tool", "")
            args = action.get("args", {})
            reasoning = action.get("reasoning", "")

        print(f"\n🔧 [Step {self.step}] {tool}")
        print(f"   Reasoning: {reasoning[:150]}")
        print(f"   Args: {json.dumps(args, ensure_ascii=False)[:200]}")

        repo_alias = args.get("repo", "primary")
        rel_path = args.get("path", "")

        try:
            if tool == "list_files":
                repo = self.workspace.get_repo(repo_alias)
                entries = repo.list_files(args.get("path", "."))
                return True, json.dumps(entries, indent=2), None

            if tool == "read_file":
                repo = self.workspace.get_repo(repo_alias)
                content = repo.read_file(rel_path)
                return True, content, None

            if tool == "search_code":
                results = self.workspace.search_code(args["query"], args.get("repo"))
                return True, json.dumps(results[:50], indent=2), None

            if tool == "find_files":
                results = self.workspace.find_files(args["pattern"], args.get("repo"))
                return True, json.dumps(results[:50], indent=2), None

            if tool == "write_file":
                repo = self.workspace.get_repo(repo_alias)
                content = args["content"]
                # Backup before write.
                self.workspace.snapshots.snapshot(self.workspace, repo_alias, rel_path)
                ok = repo.write_file(rel_path, content, confirm=not self.auto)
                return ok, f"write_file {rel_path}: {'ok' if ok else 'skipped'}", None

            if tool == "edit_file":
                repo = self.workspace.get_repo(repo_alias)
                # Backup before edit.
                self.workspace.snapshots.snapshot(self.workspace, repo_alias, rel_path)
                ok = repo.edit_file(
                    rel_path,
                    args["old_string"],
                    args["new_string"],
                    confirm=not self.auto,
                )
                return ok, f"edit_file {rel_path}: {'ok' if ok else 'skipped'}", None

            if tool == "delete_file":
                # Destructive: always require confirmation unless --auto AND user pre-approved.
                if not self.auto:
                    repo = self.workspace.get_repo(repo_alias)
                    if not confirm_action(f"DELETE {repo_alias}:{rel_path}? This cannot be undone."):
                        return False, "delete_file: cancelled by user", None
                repo = self.workspace.get_repo(repo_alias)
                self.workspace.snapshots.snapshot(self.workspace, repo_alias, rel_path)
                ok = repo.delete_file(rel_path, confirm=False)
                return ok, f"delete_file {rel_path}: {'ok' if ok else 'skipped'}", None

            if tool == "run_command":
                repo = self.workspace.get_repo(repo_alias)
                command = args["command"]
                confirm = args.get("confirm", not self.auto)
                # git push always needs explicit confirmation.
                if "git push" in command:
                    confirm = True
                code, out, err = repo.run_command(command, confirm=confirm)
                output = f"exit_code={code}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                return code == 0, output, None

            if tool == "git_diff":
                repo = self.workspace.get_repo(repo_alias)
                code, out, err = repo.run_command("git diff", confirm=self.auto)
                return code == 0, out or err, None

            if tool == "git_status":
                repo = self.workspace.get_repo(repo_alias)
                code, out, err = repo.run_command("git status", confirm=self.auto)
                return code == 0, out or err, None

            if tool == "done":
                return True, f"DONE: {args.get('summary', 'Task complete')}", "done"

            return False, f"Unknown tool: {tool}", None

        except KeyError as exc:
            return False, f"Repo '{repo_alias}' not found: {exc}", None
        except Exception as exc:  # pragma: no cover - runtime errors
            return False, f"Error executing {tool}: {exc}", None

    # -- Verification -----------------------------------------------------

    def _verify(
        self,
        repo_alias: str,
        test_commands: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Run verification commands (tests/linters/build) in a repo."""
        if not test_commands:
            # Auto-detect common test commands.
            repo = self.workspace.get_repo(repo_alias)
            candidates = [
                "python -m pytest -q",
                "python -m unittest discover -v",
                "python -m unittest test_revolver.py -v",
                "npm test",
            ]
            test_commands = []
            for cmd in candidates:
                # Check if the command makes sense (e.g., pytest exists or test files exist).
                test_commands.append(cmd)

        all_ok = True
        outputs: List[str] = []
        for cmd in test_commands:
            repo = self.workspace.get_repo(repo_alias)
            code, out, err = repo.run_command(cmd, confirm=self.auto)
            outputs.append(f"=== {cmd} ===\n{out}\n{err}")
            if code != 0:
                all_ok = False
        return all_ok, "\n".join(outputs)

    # -- Main loop --------------------------------------------------------

    def run(
        self,
        task: str,
        test_commands: Optional[List[str]] = None,
        index_first: bool = True,
    ) -> bool:
        """
        Run the autonomous SWE loop.

        Returns True if the task completed successfully (done tool called
        and verification passed), False otherwise.
        """
        if index_first:
            print("📚 Indexing codebase...")
            self.workspace.index_all()
            print("   Index complete.")

        # Build compact context from indexes.
        context_parts: List[str] = []
        for alias, idx in self.workspace.repos.items():
            files = idx.index.get("files", {})
            code_files = [f for f, finfo in files.items() if finfo.get("is_code")]
            context_parts.append(
                f"Repo '{alias}': {len(files)} files, "
                f"{len(code_files)} code files. "
                f"Key symbols: {', '.join(list(idx.index.get('symbols', {}).keys())[:20])}"
            )
        context = "\n".join(context_parts)

        # Plan phase (heavy tier).
        print("\n🎯 Plan phase — generating implementation plan...")
        plan = self._llm_plan(task, context)
        print(f"📋 Plan:\n{plan}\n")

        # Execute & verify loop.
        while self.step < self.max_steps:
            self.step += 1

            # Build history summary.
            recent = self.history[-5:] if self.history else []
            history_text = "\n".join(
                f"  {h['step']}: {h['tool']} → {h['success']} | {h['output'][:120]}"
                for h in recent
            ) or "(none yet)"

            # Decide next action.
            action = self._llm_decide(task, context, history_text, json.dumps(self.TOOL_DESCRIPTIONS, indent=2))
            if action is None:
                print("⚠️  Agent produced no action. Stopping.")
                break

            # Execute.
            success, output, hint = self._execute_action(action)
            self.history.append({
                "step": self.step,
                "tool": action.tool if HAS_PYDANTIC else action.get("tool", ""),
                "success": success,
                "output": output[:500],
            })

            if hint == "done":
                print(f"\n✅ Agent reports completion: {output}")
                # Run verification.
                repo_alias = (
                    action.args.get("repo", "primary")
                    if HAS_PYDANTIC and isinstance(action, AgentAction)
                    else action.get("args", {}).get("repo", "primary")
                )
                ok, verify_out = self._verify(repo_alias, test_commands)
                print(f"\n🔍 Verification:\n{verify_out}")
                return ok

            if not success:
                print(f"   ⚠️  Action failed: {output[:200]}")
                # Self-fix: continue the loop; the agent sees the failure in history.
                if self.step >= self.max_steps:
                    print("❌ Max steps reached without completion.")
                    return False
                continue

            print(f"   ✅ Action succeeded.")
            # Periodically re-verify after meaningful edits.
            if action.tool if HAS_PYDANTIC else action.get("tool", "") in ("write_file", "edit_file", "delete_file"):
                repo_alias = (
                    action.args.get("repo", "primary")
                    if HAS_PYDANTIC and isinstance(action, AgentAction)
                    else action.get("args", {}).get("repo", "primary")
                )
                ok, verify_out = self._verify(repo_alias, test_commands)
                if not ok:
                    print(f"   ⚠️  Verification failed:\n{verify_out[:500]}")
                    # Continue loop; agent can self-fix.
                else:
                    print("   ✅ Verification passed.")

        print(f"❌ Max steps ({self.max_steps}) reached without completion.")
        return False


# ---------------------------------------------------------------------------
# CLI command implementations
# ---------------------------------------------------------------------------

def cmd_agent(args: argparse.Namespace) -> int:
    """
    revolver agent — interactive coding agent with Revolver as its brain.

    The agent uses Revolver to generate responses and can inspect/create/edit
    files and run shell commands with explicit user confirmation.
    """
    workspace_root = Path(args.workspace).resolve() if args.workspace else Path.cwd()
    workspace = MultiRepoWorkspace(workspace_root)

    print("\n🤖 Revolver Agent — Interactive Mode")
    print(f"   Workspace: {workspace.root}")
    print("   Commands: /read <repo:path> | /write <repo:path> | /edit <repo:path> | /run <cmd> |")
    print("             /ls <repo:path> | /repo add <alias> <path> | /repo list | /undo | /quit\n")

    rev = build_revolver(race_mode=args.race_mode)
    tag_chambers_by_tier(rev)
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

        # --- Repo management ---
        if user_input.startswith("/repo "):
            parts = user_input[6:].strip().split()
            if not parts:
                print("   Usage: /repo add <alias> <path> | /repo list | /repo remove <alias>")
                continue
            sub = parts[0]
            if sub == "list":
                for alias in workspace.repos:
                    repo = workspace.repos[alias]
                    print(f"   • {alias} → {repo.root}")
            elif sub == "add" and len(parts) == 3:
                workspace.add_repo(parts[1], parts[2])
            elif sub == "remove" and len(parts) == 2:
                workspace.remove_repo(parts[1])
            else:
                print("   Usage: /repo add <alias> <path> | /repo list | /repo remove <alias>")
            continue

        if user_input == "/undo":
            workspace.snapshots.undo_last(workspace)
            continue

        # --- File system slash commands ---
        if user_input.startswith("/ls "):
            spec = user_input[4:].strip()
            alias, rel_path = workspace.resolve_repo_path(spec)
            try:
                repo = workspace.get_repo(alias)
                entries = repo.list_files(rel_path)
                if not entries:
                    print("   (empty or not found)")
                for e in entries:
                    marker = "📁" if e["is_dir"] else "📄"
                    print(f"   {marker} [{alias}] {e['path']}")
            except KeyError as exc:
                print(f"❌ {exc}")
            continue

        if user_input.startswith("/read "):
            spec = user_input[6:].strip()
            alias, rel_path = workspace.resolve_repo_path(spec)
            try:
                repo = workspace.get_repo(alias)
                content = repo.read_file(rel_path)
                print(f"\n--- [{alias}] {rel_path} ---\n{content}\n--- end ---\n")
            except Exception as exc:
                print(f"❌ {exc}")
            continue

        if user_input.startswith("/write "):
            spec = user_input[7:].strip()
            alias, rel_path = workspace.resolve_repo_path(spec)
            print(f"   Enter content for [{alias}] {rel_path}. Type /end on its own line to finish.")
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
            workspace.get_repo(alias).write_file(rel_path, content)
            continue

        if user_input.startswith("/edit "):
            spec = user_input[6:].strip()
            alias, rel_path = workspace.resolve_repo_path(spec)
            print(f"   Enter old string for [{alias}] {rel_path} (type /end on its own line to finish):")
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
            workspace.get_repo(alias).edit_file(rel_path, old_string, new_string)
            continue

        if user_input.startswith("/run "):
            command = user_input[5:].strip()
            # Default to primary repo; command runs in its cwd.
            repo = workspace.get_repo("primary")
            code, out, err = repo.run_command(command, confirm=not args.auto)
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
            wire = confirm_action("Wire Revolver System as the project's AI brain", default=False)
            scaffold_project(workspace.primary, name, ptype, wire)
            continue

        if user_input.startswith("/swe ") or user_input == "/swe":
            task = user_input[5:].strip() if user_input.startswith("/swe ") else ""
            if not task:
                task = input("   Enter SWE task description: ").strip()
            if not task:
                print("   ❌ No task provided.")
                continue
            auto = args.auto or confirm_action("Run in auto-approve mode?", default=False)
            agent = SWEAgent(workspace, rev, max_steps=args.max_steps, auto=auto)
            test_cmds = args.test_cmd.split(",") if args.test_cmd else None
            success = agent.run(task, test_commands=test_cmds)
            print(f"\n{'✅ SWE task completed.' if success else '❌ SWE task did not complete.'}")
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


def cmd_swe(args: argparse.Namespace) -> int:
    """
    revolver swe — autonomous SWE agent (non-interactive).

    Runs the plan → execute → verify loop on a given task.
    """
    workspace_root = Path(args.workspace).resolve() if args.workspace else Path.cwd()
    workspace = MultiRepoWorkspace(workspace_root)

    # Add any --repo aliases.
    for repo_spec in args.repo:
        if "=" in repo_spec:
            alias, path = repo_spec.split("=", 1)
            workspace.add_repo(alias.strip(), path.strip())
        else:
            print(f"❌ Invalid --repo format: {repo_spec} (expected alias=path)")
            return 1

    rev = build_revolver(race_mode=args.race_mode)
    tag_chambers_by_tier(rev)

    agent = SWEAgent(workspace, rev, max_steps=args.max_steps, auto=args.auto)
    test_cmds = args.test_cmd.split(",") if args.test_cmd else None

    print(f"\n🚀 Revolver SWE Agent — Autonomous Mode")
    print(f"   Task: {args.task}")
    print(f"   Workspace: {workspace.root}")
    print(f"   Max steps: {args.max_steps} | Auto: {args.auto}")
    if args.repo:
        print(f"   Repos: {', '.join(workspace.repos.keys())}")

    success = agent.run(args.task, test_commands=test_cmds)
    print(f"\n{'✅ SWE task completed successfully.' if success else '❌ SWE task did not complete.'}")
    return 0 if success else 1


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

    workspace = MultiRepoWorkspace(Path.cwd())
    wire = args.wire_revolver or (
        not args.yes and confirm_action("Wire Revolver System as the project's AI brain", default=False)
    )

    project_dir = scaffold_project(workspace.primary, name, ptype, wire)
    print(f"\n🎉 Project created at: {project_dir}")
    print("   Next steps:")
    print(f"     cd {project_dir}")
    print("     pip install -r requirements.txt")
    return 0


# ---------------------------------------------------------------------------
# Project scaffolding (BYOK templates, no hardcoded keys)
# ---------------------------------------------------------------------------

PROJECT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "python-bot": {
        "main.py": '''"""
{project_name} — AI bot powered by Revolver System.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from revolver import Revolver


def build_bot() -> Revolver:
    rev = Revolver(
        system_prompt="You are a helpful AI assistant.",
        race_mode="hedged",
    )
    # Auto-load API keys from environment variables (BYOK — no hardcoded keys).
    for provider in ("groq", "gemini", "openrouter", "openai"):
        key = os.environ.get(f"{provider.upper()}_API_KEY", "")
        if key:
            rev.add_provider(provider=provider, api_key=key)
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
import os
from revolver import Revolver


def main():
    rev = Revolver(system_prompt="You are a helpful assistant.")
    # Auto-load API keys from environment variables (BYOK).
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        rev.add_provider("groq", api_key=key)
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
import os
from fastapi import FastAPI
from pydantic import BaseModel
from revolver import Revolver

app = FastAPI(title="{project_name}")
rev = Revolver(system_prompt="You are a helpful AI assistant.")
# Auto-load API keys from environment variables (BYOK).
key = os.environ.get("GROQ_API_KEY", "")
if key:
    rev.add_provider("groq", api_key=key)


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
import os
from revolver import Revolver


def main():
    parser = argparse.ArgumentParser(description="{project_name}")
    parser.add_argument("message", help="Message to send to Revolver")
    parser.add_argument("--provider", default="groq", help="Provider to use")
    args = parser.parse_args()

    rev = Revolver()
    # Auto-load API key from environment variable (BYOK).
    env_var = f"{args.provider.upper()}_API_KEY"
    key = os.environ.get(env_var, "")
    if key:
        rev.add_provider(args.provider, api_key=key)
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
    workspace: Any,
    project_name: str,
    project_type: str,
    wire_revolver: bool,
) -> Path:
    """
    Create a new project directory with scaffold files.

    If wire_revolver is True, copies revolver.py into the project.
    """
    # workspace may be AgentWorkspace (legacy) or RepoHandle (new).
    root = workspace.root if hasattr(workspace, "root") else workspace
    project_dir = root / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    templates = PROJECT_TEMPLATES[project_type]

    for filename, template_content in templates.items():
        rendered = render_template(template_content, project_name)
        (project_dir / filename).write_text(rendered, encoding="utf-8")
        print(f"✅ Created: {project_dir / filename}")

    revolver_source = Path(__file__).resolve().parent / "revolver.py"
    if wire_revolver and revolver_source.exists():
        dest = project_dir / "revolver.py"
        shutil.copy2(revolver_source, dest)
        print(f"✅ Copied revolver.py into project: {dest}")
        print("✅ Revolver System wired as the project's AI brain.")
    elif wire_revolver:
        print("⚠️  revolver.py not found next to this CLI; skipping auto-wire.")

    return project_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="revolver",
        description="Revolver System — Full Autonomous SWE & Multi-Repo Agent (Linux & Windows)",
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
    p_agent.add_argument("--auto", action="store_true", help="Auto-approve safe workspace edits & read-only/test commands")
    p_agent.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Max SWE agent steps")
    p_agent.add_argument("--test-cmd", default=None, help="Comma-separated verification commands (e.g. 'pytest,python -m unittest')")
    p_agent.set_defaults(func=cmd_agent)

    # --- swe ---
    p_swe = sub.add_parser("swe", help="Autonomous SWE agent (plan → execute → verify loop)")
    p_swe.add_argument("task", help="SWE task description")
    p_swe.add_argument("--workspace", default=".", help="Workspace directory (default: current dir)")
    p_swe.add_argument("--repo", action="append", default=[], help="Additional repo: alias=path (repeatable)")
    p_swe.add_argument("--race-mode", choices=["hedged", "concurrent"], default="hedged", help="Race mode")
    p_swe.add_argument("--auto", action="store_true", help="Auto-approve safe workspace edits & read-only/test commands")
    p_swe.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Max SWE agent steps")
    p_swe.add_argument("--test-cmd", default=None, help="Comma-separated verification commands")
    p_swe.set_defaults(func=cmd_swe)

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
