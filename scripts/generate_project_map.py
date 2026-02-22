from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
MAP_DIR = DOCS_DIR / "project-map"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "logs",
    "outputs",
    "htmlcov",
    "tools",
}
EXCLUDED_REL_PREFIXES = ("data/nltk_data",)

ENV_GET_RE = re.compile(
    r"os\.(?:getenv|environ\.get)\(\s*['\"]([A-Z0-9_]+)['\"](?:\s*,\s*([^)]+))?",
)
PS_ENV_RE = re.compile(r"\$env:([A-Z0-9_]+)")


@dataclass(frozen=True)
class ModuleInfo:
    path: Path
    module: str
    is_package_init: bool


def ensure_docs_dirs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)


def rel_posix(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def should_skip(path: Path) -> bool:
    rel = rel_posix(path)
    parts = rel.split("/")
    if any(part in EXCLUDED_DIR_NAMES for part in parts):
        return True
    return any(rel.startswith(prefix) for prefix in EXCLUDED_REL_PREFIXES)


def visible_entries(directory: Path) -> list[Path]:
    entries = []
    for entry in directory.iterdir():
        if should_skip(entry):
            continue
        entries.append(entry)
    return sorted(entries, key=lambda item: (not item.is_dir(), item.name.lower()))


def build_structure_tree() -> list[str]:
    lines = ["."]

    def walk(directory: Path, prefix: str) -> None:
        entries = visible_entries(directory)
        for index, entry in enumerate(entries):
            is_last = index == len(entries) - 1
            branch = "`-- " if is_last else "|-- "
            lines.append(f"{prefix}{branch}{entry.name}")
            if entry.is_dir():
                child_prefix = f"{prefix}{'    ' if is_last else '|   '}"
                walk(entry, child_prefix)

    walk(PROJECT_ROOT, "")
    return lines


def module_name_from_path(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT)
    if rel == Path("app.py"):
        return "app"

    no_suffix = rel.with_suffix("")
    parts = list(no_suffix.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def collect_python_modules() -> dict[str, ModuleInfo]:
    modules: dict[str, ModuleInfo] = {}
    for path in sorted(PROJECT_ROOT.rglob("*.py")):
        if should_skip(path):
            continue
        module_name = module_name_from_path(path)
        if not module_name:
            continue
        modules[module_name] = ModuleInfo(
            path=path,
            module=module_name,
            is_package_init=(path.stem == "__init__"),
        )
    return modules


def parse_python_ast(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return None


def resolve_relative_module(
    *,
    current_module: str,
    is_package_init: bool,
    level: int,
    module: str | None,
) -> str:
    base_parts = current_module.split(".")
    if not is_package_init:
        base_parts = base_parts[:-1]
    if level > 0:
        trim = max(0, level - 1)
        if trim:
            base_parts = base_parts[: len(base_parts) - trim]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(part for part in base_parts if part)


def extract_internal_imports(
    module_info: ModuleInfo,
    *,
    known_modules: set[str],
) -> set[str]:
    tree = parse_python_ast(module_info.path)
    if tree is None:
        return set()

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = resolve_relative_module(
                current_module=module_info.module,
                is_package_init=module_info.is_package_init,
                level=node.level or 0,
                module=node.module,
            )
            if node.module is None:
                matched_submodule = False
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    candidate = f"{base}.{alias.name}" if base else alias.name
                    if candidate in known_modules:
                        imports.add(candidate)
                        matched_submodule = True
                if not matched_submodule and base:
                    imports.add(base)
            elif base:
                imports.add(base)

    internal: set[str] = set()
    for imported in imports:
        if imported == "app" or imported.startswith("kokoro_tts"):
            internal.add(imported)
    return internal


def module_layer(module: str) -> str:
    if module == "app":
        return "app"
    if module.startswith("kokoro_tts.application"):
        return "application"
    if module.startswith("kokoro_tts.domain"):
        return "domain"
    if module.startswith("kokoro_tts.storage"):
        return "storage"
    if module.startswith("kokoro_tts.integrations"):
        return "integrations"
    if module.startswith("kokoro_tts.ui"):
        return "ui"
    if module.startswith(
        (
            "kokoro_tts.config",
            "kokoro_tts.constants",
            "kokoro_tts.logging_config",
            "kokoro_tts.main",
            "kokoro_tts.runtime",
            "kokoro_tts.utils",
        )
    ):
        return "core"
    return "other"


def identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def build_dependency_edges(
    modules: dict[str, ModuleInfo],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    known = set(modules)
    module_edges: set[tuple[str, str]] = set()
    layer_edges: set[tuple[str, str]] = set()
    for module_name, info in modules.items():
        imported_modules = extract_internal_imports(info, known_modules=known)
        for imported in imported_modules:
            if imported not in known:
                continue
            module_edges.add((module_name, imported))
            source_layer = module_layer(module_name)
            target_layer = module_layer(imported)
            if source_layer != "other" and target_layer != "other":
                layer_edges.add((source_layer, target_layer))
    return module_edges, layer_edges


def write_dependency_artifacts(
    *,
    module_edges: set[tuple[str, str]],
    layer_edges: set[tuple[str, str]],
) -> None:
    layer_labels = {
        "app": "app.py facade",
        "application": "application",
        "domain": "domain",
        "storage": "storage",
        "integrations": "integrations",
        "ui": "ui",
        "core": "core/config",
    }

    mermaid_lines = ["graph TD"]
    for layer, label in layer_labels.items():
        mermaid_lines.append(f'  {identifier(layer)}["{label}"]')
    for src, dst in sorted(layer_edges):
        mermaid_lines.append(f"  {identifier(src)} --> {identifier(dst)}")

    mermaid_lines.append("")
    mermaid_lines.append("%% Module-level edges (trimmed to internal package only)")
    mermaid_lines.append("graph LR")
    for src, dst in sorted(module_edges):
        src_id = identifier(src)
        dst_id = identifier(dst)
        mermaid_lines.append(f'  {src_id}["{src}"] --> {dst_id}["{dst}"]')

    (MAP_DIR / "deps-graph.mmd").write_text("\n".join(mermaid_lines) + "\n", encoding="utf-8")

    edge_lines = ["source\ttarget"]
    for src, dst in sorted(module_edges):
        edge_lines.append(f"{src}\t{dst}")
    (MAP_DIR / "deps-graph.tsv").write_text("\n".join(edge_lines) + "\n", encoding="utf-8")


def format_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args: list[str] = []

    positional = list(node.args.posonlyargs) + list(node.args.args)
    pos_defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
    for arg_node, default_node in zip(positional, pos_defaults):
        if arg_node.arg == "self":
            continue
        if default_node is None:
            args.append(arg_node.arg)
        else:
            args.append(f"{arg_node.arg}={ast.unparse(default_node)}")

    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    elif node.args.kwonlyargs:
        args.append("*")

    for kw_arg, kw_default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        if kw_arg.arg == "self":
            continue
        if kw_default is None:
            args.append(kw_arg.arg)
        else:
            args.append(f"{kw_arg.arg}={ast.unparse(kw_default)}")

    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")

    return f"{node.name}({', '.join(args)})"


def extract_top_level_functions(path: Path) -> list[str]:
    tree = parse_python_ast(path)
    if tree is None:
        return []
    functions: list[str] = []
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            functions.append(format_function_signature(node))
    return functions


def extract_ui_callback_map() -> dict[str, list[str]]:
    callbacks: dict[str, list[str]] = {}
    for path in sorted((PROJECT_ROOT / "kokoro_tts" / "ui").rglob("*.py")):
        tree = parse_python_ast(path)
        if tree is None:
            continue
        rel = rel_posix(path)
        file_callbacks: list[str] = []
        for node in tree.body:  # type: ignore[attr-defined]
            if not isinstance(node, ast.ClassDef):
                continue
            for body_node in node.body:
                if isinstance(body_node, ast.FunctionDef) and body_node.name.startswith("_on_"):
                    file_callbacks.append(f"{node.name}.{body_node.name}")
        if file_callbacks:
            callbacks[rel] = sorted(file_callbacks)
    return callbacks


def extract_cli_scripts() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for path in sorted((PROJECT_ROOT / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if 'if __name__ == "__main__"' not in text:
            continue
        tree = parse_python_ast(path)
        if tree is None:
            continue
        doc = ast.get_docstring(tree) or ""
        summary = doc.strip().splitlines()[0].strip() if doc.strip() else "Script entrypoint"
        entries.append((rel_posix(path), summary))
    return entries


def write_routes_artifact() -> None:
    app_functions = extract_top_level_functions(PROJECT_ROOT / "app.py")
    ui_callbacks = extract_ui_callback_map()
    cli_scripts = extract_cli_scripts()

    lines = ["# Route Map", ""]
    lines.append("## Public App API Surface (`app.py`)")
    lines.append("")
    for func in app_functions:
        lines.append(f"- `{func}`")
    lines.append("")

    lines.append("## UI Callback Surface (Tkinter)")
    lines.append("")
    total_callbacks = sum(len(items) for items in ui_callbacks.values())
    lines.append(f"- Total `_on_*` callbacks discovered: `{total_callbacks}`")
    lines.append("")
    for path, callbacks in ui_callbacks.items():
        lines.append(f"### `{path}`")
        for callback in callbacks:
            lines.append(f"- `{callback}`")
        lines.append("")

    lines.append("## CLI Entrypoints (`scripts/*.py`)")
    lines.append("")
    for script_path, summary in cli_scripts:
        lines.append(f"- `{script_path}`: {summary}")
    lines.append("")

    (MAP_DIR / "routes.md").write_text("\n".join(lines), encoding="utf-8")


def parse_create_table_sql(sql: str) -> tuple[str, list[tuple[str, str]], list[str]]:
    table_name_match = re.search(r'CREATE TABLE IF NOT EXISTS "([^"]+)"', sql, re.IGNORECASE)
    table_name = table_name_match.group(1) if table_name_match else "unknown_table"
    columns: list[tuple[str, str]] = []
    constraints: list[str] = []
    for raw_line in sql.splitlines():
        line = raw_line.strip().rstrip(",")
        if not line or line.startswith("CREATE TABLE") or line == ")":
            continue
        if line.startswith('"'):
            col_match = re.match(r'"([^"]+)"\s+(.+)', line)
            if not col_match:
                continue
            columns.append((col_match.group(1), col_match.group(2)))
        else:
            constraints.append(line)
    return table_name, columns, constraints


def write_schema_artifact() -> None:
    from kokoro_tts.storage.morphology_repository import MorphologyRepository

    repo = MorphologyRepository(enabled=True, db_path=":memory:")
    table_sql = [
        repo._sql_create_lexemes_table(),
        repo._sql_create_occurrences_table(),
        repo._sql_create_expressions_table(),
    ]
    index_sql = repo._sql_create_occurrence_indexes()

    lines = ["# Data Schema", ""]
    lines.append("## SQLite (`MORPH_DB_PATH`)")
    lines.append("")
    for sql in table_sql:
        table_name, columns, constraints = parse_create_table_sql(sql)
        lines.append(f"### `{table_name}`")
        lines.append("")
        lines.append("| Column | Definition |")
        lines.append("| --- | --- |")
        for col_name, definition in columns:
            lines.append(f"| `{col_name}` | `{definition}` |")
        if constraints:
            lines.append("")
            lines.append("Constraints:")
            for constraint in constraints:
                lines.append(f"- `{constraint}`")
        lines.append("")

    lines.append("### Indexes")
    lines.append("")
    for statement in index_sql:
        lines.append(f"- `{statement}`")
    lines.append("")

    lines.append("## Pronunciation Rules JSON (`PRONUNCIATION_RULES_PATH`)")
    lines.append("")
    lines.append("- Root object: language code -> dictionary")
    lines.append(
        "- Language keys: normalized single-letter Kokoro language code (`a,b,e,f,h,i,j,p,z`)"
    )
    lines.append("- Entry value: word -> phoneme string")
    lines.append("- Invalid language buckets are skipped at load-time")
    lines.append("")

    (MAP_DIR / "schema.md").write_text("\n".join(lines), encoding="utf-8")


def parse_env_example() -> dict[str, str]:
    env_file = PROJECT_ROOT / ".env.example"
    if not env_file.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            values[key] = value
    return values


def collect_env_references() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    defaults: dict[str, set[str]] = defaultdict(set)
    references: dict[str, set[str]] = defaultdict(set)

    for path in sorted(PROJECT_ROOT.rglob("*")):
        if should_skip(path) or not path.is_file():
            continue
        rel = rel_posix(path)
        if path.suffix.lower() not in {".py", ".ps1", ".bat", ".md"}:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue

        for match in ENV_GET_RE.finditer(text):
            name = match.group(1)
            raw_default = (match.group(2) or "").strip()
            default = raw_default if raw_default else "<unset>"
            defaults[name].add(default)
            references[name].add(rel)

        for match in PS_ENV_RE.finditer(text):
            name = match.group(1)
            defaults[name].add("<process-env>")
            references[name].add(rel)

    return defaults, references


def write_env_artifact() -> None:
    defaults, references = collect_env_references()
    env_example = parse_env_example()

    all_keys = sorted(set(defaults) | set(env_example))
    lines = ["# Environment Variable Catalog", ""]
    lines.append("| Variable | Code Defaults | `.env.example` | Referenced In |")
    lines.append("| --- | --- | --- | --- |")
    for key in all_keys:
        code_defaults = ", ".join(sorted(defaults.get(key, {"<none>"})))
        env_value = env_example.get(key, "<not documented>")
        refs = ", ".join(sorted(references.get(key, {"<not found>"})))
        lines.append(f"| `{key}` | `{code_defaults}` | `{env_value}` | `{refs}` |")
    lines.append("")

    (MAP_DIR / "env.md").write_text("\n".join(lines), encoding="utf-8")


def run_morph_benchmark() -> tuple[int, int, float]:
    from kokoro_tts.storage.morphology_repository import MorphologyRepository

    part_count = 5
    segments_per_part = 120
    tokens_per_segment = 200
    sample_text = " ".join("Token" for _ in range(tokens_per_segment))
    parts = [
        [("af_heart", sample_text) for _ in range(segments_per_part)] for _ in range(part_count)
    ]

    def analyzer(text: str) -> dict[str, object]:
        items: list[dict[str, object]] = []
        offset = 0
        for token in text.split():
            start = text.find(token, offset)
            end = start + len(token)
            offset = end
            items.append(
                {
                    "token": token,
                    "lemma": token.lower(),
                    "upos": "NOUN",
                    "feats": {},
                    "start": start,
                    "end": end,
                    "key": f"{token.lower()}|noun",
                }
            )
        return {"items": items}

    repo = MorphologyRepository(
        enabled=False,
        db_path=":memory:",
        analyzer=analyzer,
        expression_extractor=lambda _text: [],
    )
    started = time.perf_counter()
    tokens, expressions = repo._collect_ingest_rows(parts, source="benchmark")
    elapsed = time.perf_counter() - started
    return len(tokens), len(expressions), elapsed


def write_performance_artifact() -> None:
    python_files = [path for path in PROJECT_ROOT.rglob("*.py") if not should_skip(path)]
    loc_items: list[tuple[str, int]] = []
    for path in python_files:
        try:
            line_count = len(path.read_text(encoding="utf-8").splitlines())
        except OSError:
            continue
        loc_items.append((rel_posix(path), line_count))
    largest_files = sorted(loc_items, key=lambda item: item[1], reverse=True)[:10]

    token_count, expression_count, elapsed = run_morph_benchmark()

    lines = ["# Performance Summary", ""]
    lines.append("## Static Footprint")
    lines.append("")
    lines.append(f"- Python files scanned: `{len(loc_items)}`")
    lines.append(f"- Total Python LOC: `{sum(count for _, count in loc_items)}`")
    lines.append("- Largest files by LOC:")
    for path, count in largest_files:
        lines.append(f"  - `{path}`: `{count}`")
    lines.append("")

    lines.append("## Synthetic Morphology Ingest Benchmark")
    lines.append("")
    lines.append("- Input: `5 parts x 120 segments x 200 tokens`")
    lines.append(f"- Token rows produced: `{token_count}`")
    lines.append(f"- Expression rows produced: `{expression_count}`")
    lines.append(f"- Elapsed seconds: `{elapsed:.6f}`")
    if elapsed > 0:
        lines.append(f"- Approx token rows/sec: `{token_count / elapsed:,.0f}`")
    lines.append("")

    lines.append("## Operational Notes")
    lines.append("")
    lines.append("- Main runtime hotspots are the UI layer and NLP-heavy domain modules.")
    lines.append(
        "- Use `scripts/profile_tts_inference.py` for device-level first/warm generation latency."
    )
    lines.append(
        "- Use `scripts/benchmark_morph_ingest.py` for regression checks on ingest throughput."
    )
    lines.append("")

    (MAP_DIR / "performance.md").write_text("\n".join(lines), encoding="utf-8")


def testing_map_section() -> str:
    test_roots = sorted(
        [
            path
            for path in (PROJECT_ROOT / "tests").iterdir()
            if path.is_dir() and not path.name.startswith("__")
        ],
        key=lambda item: item.name,
    )
    lines = ["- Test roots:"]
    for path in test_roots:
        lines.append(f"- `tests/{path.name}`")
    lines.append("- Commands:")
    lines.append("- `python -m pytest -q`")
    lines.append("- `python -m pytest --cov=kokoro_tts --cov=app --cov-report=term-missing -q`")
    lines.append("- `ruff check .`")
    lines.append("- `mypy`")
    return "\n".join(lines)


def build_project_map_markdown(
    *,
    module_edges: set[tuple[str, str]],
    layer_edges: set[tuple[str, str]],
) -> str:
    top_level_dirs = [
        path
        for path in sorted(PROJECT_ROOT.iterdir(), key=lambda item: item.name.lower())
        if path.is_dir() and not should_skip(path)
    ]
    top_level_files = [
        path
        for path in sorted(PROJECT_ROOT.iterdir(), key=lambda item: item.name.lower())
        if path.is_file()
    ]
    layer_lines: list[str] = []
    for src, dst in sorted(layer_edges):
        layer_lines.append(f"  {identifier(src)} --> {identifier(dst)}")
    if not layer_lines:
        layer_lines.append("  app --> application")

    directory_lines = []
    for directory in top_level_dirs:
        name = directory.name
        if name == "kokoro_tts":
            purpose = "Main package with layered runtime modules."
        elif name == "tests":
            purpose = "Unit/integration/UI and contract tests."
        elif name == "scripts":
            purpose = "Operational and diagnostic scripts."
        elif name == "data":
            purpose = "Runtime data files (pronunciation rules, local NLP assets)."
        elif name == "docs":
            purpose = "Project architecture and generated mapping artifacts."
        elif name == ".github":
            purpose = "CI workflow configuration."
        else:
            purpose = "Project directory."
        directory_lines.append(f"- `{name}`: {purpose}")

    for file_path in top_level_files:
        if file_path.name in {"app.py", "README.md", "requirements.txt", "requirements-dev.txt"}:
            directory_lines.append(
                f"- `{file_path.name}`: top-level runtime or project metadata file."
            )

    module_edge_count = len(module_edges)
    layer_edge_count = len(layer_edges)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    testing_section = testing_map_section()

    lines: list[str] = []
    lines.append("# PROJECT_MAP")
    lines.append("")
    lines.append(f"Generated at: `{generated_at}`")
    lines.append("")
    lines.append(
        "This document is generated and maintained from the codebase. "
        "Detailed machine-generated artifacts live in `docs/project-map/`."
    )
    lines.append("")
    lines.append("## 1) High-Level Architecture")
    lines.append("")
    lines.append("```mermaid")
    lines.append("graph TD")
    lines.append('  app["app.py facade"]')
    lines.append('  application["application layer"]')
    lines.append('  domain["domain services"]')
    lines.append('  storage["storage adapters"]')
    lines.append('  integrations["integrations adapters"]')
    lines.append('  ui["desktop ui"]')
    lines.append('  core["core/config"]')
    lines.extend(layer_lines)
    lines.append("```")
    lines.append("")
    lines.append("Runtime model:")
    lines.append("- `app.py` is the compatibility facade and process bootstrap.")
    lines.append("- `kokoro_tts.application` orchestrates state, ports, and service wiring.")
    lines.append(
        "- `kokoro_tts.domain` contains normalization, splitting, style, voice, morphology, and expression logic."
    )
    lines.append("- `kokoro_tts.storage` handles local files plus SQLite persistence/export.")
    lines.append(
        "- `kokoro_tts.integrations` wraps model loading, ffmpeg setup, and GPU forwarding."
    )
    lines.append("- `kokoro_tts.ui` hosts the Tkinter desktop interface and feature modules.")
    lines.append("")
    lines.append("## 2) Directory Map")
    lines.append("")
    lines.extend(directory_lines)
    lines.append("")
    lines.append("Detailed tree: `docs/project-map/structure.txt`")
    lines.append("")
    lines.append("## 3) Entry Points & Lifecycle")
    lines.append("")
    lines.append("- Primary entrypoint: `app.py` (`launch()` for desktop runtime).")
    lines.append("- Module launcher: `kokoro_tts/main.py` calls `app.launch()`.")
    lines.append("- Startup flow:")
    lines.append("1. `load_config()` -> build `AppConfig`.")
    lines.append("2. `setup_logging()` + ffmpeg discovery.")
    lines.append("3. Build `AppContext`.")
    lines.append(
        "4. `initialize_app_services()` wires model manager, repositories, app state, history service, and Tkinter UI."
    )
    lines.append(
        "5. App facade functions (`generate_first`, `generate_all`, `predict`, `tokenize_first`) delegate through `LocalKokoroApi`."
    )
    lines.append("- Background work:")
    lines.append("- Optional morphology async ingest uses a single-worker `ThreadPoolExecutor`.")
    lines.append("- Optional TTS prewarm can run sync or async thread (`tts-prewarm`).")
    lines.append("")
    lines.append("## 4) Dependency Graph Overview")
    lines.append("")
    lines.append(f"- Internal module edges discovered: `{module_edge_count}`")
    lines.append(f"- Layer edges discovered: `{layer_edge_count}`")
    lines.append("- Artifacts:")
    lines.append("- `docs/project-map/deps-graph.mmd`")
    lines.append("- `docs/project-map/deps-graph.tsv`")
    lines.append(
        "- Dependency direction is primarily inward from `app/ui` -> `application` -> `domain/storage/integrations`."
    )
    lines.append("")
    lines.append("## 5) Configuration & Environment")
    lines.append("")
    lines.append(
        "- Environment variables are loaded via `os.getenv` and script process environment injection."
    )
    lines.append("- Primary config aggregation: `kokoro_tts/config.py::load_config`.")
    lines.append("- Secret handling:")
    lines.append("- Optional `HF_TOKEN` read from environment only (not committed).")
    lines.append("- `.env` is ignored by git; `.env.example` documents non-secret defaults.")
    lines.append("- Full catalog with defaults and references: `docs/project-map/env.md`.")
    lines.append("")
    lines.append("## 6) Data Model")
    lines.append("")
    lines.append("- SQLite morphology database (optional) at `MORPH_DB_PATH`.")
    lines.append("- Core tables (prefix configurable):")
    lines.append("- `*_lexemes`")
    lines.append("- `*_token_occurrences`")
    lines.append("- `*_expressions`")
    lines.append("- Pronunciation dictionary stored as JSON map at `PRONUNCIATION_RULES_PATH`.")
    lines.append("- Schema details: `docs/project-map/schema.md`.")
    lines.append("")
    lines.append("## 7) Critical Business Flows")
    lines.append("")
    lines.append("- Text generation flow:")
    lines.append("1. UI/app wrapper receives text + generation settings.")
    lines.append("2. `LocalKokoroApi` resolves voice and delegates to `KokoroState`.")
    lines.append(
        "3. `KokoroState` preprocesses text, performs model inference (CPU/GPU), post-processes audio, writes output."
    )
    lines.append("- Pronunciation dictionary flow:")
    lines.append("1. Repository loads/parses rules JSON.")
    lines.append("2. `ModelManager` applies rules to language pipeline lexicons.")
    lines.append("3. UI and app wrappers allow load/apply/import/export at runtime.")
    lines.append("- Morphology flow (when enabled):")
    lines.append("1. Dialogue segments are tokenized and optionally expression-analyzed.")
    lines.append("2. Rows are inserted into SQLite with dedup/ignore semantics.")
    lines.append("3. UI exposes read-only browsing and export.")
    lines.append("")
    lines.append("## 8) Testing Map")
    lines.append("")
    lines.extend(testing_section.splitlines())
    lines.append("")
    lines.append("Supplemental artifacts:")
    lines.append("- Route map: `docs/project-map/routes.md`")
    lines.append("- Performance summary: `docs/project-map/performance.md`")
    lines.append("")
    return "\n".join(lines)


def write_project_map(
    *, module_edges: set[tuple[str, str]], layer_edges: set[tuple[str, str]]
) -> None:
    content = build_project_map_markdown(module_edges=module_edges, layer_edges=layer_edges)
    (DOCS_DIR / "PROJECT_MAP.md").write_text(content, encoding="utf-8")


def run() -> None:
    ensure_docs_dirs()

    structure_lines = build_structure_tree()
    (MAP_DIR / "structure.txt").write_text("\n".join(structure_lines) + "\n", encoding="utf-8")

    modules = collect_python_modules()
    module_edges, layer_edges = build_dependency_edges(modules)
    write_dependency_artifacts(module_edges=module_edges, layer_edges=layer_edges)
    write_routes_artifact()
    write_schema_artifact()
    write_env_artifact()
    write_performance_artifact()
    write_project_map(module_edges=module_edges, layer_edges=layer_edges)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate architecture/project map documentation artifacts."
    )
    _ = parser.parse_args()
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
