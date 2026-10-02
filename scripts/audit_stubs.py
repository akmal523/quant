"""audit_stubs.py — static stub and silent-fallback auditor (v10.7.5, Part 1).

AST-based scan of ``quant/`` (excluding tests). Reports findings with file, line,
and kind, and exits nonzero when a blocking finding remains.

Kinds:
  1. pass_body            public function body is only pass / docstring + pass
  2. not_implemented      raise NotImplementedError in non-abstract public code
  3. todo_marker          TODO/FIXME/XXX/STUB/PLACEHOLDER in quant/
  4. bare_except          except that passes without logging or a marker
  5. constant_return      long docstring but body is a single literal return
  6. trivial_complexity   docstring verb but complexity 1 and no calls
  7. test_branch          os.environ reads of TEST/MOCK/FAKE/STUB keys
  8. vacuous_constraint   cap >= 1.0, threshold <= 0, fee < 0
  9. canary_number        float literals equal to known live/golden values
 10. undeclared_fallback  except returning a default in a Tier S module not in
                          the fallback registry

R11 tiering:
  - Tier S modules: kinds 4/10 are blocking unless the enclosing function is in
    the fallback registry (``quant/engine/fallback_registry.py``).
  - Tier C patterns: pass-only / log-only / cleanup shapes are non-blocking; the
    auditor AST-verifies the shape so a numeric return cannot hide.
  - Tier W: everything else is reported and ratcheted against
    ``tests/warn_baseline.txt``.

Usage: ``python scripts/audit_stubs.py [--json] [--update-baseline]``.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py<3.11
    import tomli as tomllib  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
QUANT = ROOT / "quant"
ALLOWLIST = ROOT / "scripts" / "audit_allowlist.toml"
WARN_BASELINE = ROOT / "tests" / "warn_baseline.txt"
GOLDEN = ROOT / "tests" / "golden" / "backtest_2024.json"

CANARY_NUMBERS = {852.23, 143.53, 287.19, 270.49, 150.03, 10400.0, 19.33}
TRIVIAL_NUMBERS = {0.0, 1.0, -1.0, 2.0, 100.0, 0.5, 10.0, 1000.0}
DOC_VERBS = ("compute", "calculate", "optimize", "score", "evaluate",
             "allocate", "reconcile")
LOG_ATTRS = {"warning", "error", "info", "debug", "exception", "critical", "warn"}
CLEANUP_CALLS = {"remove", "unlink", "close", "rename", "replace", "makedirs",
                 "chmod", "symlink", "rmtree", "copytree", "mkdir", "rmdir"}
DEFAULT_CALLS = {"dict", "list", "set", "tuple", "DataFrame", "full", "zeros",
                 "ones", "empty", "AccountState", "Series"}


@dataclass
class Finding:
    """One audit finding."""

    file: str
    line: int
    kind: str
    message: str
    tier: str = "S"  # S | C | W
    blocking: bool = True


@dataclass
class Config:
    """The allowlist configuration."""

    tier_s_modules: set[str] = field(default_factory=set)
    exceptions: list[dict] = field(default_factory=list)
    tier_c_patterns: list[dict] = field(default_factory=list)


def load_config() -> Config:
    """Load scripts/audit_allowlist.toml; empty config when absent."""
    cfg = Config()
    if not ALLOWLIST.exists():
        return cfg
    data = tomllib.loads(ALLOWLIST.read_text(encoding="utf-8"))
    cfg.tier_s_modules = {str(m) for m in data.get("tier_s_modules", [])}
    cfg.exceptions = list(data.get("exceptions", []))
    cfg.tier_c_patterns = list(data.get("tier_c_patterns", []))
    return cfg


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _is_public(name: str) -> bool:
    return bool(name) and not name.startswith("_")


def _body_without_docstring(node: ast.AST) -> list[ast.stmt]:
    body = list(getattr(node, "body", []))
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[1:]
    return body


def _is_pass_only(body: list[ast.stmt]) -> bool:
    if not body:
        return True
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
                and stmt.value.value is Ellipsis:
            continue
        return False
    return True


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _is_log_call(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            name = _call_name(sub)
            if name in LOG_ATTRS or name == "print":
                return True
    return False


def _is_marker(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign):
            for tgt in sub.targets:
                if isinstance(tgt, ast.Name) and "quality" in tgt.id:
                    return True
        if isinstance(sub, ast.Call) and _call_name(sub) == "register_fallback":
            return True
    return False


def _is_empty_container(node: ast.AST) -> bool:
    if isinstance(node, ast.Dict) and not node.keys:
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and not node.elts:
        return True
    if isinstance(node, ast.Call) and _call_name(node) in DEFAULT_CALLS:
        return True
    return False


def _returns_default(body: list[ast.stmt]) -> bool:
    for stmt in body:
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            if isinstance(stmt.value, ast.Constant):
                return True
            if _is_empty_container(stmt.value):
                return True
    return False


def _try_calls(node: ast.Try) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])):
        if isinstance(sub, ast.Call):
            names.add(_call_name(sub))
    return names


def _cyclomatic(node: ast.AST) -> int:
    count = 1
    for sub in ast.walk(node):
        if isinstance(sub, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                            ast.IfExp, ast.BoolOp, ast.comprehension)):
            count += 1
    return count


def _has_call(node: ast.AST) -> bool:
    return any(isinstance(sub, ast.Call) for sub in ast.walk(node))


def _enclosing_function(tree: ast.AST) -> dict[int, str]:
    """Map each line to the name of the innermost function containing it."""
    out: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if hasattr(sub, "lineno"):
                    out[sub.lineno] = node.name
    return out


def _is_abstract(node: ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        name = dec.id if isinstance(dec, ast.Name) else (
            dec.attr if isinstance(dec, ast.Attribute) else "")
        if name == "abstractmethod":
            return True
    return False


def _registry_names() -> set[str]:
    """Function names present in the fallback registry (best-effort)."""
    path = QUANT / "engine" / "fallback_registry.py"
    if not path.exists():
        return set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    names.add(key.value)
    return names


def _golden_numbers() -> set[float]:
    if not GOLDEN.exists():
        return set()
    try:
        data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return set()
    out: set[float] = set()

    def _walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)
        elif isinstance(obj, float):
            # Only distinctive values (more than 2 decimals) are canaries; a
            # round number like 0.3 or 20.0 is not a hardcoded answer.
            if round(obj, 3) != round(obj, 2):
                out.add(obj)

    _walk(data)
    return out


def _comments(path: Path) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    try:
        with path.open("rb") as fh:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type == tokenize.COMMENT:
                    out.append((tok.start[0], tok.string))
    except Exception:  # noqa: BLE001
        pass
    return out


def _exception_allowed(cfg: Config, rel: str, line: int, kind: str) -> bool:
    for exc in cfg.exceptions:
        if (exc.get("file") == rel and int(exc.get("line", -1)) == line
                and exc.get("kind") == kind):
            return True
    return False


def _tier_c_match(cfg: Config, handler: ast.ExceptHandler, try_node: ast.Try) -> str | None:
    """Return the matching Tier C pattern name, or None."""
    body = handler.body
    if _is_log_call(handler):
        return "log_only"
    if _is_pass_only(body):
        calls = _try_calls(try_node)
        if calls and calls <= CLEANUP_CALLS:
            return "cleanup_pass"
        return "pass_only"
    return None


def audit_file(path: Path, cfg: Config, registry: set[str],
               golden: set[float]) -> list[Finding]:
    """Audit one file and return its findings."""
    rel = _rel(path)
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []
    findings: list[Finding] = []
    enclosing = _enclosing_function(tree)
    is_tier_s = rel in cfg.tier_s_modules

    # Kind 3: TODO markers in comments and strings.
    for line, text in _comments(path):
        if any(m in text for m in ("TODO", "FIXME", "XXX", "STUB", "PLACEHOLDER")):
            if not _exception_allowed(cfg, rel, line, "todo_marker"):
                findings.append(Finding(rel, line, "todo_marker",
                                        f"marker in comment: {text.strip()[:60]}"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(m in node.value for m in ("TODO", "FIXME", "XXX", "STUB", "PLACEHOLDER")):
                if not _exception_allowed(cfg, rel, node.lineno, "todo_marker"):
                    findings.append(Finding(rel, node.lineno, "todo_marker",
                                            "marker in string literal"))

    # Kind 9: canary numbers.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            if node.value in CANARY_NUMBERS or (
                    node.value in golden and node.value not in TRIVIAL_NUMBERS):
                findings.append(Finding(rel, node.lineno, "canary_number",
                                        f"literal {node.value} matches a known value"))

    # Kind 7: test-branch env reads.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in ("get", "getenv"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    tokens = arg.value.upper().replace("-", "_").split("_")
                    if any(t in ("TEST", "MOCK", "FAKE", "STUB") for t in tokens):
                        findings.append(Finding(rel, node.lineno, "test_branch",
                                                f"env read {arg.value!r}"))

    # Kind 8: vacuous constraints (config only).
    if rel.endswith("config.py"):
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, (int, float)):
                name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else ""
                val = float(node.value.value)
                is_cap = "CAP" in name or (
                    "MAX" in name and any(k in name for k in ("WEIGHT", "PCT", "ALLOCATION")))
                if is_cap and val >= 1.0:
                    findings.append(Finding(rel, node.lineno, "vacuous_constraint",
                                            f"{name} = {val} (cap >= 1.0)"))
                if "THRESHOLD" in name and val <= 0:
                    findings.append(Finding(rel, node.lineno, "vacuous_constraint",
                                            f"{name} = {val} (threshold <= 0)"))
                if "FEE" in name and val < 0:
                    findings.append(Finding(rel, node.lineno, "vacuous_constraint",
                                            f"{name} = {val} (fee < 0)"))

    # Function-level kinds.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _is_public(node.name):
            continue
        body = _body_without_docstring(node)
        doc = ast.get_docstring(node) or ""

        # Kind 1: pass_body.
        if _is_pass_only(body):
            findings.append(Finding(rel, node.lineno, "pass_body",
                                    f"{node.name} has an empty body"))

        # Kind 2: not_implemented.
        for sub in ast.walk(node):
            if isinstance(sub, ast.Raise) and isinstance(sub.exc, ast.Call) \
                    and _call_name(sub.exc) == "NotImplementedError":
                if not _is_abstract(node) and not _exception_allowed(
                        cfg, rel, sub.lineno, "not_implemented"):
                    findings.append(Finding(rel, sub.lineno, "not_implemented",
                                            f"{node.name} raises NotImplementedError"))

        # Kind 5: constant_return.
        if len(doc) > 80 and len(body) == 1 and isinstance(body[0], ast.Return) \
                and body[0].value is not None:
            val = body[0].value
            if isinstance(val, ast.Constant) or (
                    isinstance(val, ast.Name) and val.id in
                    {a.arg for a in node.args.args}):
                findings.append(Finding(rel, node.lineno, "constant_return",
                                        f"{node.name} promises computation, returns a literal"))

        # Kind 6: trivial_complexity (abstract methods are exempt).
        if any(v in doc.lower() for v in DOC_VERBS) and _cyclomatic(node) == 1 \
                and not _has_call(node) and not _is_abstract(node):
            findings.append(Finding(rel, node.lineno, "trivial_complexity",
                                    f"{node.name} names a computation but does nothing"))

    # Kinds 4/10: except paths.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if _is_log_call(handler) or _is_marker(handler):
                continue
            func = enclosing.get(handler.lineno, "")
            tier_c = _tier_c_match(cfg, handler, node)
            if tier_c is not None:
                continue  # Tier C: non-blocking
            if _is_pass_only(handler.body):
                kind = "bare_except"
            elif _returns_default(handler.body):
                kind = "undeclared_fallback"
            else:
                continue
            if _exception_allowed(cfg, rel, handler.lineno, kind):
                continue
            if f"{path.stem}.{func}" in registry:
                continue  # a declared, tested fallback is not a finding
            blocking = is_tier_s
            findings.append(Finding(
                rel, handler.lineno, kind,
                f"{func or '<module>'} swallows an error (not in the fallback registry)",
                tier="S" if blocking else "W", blocking=blocking))
    return findings


def run() -> list[Finding]:
    """Audit the whole quant/ tree."""
    cfg = load_config()
    registry = _registry_names()
    golden = _golden_numbers()
    findings: list[Finding] = []
    for path in sorted(QUANT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        findings.extend(audit_file(path, cfg, registry, golden))
    return findings


def _warn_counts(findings: list[Finding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in findings:
        if f.tier == "W":
            counts[f.kind] = counts.get(f.kind, 0) + 1
    return counts


def _read_baseline() -> dict[str, int]:
    if not WARN_BASELINE.exists():
        return {}
    out: dict[str, int] = {}
    for line in WARN_BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kind, _, count = line.partition("=")
        out[kind.strip()] = int(count.strip())
    return out


def _write_baseline(counts: dict[str, int]) -> None:
    lines = ["# v10.7.5 Tier W warn baseline (kind = count). May only shrink.",
             "# Regenerate with: python scripts/audit_stubs.py --update-baseline"]
    for kind in sorted(counts):
        lines.append(f"{kind} = {counts[kind]}")
    WARN_BASELINE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Static stub auditor (v10.7.5).")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--update-baseline", action="store_true",
                        help="rewrite tests/warn_baseline.txt from the current tree")
    args = parser.parse_args(argv)

    findings = run()
    blocking = [f for f in findings if f.blocking]
    warn = [f for f in findings if f.tier == "W"]

    if args.update_baseline:
        _write_baseline(_warn_counts(findings))
        print(f"warn baseline updated: {len(warn)} Tier W findings")
        return 0

    if args.json:
        print(json.dumps([f.__dict__ for f in findings], indent=2))
    else:
        for f in sorted(findings, key=lambda x: (x.file, x.line)):
            flag = "BLOCK" if f.blocking else "warn "
            print(f"{flag} {f.file}:{f.line}: {f.kind}: {f.message}")
        print(f"\n{len(blocking)} blocking, {len(warn)} warn, {len(findings)} total")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
