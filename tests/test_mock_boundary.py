"""test_mock_boundary.py — no test mocks the decision logic it verifies (v10.7.5).

Intent: a test that patches the very function it claims to test proves nothing.
This scanner fails when a test patches a target inside the DECISION core (the
math modules whose behavior the grid asserts). Fakes are allowed only at true I/O
boundaries (network, clock, filesystem, subprocess, the database, the UI render
layer) and at heavy external solvers.

Invariants:
  - No test patches a function in the decision core.
  - The real advice pipeline, allocator, and flows math run unmocked on
    in-memory frames.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

# The decision core: patching any of these in a test is forbidden.
DECISION_MODULES = {
    "quant.engine.advice",
    "quant.engine.allocator",
    "quant.engine.flows",
    "quant.engine.sizing",
    "quant.analytics.scoring",
    "quant.portfolio.optimizer",
    "quant.portfolio.risk",
}


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    """Map a local alias to its dotted quant module path."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("quant"):
                    aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("quant"):
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _resolve(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """Resolve a Name/Attribute/Constant node to a dotted module path."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve(node.value, aliases)
        return f"{base}.{node.attr}" if base else None
    return None


def _patched_targets(tree: ast.AST, aliases: dict[str, str]) -> list[str]:
    """Collect the dotted targets of setattr / patch / patch.object calls."""
    targets: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else "")
        if name == "setattr" and len(node.args) >= 2:
            base = _resolve(node.args[0], aliases)
            attr = node.args[1].value if isinstance(node.args[1], ast.Constant) else None
            if base and attr:
                targets.append(f"{base}.{attr}")
        elif name == "patch" and node.args:
            resolved = _resolve(node.args[0], aliases)
            if resolved:
                targets.append(resolved)
        elif name == "object" and len(node.args) >= 2:
            base = _resolve(node.args[0], aliases)
            attr = node.args[1].value if isinstance(node.args[1], ast.Constant) else None
            if base and attr:
                targets.append(f"{base}.{attr}")
    return targets


def _is_decision(target: str) -> bool:
    return any(target == mod or target.startswith(mod + ".")
               for mod in DECISION_MODULES)


def test_no_test_patches_the_decision_core():
    violations: list[str] = []
    for path in sorted(TESTS.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        aliases = _import_aliases(tree)
        for target in _patched_targets(tree, aliases):
            if _is_decision(target):
                violations.append(f"{path.name}: patches {target}")
    assert not violations, "decision-core mocks found:\n" + "\n".join(violations)


def test_real_pipeline_runs_unmocked():
    """The advice pipeline, allocator, and flows math run on real inputs."""
    from datetime import date

    from quant.engine import allocator, flows
    from quant.engine.advice import build_advice
    from tests.fixtures.live_portfolio import live_shaped, live_tiers

    holdings = live_shaped()
    advice, _rejected = build_advice(holdings, tiers=live_tiers(), as_of=date(2026, 10, 2))
    assert advice, "the real advice pipeline produced no advice"

    legs = allocator.allocate(200.0, holdings, regime="bull", candidates=[])
    assert abs(sum(leg["amount_eur"] for leg in legs) - 200.0) < 1e-6

    ret = flows.modified_dietz(1000.0, 1100.0, [], date(2026, 1, 1), date(2026, 1, 31))
    assert abs(ret - 0.10) < 1e-9
