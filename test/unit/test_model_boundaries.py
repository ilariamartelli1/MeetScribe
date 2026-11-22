"""Verify core processing does not depend on provider adapter modules."""

import ast
from pathlib import Path


def test_core_does_not_import_provider_adapters() -> None:
    core = Path(__file__).resolve().parents[2] / "app" / "core"
    imports: list[tuple[Path, str]] = []
    for path in core.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend((path, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append((path, module))
                if module == "app":
                    imports.extend((path, f"app.{alias.name}") for alias in node.names)
    violations = [
        f"{path.name}: {module}"
        for path, module in imports
        if module == "app.adapters" or module.startswith("app.adapters.")
    ]
    assert not violations, violations
