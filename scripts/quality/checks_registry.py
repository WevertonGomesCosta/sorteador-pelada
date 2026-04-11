#!/usr/bin/env python3
"""Registro canônico da rotina oficial de checks da base.

Objetivo:
- definir uma fonte única de verdade para a lista oficial de checks canônicos;
- evitar duplicação entre ``quality_gate.py`` e ``release_health_report.py``;
- centralizar nomes, categorias, caminhos, timeouts e flags da rotina operacional.

Uso:
    from scripts.quality.checks_registry import (
        CHECKS_REGISTRY,
        DEFAULT_CHECK_TIMEOUT_SECONDS,
        CHECK_TIMEOUT_OVERRIDES,
        build_check_commands,
        expected_official_checks,
        registry_entries,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

DEFAULT_CHECK_TIMEOUT_SECONDS: Final[int] = 120
TARGETS: Final[tuple[str, str]] = ("quality_gate", "release_health_report")

CHECKS_REGISTRY: Final[list[dict[str, object]]] = [
    {"id": "check_base", "name": "check_base", "category": "structural", "kind": "script", "rel_path": "scripts/quality/check_base.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "smoke_test_base", "name": "smoke_test_base", "category": "validation", "kind": "script", "rel_path": "scripts/validation/smoke_test_base.py", "timeout_seconds": 180, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "compileall", "name": "compileall", "category": "structural", "kind": "module", "args": ["-m", "compileall", "."], "timeout_seconds": 180, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "release_metadata_guard", "name": "release_metadata_guard", "category": "release", "kind": "script", "rel_path": "scripts/quality/release_metadata_guard.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "compatibility_contract_guard", "name": "compatibility_contract_guard", "category": "compatibility", "kind": "script", "rel_path": "scripts/quality/compatibility_contract_guard.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "operational_checks_contract_guard", "name": "operational_checks_contract_guard", "category": "operations", "kind": "script", "rel_path": "scripts/quality/operational_checks_contract_guard.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "canonical_paths_reference_guard", "name": "canonical_paths_reference_guard", "category": "documentation", "kind": "script", "rel_path": "scripts/quality/canonical_paths_reference_guard.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "script_cli_contract_guard", "name": "script_cli_contract_guard", "category": "operations", "kind": "script", "rel_path": "scripts/quality/script_cli_contract_guard.py", "timeout_seconds": 180, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "release_artifacts_hygiene_guard", "name": "release_artifacts_hygiene_guard", "category": "hygiene", "kind": "script", "rel_path": "scripts/quality/release_artifacts_hygiene_guard.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "runtime_dependencies_contract_guard", "name": "runtime_dependencies_contract_guard", "category": "runtime", "kind": "script", "rel_path": "scripts/quality/runtime_dependencies_contract_guard.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "documentation_commands_examples_guard", "name": "documentation_commands_examples_guard", "category": "documentation", "kind": "script", "rel_path": "scripts/quality/documentation_commands_examples_guard.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "release_manifest_guard", "name": "release_manifest_guard", "category": "release", "kind": "script", "rel_path": "scripts/quality/release_manifest_guard.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "quality_runtime_budget_guard", "name": "quality_runtime_budget_guard", "category": "operations", "kind": "script", "rel_path": "scripts/quality/quality_runtime_budget_guard.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "script_exit_codes_contract_guard", "name": "script_exit_codes_contract_guard", "category": "operations", "kind": "script", "rel_path": "scripts/quality/script_exit_codes_contract_guard.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "governance_docs_crosslinks_guard", "name": "governance_docs_crosslinks_guard", "category": "documentation", "kind": "script", "rel_path": "scripts/quality/governance_docs_crosslinks_guard.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "protected_scope_hash_guard", "name": "protected_scope_hash_guard", "category": "release", "kind": "script", "rel_path": "scripts/quality/protected_scope_hash_guard.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "checks_registry_contract_guard", "name": "checks_registry_contract_guard", "category": "governance", "kind": "script", "rel_path": "scripts/quality/checks_registry_contract_guard.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "checks_registry_schema_guard", "name": "checks_registry_schema_guard", "category": "governance", "kind": "script", "rel_path": "scripts/quality/checks_registry_schema_guard.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "checks_registry_consumers_guard", "name": "checks_registry_consumers_guard", "category": "governance", "kind": "script", "rel_path": "scripts/quality/checks_registry_consumers_guard.py", "timeout_seconds": 120, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
    {"id": "release_guard", "name": "release_guard", "category": "release", "kind": "script", "rel_path": "scripts/quality/release_guard.py", "timeout_seconds": 180, "enabled_in_quality_gate": True, "enabled_in_release_health_report": True},
]

FORCED_TIMEOUT_OVERRIDE_NAMES: Final[set[str]] = {
    "quality_runtime_budget_guard",
    "script_exit_codes_contract_guard",
    "protected_scope_hash_guard",
    "checks_registry_schema_guard",
}

CHECK_TIMEOUT_OVERRIDES: Final[dict[str, int]] = {
    str(spec["name"]): int(spec["timeout_seconds"])
    for spec in CHECKS_REGISTRY
    if int(spec["timeout_seconds"]) != DEFAULT_CHECK_TIMEOUT_SECONDS or str(spec["name"]) in FORCED_TIMEOUT_OVERRIDE_NAMES
}


def registry_entries(*, target: str | None = None) -> list[dict[str, object]]:
    if target is not None and target not in TARGETS:
        raise ValueError(f"target inválido: {target}")
    if target is None:
        return [dict(spec) for spec in CHECKS_REGISTRY]
    flag_name = f"enabled_in_{target}"
    return [dict(spec) for spec in CHECKS_REGISTRY if bool(spec.get(flag_name, False))]


def build_check_commands(root: Path, python_executable: str, *, target: str = "quality_gate") -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    for spec in registry_entries(target=target):
        name = str(spec["name"])
        kind = str(spec["kind"])
        if kind == "module":
            args = [str(x) for x in spec["args"]]
            command = [python_executable, *args]
        else:
            rel_path = str(spec["rel_path"])
            command = [python_executable, str(root / rel_path)]
        commands.append((name, command))
    return commands


def expected_official_checks(*, target: str = "quality_gate") -> list[tuple[str, str]]:
    expected: list[tuple[str, str]] = []
    for spec in registry_entries(target=target):
        name = str(spec["name"])
        kind = str(spec["kind"])
        if kind == "module":
            args = " ".join(str(x) for x in spec["args"])
            expected.append((name, f"python {args}"))
        else:
            rel_path = str(spec["rel_path"])
            expected.append((name, f"python {rel_path}"))
    return expected


def canonical_check_paths() -> list[str]:
    return [str(spec["rel_path"]) for spec in CHECKS_REGISTRY if isinstance(spec.get("rel_path"), str)]
