#!/usr/bin/env python3
"""Guard leve para garantir que os caminhos canônicos sigam como referência oficial.

Valida que README, documentos operacionais e wrappers históricos promovem
os caminhos canônicos como padrão oficial, mantendo os caminhos antigos
apenas como compatibilidade temporária.

Uso:
    python scripts/quality/canonical_paths_reference_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality import compatibility_contract_guard
from scripts.reports import release_health_report

CANONICAL_DOCS = [
    "README.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
]

CANONICAL_COMMANDS = [
    "python scripts/quality/check_base.py",
    "python scripts/validation/smoke_test_base.py",
    "python scripts/quality/release_metadata_guard.py",
    "python scripts/quality/compatibility_contract_guard.py",
    "python scripts/quality/operational_checks_contract_guard.py",
    "python scripts/quality/canonical_paths_reference_guard.py",
    "python scripts/quality/script_cli_contract_guard.py",
    "python scripts/quality/release_artifacts_hygiene_guard.py",
    "python scripts/quality/runtime_dependencies_contract_guard.py",
    "python scripts/quality/documentation_commands_examples_guard.py",
    "python scripts/quality/release_manifest_guard.py",
    "python scripts/quality/quality_runtime_budget_guard.py",
    "python scripts/quality/release_guard.py",
    "python scripts/quality/quality_gate.py",
    "python scripts/reports/release_health_report.py",
    "python scripts/reports/maintenance_snapshot_report.py",
    "python scripts/reports/maintenance_handoff_pack.py",
]

HISTORICAL_COMMANDS = [
    "python scripts/check_base.py",
    "python scripts/smoke_test_base.py",
    "python scripts/release_metadata_guard.py",
    "python scripts/compatibility_contract_guard.py",
    "python scripts/operational_checks_contract_guard.py",
    "python scripts/canonical_paths_reference_guard.py",
    "python scripts/script_cli_contract_guard.py",
    "python scripts/release_artifacts_hygiene_guard.py",
    "python scripts/runtime_dependencies_contract_guard.py",
    "python scripts/documentation_commands_examples_guard.py",
    "python scripts/release_manifest_guard.py",
    "python scripts/quality_runtime_budget_guard.py",
    "python scripts/release_guard.py",
    "python scripts/quality_gate.py",
    "python scripts/release_health_report.py",
]

WRAPPER_MARKERS = [
    "Wrapper temporário de compatibilidade",
    "Padrão oficial atual:",
]

DOC_BRIDGE_MARKERS = [
    "ponte temporária de compatibilidade",
    "padrão oficial atual",
]

EXPECTED_REPORT_PATHS = {
    "scripts/quality/canonical_paths_reference_guard.py",
    "scripts/quality/script_cli_contract_guard.py",
    "scripts/quality/release_artifacts_hygiene_guard.py",
    "scripts/quality/runtime_dependencies_contract_guard.py",
    "scripts/quality/documentation_commands_examples_guard.py",
    "scripts/quality/release_manifest_guard.py",
    "scripts/quality/quality_runtime_budget_guard.py",
    "scripts/quality/governance_docs_crosslinks_guard.py",
    "scripts/reports/release_health_report.py",
    "scripts/reports/maintenance_snapshot_report.py",
    "scripts/reports/maintenance_handoff_pack.py",
    "scripts/quality/checks_registry.py",
    "scripts/quality/checks_registry_contract_guard.py",
    "scripts/quality/checks_registry_schema_guard.py",
    "scripts/quality/quality_gate_composition_guard.py",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
}


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []

    for rel_path in CANONICAL_DOCS:
        text = read_text(rel_path)
        if "caminhos canônicos" not in text.lower():
            errors.append(f"{rel_path} deve reforçar explicitamente os caminhos canônicos como padrão oficial")
        else:
            notes.append(f"OK documento reforça caminhos canônicos: {rel_path}")

        if "compatibilidade temporária" not in text.lower():
            errors.append(f"{rel_path} deve explicitar que caminhos históricos existem apenas como compatibilidade temporária")
        else:
            notes.append(f"OK documento explicita compatibilidade temporária: {rel_path}")

        for command in CANONICAL_COMMANDS:
            if command in text:
                notes.append(f"OK comando canônico documentado em {rel_path}: {command}")

        historical_hits = [command for command in HISTORICAL_COMMANDS if command in text]
        if historical_hits:
            errors.append(
                f"{rel_path} não deve promover comandos históricos como referência principal: " + ", ".join(historical_hits)
            )

    for rel_path, expected in compatibility_contract_guard.WRAPPER_EXPECTATIONS.items():
        text = read_text(rel_path)
        for marker in WRAPPER_MARKERS:
            if marker not in text:
                errors.append(f"{rel_path} deve conter o marcador `{marker}`")
        if expected["canonical_path"] not in text:
            errors.append(f"{rel_path} deve apontar para o caminho canônico `{expected['canonical_path']}`")
        else:
            notes.append(f"OK wrapper aponta para o caminho canônico: {rel_path} -> {expected['canonical_path']}")

    for rel_path, canonical_path in compatibility_contract_guard.DOC_BRIDGE_EXPECTATIONS.items():
        text = read_text(rel_path)
        lowered = text.lower()
        for marker in DOC_BRIDGE_MARKERS:
            if marker not in lowered:
                errors.append(f"{rel_path} deve conter o marcador de ponte `{marker}`")
        if canonical_path not in text:
            errors.append(f"{rel_path} deve apontar para `{canonical_path}`")
        else:
            notes.append(f"OK arquivo-ponte aponta para o documento canônico: {rel_path} -> {canonical_path}")

    report_paths = set(release_health_report.CANONICAL_PATHS)
    missing = sorted(EXPECTED_REPORT_PATHS - report_paths)
    if missing:
        errors.append(
            "scripts/reports/release_health_report.py deve listar os caminhos canônicos esperados: " + ", ".join(missing)
        )
    else:
        notes.append("OK release_health_report lista os caminhos canônicos principais esperados")

    print("=== CANONICAL PATHS REFERENCE GUARD | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")

    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nReferência canônica oficial íntegra.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
