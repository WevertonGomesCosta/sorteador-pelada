#!/usr/bin/env python3
"""Guard leve do contrato operacional dos checks canônicos da release.

Valida a sincronização entre:
- a rotina oficial executada por ``scripts/quality/quality_gate.py``;
- a evidência consolidada em ``scripts/reports/release_health_report.py``;
- a documentação operacional oficial da base.

Uso:
    python scripts/quality/operational_checks_contract_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality import quality_gate
from scripts.reports import release_health_report

EXPECTED_OFFICIAL_CHECKS: list[tuple[str, str]] = [
    ("check_base", "python scripts/quality/check_base.py"),
    ("smoke_test_base", "python scripts/validation/smoke_test_base.py"),
    ("compileall", "python -m compileall ."),
    ("release_metadata_guard", "python scripts/quality/release_metadata_guard.py"),
    ("compatibility_contract_guard", "python scripts/quality/compatibility_contract_guard.py"),
    ("operational_checks_contract_guard", "python scripts/quality/operational_checks_contract_guard.py"),
    ("canonical_paths_reference_guard", "python scripts/quality/canonical_paths_reference_guard.py"),
    ("script_cli_contract_guard", "python scripts/quality/script_cli_contract_guard.py"),
    ("release_artifacts_hygiene_guard", "python scripts/quality/release_artifacts_hygiene_guard.py"),
    ("runtime_dependencies_contract_guard", "python scripts/quality/runtime_dependencies_contract_guard.py"),
    ("documentation_commands_examples_guard", "python scripts/quality/documentation_commands_examples_guard.py"),
    ("release_manifest_guard", "python scripts/quality/release_manifest_guard.py"),
    ("quality_runtime_budget_guard", "python scripts/quality/quality_runtime_budget_guard.py"),
    ("script_exit_codes_contract_guard", "python scripts/quality/script_exit_codes_contract_guard.py"),
    ("release_guard", "python scripts/quality/release_guard.py"),
]

DOCS_TO_VALIDATE = [
    "README.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
]

AUXILIARY_MARKERS = [
    "python scripts/quality/quality_gate.py",
    "python scripts/reports/release_health_report.py",
]


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def normalize_command(command: list[str]) -> str:
    normalized: list[str] = []
    for index, token in enumerate(command):
        if index == 0:
            normalized.append("python")
            continue
        token_path = Path(token)
        if token_path.is_absolute():
            try:
                normalized.append(token_path.relative_to(ROOT).as_posix())
                continue
            except ValueError:
                pass
        normalized.append(token)
    return " ".join(normalized)


def normalize_checks(checks: list[tuple[str, list[str]]]) -> list[tuple[str, str]]:
    return [(name, normalize_command(command)) for name, command in checks]


def validate_doc_commands(rel_path: str) -> list[str]:
    text = read_text(rel_path)
    errors: list[str] = []
    for _, command in EXPECTED_OFFICIAL_CHECKS:
        if command not in text:
            errors.append(f"{rel_path} deve citar a rotina oficial `{command}`")
    for marker in AUXILIARY_MARKERS:
        if marker not in text:
            errors.append(f"{rel_path} deve citar o complemento operacional `{marker}`")
    return errors


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []

    quality_checks = normalize_checks(quality_gate.CHECKS)
    report_checks = normalize_checks(release_health_report.CHECKS)

    if quality_checks != EXPECTED_OFFICIAL_CHECKS:
        errors.append("scripts/quality/quality_gate.py deve usar exatamente a lista oficial de checks canônicos")
    else:
        notes.append("OK quality_gate usa a lista oficial de checks canônicos")

    if report_checks != EXPECTED_OFFICIAL_CHECKS:
        errors.append("scripts/reports/release_health_report.py deve reportar exatamente a lista oficial de checks canônicos")
    else:
        notes.append("OK release_health_report reporta a lista oficial de checks canônicos")

    if quality_checks != report_checks:
        errors.append("quality_gate e release_health_report devem permanecer sincronizados na lista de checks")
    else:
        notes.append("OK quality_gate e release_health_report estão sincronizados")

    report_canonical_paths = set(release_health_report.CANONICAL_PATHS)
    required_canonical_paths = {
        "scripts/quality/check_base.py",
        "scripts/validation/smoke_test_base.py",
        "scripts/quality/release_metadata_guard.py",
        "scripts/quality/compatibility_contract_guard.py",
        "scripts/quality/operational_checks_contract_guard.py",
        "scripts/quality/canonical_paths_reference_guard.py",
        "scripts/quality/script_cli_contract_guard.py",
        "scripts/quality/release_artifacts_hygiene_guard.py",
        "scripts/quality/release_manifest_guard.py",
        "scripts/quality/release_guard.py",
        "scripts/quality/quality_gate.py",
        "scripts/reports/release_health_report.py",
    }
    missing_paths = sorted(required_canonical_paths - report_canonical_paths)
    if missing_paths:
        errors.append("scripts/reports/release_health_report.py deve listar os caminhos canônicos principais: " + ", ".join(missing_paths))
    else:
        notes.append("OK release_health_report lista os caminhos canônicos principais esperados")

    for rel_path in DOCS_TO_VALIDATE:
        if not (ROOT / rel_path).exists():
            errors.append(f"Documento operacional ausente: {rel_path}")
            continue
        doc_errors = validate_doc_commands(rel_path)
        if doc_errors:
            errors.extend(doc_errors)
        else:
            notes.append(f"OK documentação operacional sincronizada: {rel_path}")

    print("=== OPERATIONAL CHECKS CONTRACT GUARD | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")

    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nContrato operacional dos checks canônicos íntegro.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
