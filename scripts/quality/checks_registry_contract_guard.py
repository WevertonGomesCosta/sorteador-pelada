#!/usr/bin/env python3
"""Guard leve do registro canônico dos checks operacionais.

Valida que:
- ``scripts/quality/checks_registry.py`` seja a fonte única de verdade dos checks canônicos;
- ``quality_gate.py`` e ``release_health_report.py`` consumam esse registro;
- a lista oficial não tenha nomes ou comandos duplicados;
- a documentação operacional oficial cite o novo contrato.

Uso:
    python scripts/quality/checks_registry_contract_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality import checks_registry, quality_gate
from scripts.reports import release_health_report

DOCS_TO_VALIDATE = [
    "README.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
]
DOC_MARKERS = [
    "python scripts/quality/checks_registry_contract_guard.py",
    "python scripts/quality/checks_registry_schema_guard.py",
    "python scripts/quality/checks_registry_consumers_guard.py",
    "scripts/quality/checks_registry.py",
    "fonte única de verdade",
    "Schema canônico",
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


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(__doc__.strip())
        return 0

    errors: list[str] = []
    notes: list[str] = []

    registry_checks = checks_registry.expected_official_checks(target="quality_gate")
    report_registry_checks = checks_registry.expected_official_checks(target="release_health_report")
    quality_checks = normalize_checks(quality_gate.CHECKS)
    report_checks = normalize_checks(release_health_report.CHECKS)

    names = [name for name, _ in registry_checks]
    commands = [command for _, command in registry_checks]
    if len(names) != len(set(names)):
        errors.append("scripts/quality/checks_registry.py não deve conter nomes de checks duplicados")
    else:
        notes.append("OK registro canônico sem nomes duplicados")
    if len(commands) != len(set(commands)):
        errors.append("scripts/quality/checks_registry.py não deve conter comandos duplicados")
    else:
        notes.append("OK registro canônico sem comandos duplicados")

    if quality_checks != registry_checks:
        errors.append("scripts/quality/quality_gate.py deve consumir exatamente o registro canônico de checks")
    else:
        notes.append("OK quality_gate consome o registro canônico")

    if report_checks != report_registry_checks:
        errors.append("scripts/reports/release_health_report.py deve consumir exatamente o registro canônico de checks")
    else:
        notes.append("OK release_health_report consome o registro canônico")

    quality_text = read_text("scripts/quality/quality_gate.py")
    if "from scripts.quality.checks_registry import" not in quality_text:
        errors.append("scripts/quality/quality_gate.py deve importar o registro canônico de checks")
    else:
        notes.append("OK quality_gate importa checks_registry")

    report_text = read_text("scripts/reports/release_health_report.py")
    if "from scripts.quality.checks_registry import" not in report_text:
        errors.append("scripts/reports/release_health_report.py deve importar o registro canônico de checks")
    else:
        notes.append("OK release_health_report importa checks_registry")

    for rel_path in DOCS_TO_VALIDATE:
        text = read_text(rel_path)
        for marker in DOC_MARKERS:
            if marker not in text:
                errors.append(f"{rel_path} deve citar `{marker}`")
            else:
                notes.append(f"OK marcador documental presente: {rel_path} -> {marker}")

    print("=== CHECKS REGISTRY CONTRACT GUARD | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")

    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nRegistro canônico dos checks íntegro.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
