#!/usr/bin/env python3
"""Guard leve dos exemplos de comandos na documentação operacional.

Valida que README e documentos operacionais mantenham exemplos de comandos:
- válidos e apontando para scripts existentes;
- canônicos, sem promover wrappers históricos como padrão;
- coerentes com a rotina oficial da baseline.

Uso:
    python scripts/quality/documentation_commands_examples_guard.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality import compatibility_contract_guard

DOCS_TO_VALIDATE: list[str] = [
    "README.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
    "docs/validation/VALIDACAO_MANUAL_GUIA.md",
]

REQUIRED_COMMANDS_BY_DOC: dict[str, list[str]] = {
    "README.md": [
        "python scripts/quality/runtime_preflight.py",
        "python scripts/quality/quality_gate.py",
        "python scripts/quality/documentation_commands_examples_guard.py",
        "python scripts/quality/release_manifest_guard.py",
        "python scripts/quality/quality_runtime_budget_guard.py",
        "python scripts/quality/script_exit_codes_contract_guard.py",
        "python scripts/quality/governance_docs_crosslinks_guard.py",
        "python scripts/quality/protected_scope_hash_guard.py",
        "python scripts/quality/checks_registry_contract_guard.py",
        "python scripts/quality/checks_registry_schema_guard.py",
        "python scripts/quality/quality_gate_composition_guard.py",
        "python scripts/reports/release_health_report.py",
        "python scripts/reports/maintenance_snapshot_report.py",
        "python scripts/reports/maintenance_handoff_pack.py",
        "python scripts/reports/maintenance_resume_brief.py",
        "python scripts/reports/maintenance_command_journal.py",
        "python scripts/reports/maintenance_reports_cleanup.py",
    ],
    "docs/operations/OPERACAO_LOCAL.md": [
        "pip install -r requirements.txt",
        "python scripts/quality/runtime_preflight.py",
        "python scripts/quality/quality_gate.py",
        "python scripts/quality/documentation_commands_examples_guard.py",
        "python scripts/quality/release_manifest_guard.py",
        "python scripts/quality/quality_runtime_budget_guard.py",
        "python scripts/quality/script_exit_codes_contract_guard.py",
        "python scripts/quality/governance_docs_crosslinks_guard.py",
        "python scripts/quality/protected_scope_hash_guard.py",
        "python scripts/quality/checks_registry_contract_guard.py",
        "python scripts/quality/checks_registry_schema_guard.py",
        "python scripts/quality/quality_gate_composition_guard.py",
        "python scripts/reports/manual_validation_pack.py",
        "python scripts/reports/release_health_report.py",
        "python scripts/reports/maintenance_snapshot_report.py",
        "python scripts/reports/maintenance_handoff_pack.py",
        "python scripts/reports/maintenance_resume_brief.py",
        "python scripts/reports/maintenance_command_journal.py",
        "python scripts/reports/maintenance_reports_cleanup.py",
        "streamlit run app.py",
    ],
    "docs/releases/RELEASE_OPERACIONAL.md": [
        "python scripts/quality/check_base.py",
        "python scripts/validation/smoke_test_base.py",
        "python -m compileall .",
        "python scripts/quality/documentation_commands_examples_guard.py",
        "python scripts/quality/release_manifest_guard.py",
        "python scripts/quality/quality_runtime_budget_guard.py",
        "python scripts/quality/script_exit_codes_contract_guard.py",
        "python scripts/quality/governance_docs_crosslinks_guard.py",
        "python scripts/quality/protected_scope_hash_guard.py",
        "python scripts/quality/checks_registry_contract_guard.py",
        "python scripts/quality/checks_registry_schema_guard.py",
        "python scripts/quality/quality_gate_composition_guard.py",
        "python scripts/quality/release_guard.py",
        "python scripts/quality/quality_gate.py",
        "python scripts/reports/release_health_report.py",
        "python scripts/reports/maintenance_snapshot_report.py",
        "python scripts/reports/maintenance_handoff_pack.py",
        "python scripts/reports/maintenance_resume_brief.py",
        "python scripts/reports/maintenance_command_journal.py",
        "python scripts/reports/maintenance_reports_cleanup.py",
    ],
    "docs/validation/VALIDACAO_MANUAL_GUIA.md": [
        "pip install -r requirements.txt",
        "python scripts/quality/runtime_preflight.py",
        "python scripts/quality/quality_gate.py",
        "python scripts/quality/documentation_commands_examples_guard.py",
        "python scripts/quality/release_manifest_guard.py",
        "python scripts/quality/quality_runtime_budget_guard.py",
        "python scripts/quality/script_exit_codes_contract_guard.py",
        "python scripts/quality/governance_docs_crosslinks_guard.py",
        "python scripts/quality/protected_scope_hash_guard.py",
        "python scripts/quality/checks_registry_contract_guard.py",
        "python scripts/quality/checks_registry_schema_guard.py",
        "python scripts/quality/quality_gate_composition_guard.py",
        "python scripts/reports/manual_validation_pack.py",
        "python scripts/reports/release_health_report.py",
        "streamlit run app.py",
    ],
}

ALLOWED_NON_SCRIPT_COMMANDS = {
    "pip install -r requirements.txt",
    "streamlit run app.py",
    "python -m compileall .",
    "python -m venv .venv",
}

COMMAND_PATTERN = re.compile(r"`((?:python|pip|streamlit)\s+[^`]+)`")
HISTORICAL_COMMANDS = {f"python {path}" for path in compatibility_contract_guard.WRAPPER_EXPECTATIONS}
CANONICAL_PREFIXES = (
    "python scripts/quality/",
    "python scripts/validation/",
    "python scripts/reports/",
)


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def extract_commands(text: str) -> list[str]:
    commands: list[str] = []
    seen: set[str] = set()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("python ", "pip ", "streamlit ")):
            cmd = stripped
            if cmd not in seen:
                commands.append(cmd)
                seen.add(cmd)
        for match in COMMAND_PATTERN.findall(line):
            cmd = match.strip()
            if cmd not in seen:
                commands.append(cmd)
                seen.add(cmd)
    return commands


def validate_command(command: str, rel_path: str, notes: list[str], errors: list[str]) -> None:
    if command in ALLOWED_NON_SCRIPT_COMMANDS:
        if command == "pip install -r requirements.txt" and not (ROOT / "requirements.txt").exists():
            errors.append(f"{rel_path} referencia `{command}`, mas requirements.txt não existe")
        elif command == "streamlit run app.py" and not (ROOT / "app.py").exists():
            errors.append(f"{rel_path} referencia `{command}`, mas app.py não existe")
        else:
            notes.append(f"OK comando documentado válido em {rel_path}: {command}")
        return

    if command in HISTORICAL_COMMANDS:
        errors.append(f"{rel_path} não deve promover wrapper histórico como exemplo principal: `{command}`")
        return

    if command.startswith("python "):
        target = command[len("python ") :].strip()
        if target.startswith("-m "):
            if command not in ALLOWED_NON_SCRIPT_COMMANDS:
                errors.append(f"{rel_path} usa comando de módulo Python não autorizado: `{command}`")
            else:
                notes.append(f"OK comando Python de módulo permitido em {rel_path}: {command}")
            return
        if not target.startswith("scripts/"):
            errors.append(f"{rel_path} usa comando Python fora do padrão operacional esperado: `{command}`")
            return
        if not command.startswith(CANONICAL_PREFIXES):
            errors.append(f"{rel_path} deve usar caminho canônico no exemplo: `{command}`")
            return
        script_path = ROOT / target
        if not script_path.exists():
            errors.append(f"{rel_path} referencia comando inexistente: `{command}`")
            return
        notes.append(f"OK comando canônico existente em {rel_path}: {command}")
        return

    if command.startswith(("pip ", "streamlit ")):
        errors.append(f"{rel_path} contém comando não reconhecido pelo guard: `{command}`")


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []

    for rel_path in DOCS_TO_VALIDATE:
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f"Documento ausente para validação: {rel_path}")
            continue

        text = read_text(rel_path)
        commands = extract_commands(text)
        if not commands:
            errors.append(f"{rel_path} deve conter pelo menos um exemplo de comando operacional")
            continue

        for required in REQUIRED_COMMANDS_BY_DOC.get(rel_path, []):
            if required not in commands:
                errors.append(f"{rel_path} deve citar o exemplo canônico `{required}`")
            else:
                notes.append(f"OK exemplo obrigatório presente em {rel_path}: {required}")

        for command in commands:
            validate_command(command, rel_path, notes, errors)

    print("=== DOCUMENTATION COMMANDS EXAMPLES GUARD | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")

    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nExemplos de comandos da documentação íntegros.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
