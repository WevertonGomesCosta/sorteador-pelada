#!/usr/bin/env python3
"""Guard leve do contrato mínimo de CLI dos scripts operacionais.

Valida que os scripts canônicos e wrappers temporários mantenham uma interface
mínima previsível para uso manual e documental, sem tocar no núcleo funcional.

Contrato adotado:
- scripts seguros devem aceitar ``--help`` sem traceback bruto;
- scripts compostos ou com efeitos colaterais ficam sob verificação estrutural,
  porque sua execução real já é coberta por outros guards e relatórios;
- wrappers históricos devem se declarar como compatibilidade temporária e
  continuar apontando para o caminho canônico correto.

Uso:
    python scripts/quality/script_cli_contract_guard.py
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality import compatibility_contract_guard

CANONICAL_SAFE_HELP_PROBE_TARGETS: list[str] = [
    "scripts/quality/release_metadata_guard.py",
    "scripts/quality/compatibility_contract_guard.py",
    "scripts/quality/operational_checks_contract_guard.py",
    "scripts/quality/canonical_paths_reference_guard.py",
    "scripts/quality/runtime_dependencies_contract_guard.py",
]

CANONICAL_STRUCTURAL_ONLY_TARGETS: list[str] = [
    "scripts/quality/check_base.py",
    "scripts/quality/runtime_preflight.py",
    "scripts/quality/release_guard.py",
    "scripts/validation/smoke_test_base.py",
    "scripts/quality/quality_gate.py",
    "scripts/reports/manual_validation_pack.py",
    "scripts/reports/release_health_report.py",
    "scripts/quality/script_cli_contract_guard.py",
]

WRAPPER_SAFE_HELP_PROBE_TARGETS: list[str] = [
    "scripts/release_metadata_guard.py",
    "scripts/compatibility_contract_guard.py",
    "scripts/operational_checks_contract_guard.py",
    "scripts/canonical_paths_reference_guard.py",
    "scripts/runtime_dependencies_contract_guard.py",
]

WRAPPER_STRUCTURAL_ONLY_TARGETS: list[str] = [
    "scripts/check_base.py",
    "scripts/runtime_preflight.py",
    "scripts/release_guard.py",
    "scripts/smoke_test_base.py",
    "scripts/quality_gate.py",
    "scripts/manual_validation_pack.py",
    "scripts/release_health_report.py",
    "scripts/script_cli_contract_guard.py",
]

DOCS_TO_VALIDATE: list[str] = [
    "README.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
]

CANONICAL_DOC_MARKERS: list[str] = [
    "python scripts/quality/script_cli_contract_guard.py",
    "python scripts/quality/runtime_dependencies_contract_guard.py",
]

TRACEBACK_MARKERS = [
    "Traceback (most recent call last)",
    "ModuleNotFoundError",
    "ImportError",
]

WRAPPER_MARKERS = [
    "Wrapper temporário de compatibilidade",
    "Padrão oficial atual:",
]


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def extract_docstring(rel_path: str) -> str:
    module = ast.parse(read_text(rel_path))
    return ast.get_docstring(module) or ""


def has_defined_main(rel_path: str) -> bool:
    module = ast.parse(read_text(rel_path))
    return any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in module.body)


def run_help_probe(rel_path: str) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(ROOT / rel_path), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    excerpt = " | ".join(line.strip() for line in combined.splitlines()[:3] if line.strip())
    for marker in TRACEBACK_MARKERS:
        if marker in combined:
            return False, f"{rel_path} exibiu erro bruto no probe de --help: {marker}"
    if proc.returncode in (0, 1):
        return True, excerpt or f"saída controlada com return code {proc.returncode}"
    return False, f"{rel_path} retornou code inesperado {proc.returncode} no probe de --help"


def validate_canonical_script(rel_path: str, probe_help: bool, notes: list[str], errors: list[str]) -> None:
    if not (ROOT / rel_path).exists():
        errors.append(f"Script canônico ausente: {rel_path}")
        return
    docstring = extract_docstring(rel_path)
    if "Uso:" not in docstring:
        errors.append(f"{rel_path} deve declarar seção Uso: na docstring")
    else:
        notes.append(f"OK docstring com Uso: {rel_path}")
    if not has_defined_main(rel_path):
        errors.append(f"{rel_path} deve expor função main()")
    else:
        notes.append(f"OK função main() presente: {rel_path}")
    if probe_help:
        ok, detail = run_help_probe(rel_path)
        if not ok:
            errors.append(detail)
        else:
            notes.append(f"OK probe de --help: {rel_path} ({detail})")


def validate_wrapper(rel_path: str, probe_help: bool, notes: list[str], errors: list[str]) -> None:
    if not (ROOT / rel_path).exists():
        errors.append(f"Wrapper histórico ausente: {rel_path}")
        return
    text = read_text(rel_path)
    for marker in WRAPPER_MARKERS:
        if marker not in text:
            errors.append(f"{rel_path} deve conter o marcador `{marker}`")
        else:
            notes.append(f"OK marcador de wrapper presente: {rel_path} -> {marker}")
    expected = compatibility_contract_guard.WRAPPER_EXPECTATIONS.get(rel_path)
    if not expected:
        errors.append(f"{rel_path} deve constar em WRAPPER_EXPECTATIONS do compatibility_contract_guard")
    else:
        if expected["canonical_path"] not in text:
            errors.append(f"{rel_path} deve apontar para o caminho canônico `{expected['canonical_path']}`")
        else:
            notes.append(f"OK wrapper aponta para caminho canônico: {rel_path} -> {expected['canonical_path']}")
        if expected["import_line"] not in text:
            errors.append(f"{rel_path} deve importar main via `{expected['import_line']}`")
        else:
            notes.append(f"OK import canônico do wrapper: {rel_path}")
    if probe_help:
        ok, detail = run_help_probe(rel_path)
        if not ok:
            errors.append(detail)
        else:
            notes.append(f"OK probe de --help no wrapper: {rel_path} ({detail})")


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []

    for rel_path in CANONICAL_SAFE_HELP_PROBE_TARGETS:
        validate_canonical_script(rel_path, probe_help=True, notes=notes, errors=errors)

    for rel_path in CANONICAL_STRUCTURAL_ONLY_TARGETS:
        validate_canonical_script(rel_path, probe_help=False, notes=notes, errors=errors)

    for rel_path in WRAPPER_SAFE_HELP_PROBE_TARGETS:
        validate_wrapper(rel_path, probe_help=True, notes=notes, errors=errors)

    for rel_path in WRAPPER_STRUCTURAL_ONLY_TARGETS:
        validate_wrapper(rel_path, probe_help=False, notes=notes, errors=errors)

    for rel_path in DOCS_TO_VALIDATE:
        text = read_text(rel_path)
        for marker in CANONICAL_DOC_MARKERS:
            if marker not in text:
                errors.append(f"{rel_path} deve citar o comando canônico `{marker}`")
            else:
                notes.append(f"OK documentação cita comando canônico: {rel_path} -> {marker}")

    print("=== SCRIPT CLI CONTRACT GUARD | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")

    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nContrato mínimo de CLI dos scripts operacionais íntegro.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
