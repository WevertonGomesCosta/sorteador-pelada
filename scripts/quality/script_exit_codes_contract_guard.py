#!/usr/bin/env python3
"""Guard leve do contrato de códigos de saída dos scripts operacionais.

Valida, sem tocar no núcleo funcional do app:
- que scripts canônicos leves retornem exit code 0 em sucesso real;
- que wrappers temporários preservem a delegação explícita via ``raise SystemExit(main())``;
- que scripts canônicos exponham ``main()`` e apresentem caminhos estruturais previsíveis para sucesso e falha controlada;
- que a documentação operacional oficial cite o novo contrato canônico.

Uso:
    python scripts/quality/script_exit_codes_contract_guard.py
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

CANONICAL_SUCCESS_PROBE_TARGETS: list[str] = [
    "scripts/quality/release_metadata_guard.py",
    "scripts/quality/compatibility_contract_guard.py",
    "scripts/quality/canonical_paths_reference_guard.py",
    "scripts/quality/release_manifest_guard.py",
    "scripts/quality/release_artifacts_hygiene_guard.py",
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
    "scripts/quality/operational_checks_contract_guard.py",
    "scripts/quality/script_cli_contract_guard.py",
    "scripts/quality/documentation_commands_examples_guard.py",
    "scripts/quality/quality_runtime_budget_guard.py",
    "scripts/quality/script_exit_codes_contract_guard.py",
]

STRICT_RETURN_CONTRACT_TARGETS = set(CANONICAL_SUCCESS_PROBE_TARGETS) | {
    "scripts/quality/check_base.py",
    "scripts/quality/runtime_preflight.py",
    "scripts/quality/release_guard.py",
    "scripts/quality/quality_gate.py",
    "scripts/quality/operational_checks_contract_guard.py",
    "scripts/quality/script_cli_contract_guard.py",
    "scripts/quality/documentation_commands_examples_guard.py",
    "scripts/quality/script_exit_codes_contract_guard.py",
}

WRAPPER_TARGETS: list[str] = [
    "scripts/check_base.py",
    "scripts/runtime_preflight.py",
    "scripts/release_guard.py",
    "scripts/smoke_test_base.py",
    "scripts/quality_gate.py",
    "scripts/manual_validation_pack.py",
    "scripts/release_health_report.py",
    "scripts/release_metadata_guard.py",
    "scripts/compatibility_contract_guard.py",
    "scripts/operational_checks_contract_guard.py",
    "scripts/canonical_paths_reference_guard.py",
    "scripts/script_cli_contract_guard.py",
    "scripts/release_artifacts_hygiene_guard.py",
    "scripts/runtime_dependencies_contract_guard.py",
    "scripts/documentation_commands_examples_guard.py",
    "scripts/release_manifest_guard.py",
    "scripts/quality_runtime_budget_guard.py",
    "scripts/script_exit_codes_contract_guard.py",
]

DOCS_TO_VALIDATE: list[str] = [
    "README.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
]

DOC_MARKERS: list[str] = [
    "python scripts/quality/script_exit_codes_contract_guard.py",
    "códigos de saída previsíveis",
]

TRACEBACK_MARKERS = [
    "Traceback (most recent call last)",
    "ModuleNotFoundError",
    "ImportError",
]


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def parse_module(rel_path: str) -> ast.Module:
    return ast.parse(read_text(rel_path))


def has_defined_main(rel_path: str) -> bool:
    module = parse_module(rel_path)
    return any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in module.body)


def extract_docstring(rel_path: str) -> str:
    return ast.get_docstring(parse_module(rel_path)) or ""


def has_raise_system_exit_main(rel_path: str) -> bool:
    return "raise SystemExit(main())" in read_text(rel_path)


def main_return_contract(rel_path: str) -> tuple[bool, bool]:
    module = parse_module(rel_path)
    returns_zero = False
    returns_one = False
    for node in ast.walk(module):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
            if node.value.value == 0:
                returns_zero = True
            elif node.value.value == 1:
                returns_one = True
    return returns_zero, returns_one


def run_success_probe(rel_path: str, timeout_seconds: int = 120) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(ROOT / rel_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    excerpt = " | ".join(line.strip() for line in combined.splitlines()[:4] if line.strip())
    for marker in TRACEBACK_MARKERS:
        if marker in combined:
            return False, f"{rel_path} exibiu erro bruto: {marker}"
    if proc.returncode != 0:
        return False, f"{rel_path} deve retornar exit code 0 em sucesso real; retornou {proc.returncode}"
    return True, excerpt or "exit code 0 sem saída relevante"


def validate_canonical(rel_path: str, runtime_probe: bool, notes: list[str], errors: list[str]) -> None:
    if not (ROOT / rel_path).exists():
        errors.append(f"Script canônico ausente: {rel_path}")
        return
    if "Uso:" not in extract_docstring(rel_path):
        errors.append(f"{rel_path} deve declarar seção Uso: na docstring")
    else:
        notes.append(f"OK docstring com Uso: {rel_path}")
    if not has_defined_main(rel_path):
        errors.append(f"{rel_path} deve expor função main()")
    else:
        notes.append(f"OK função main() presente: {rel_path}")
    if not has_raise_system_exit_main(rel_path):
        errors.append(f"{rel_path} deve encerrar com raise SystemExit(main())")
    else:
        notes.append(f"OK SystemExit(main()) presente: {rel_path}")
    if rel_path in STRICT_RETURN_CONTRACT_TARGETS:
        returns_zero, returns_one = main_return_contract(rel_path)
        if not returns_zero:
            errors.append(f"{rel_path} deve conter caminho explícito para return 0")
        else:
            notes.append(f"OK contrato de sucesso encontrado: {rel_path}")
        if not returns_one:
            errors.append(f"{rel_path} deve conter caminho explícito para return 1")
        else:
            notes.append(f"OK contrato de falha controlada encontrado: {rel_path}")
    else:
        notes.append(f"OK contrato estrutural mínimo aceito: {rel_path}")
    if runtime_probe:
        ok, detail = run_success_probe(rel_path)
        if not ok:
            errors.append(detail)
        else:
            notes.append(f"OK probe de sucesso com exit code 0: {rel_path} ({detail})")


def validate_wrapper(rel_path: str, notes: list[str], errors: list[str]) -> None:
    if not (ROOT / rel_path).exists():
        errors.append(f"Wrapper histórico ausente: {rel_path}")
        return
    text = read_text(rel_path)
    expected = compatibility_contract_guard.WRAPPER_EXPECTATIONS.get(rel_path)
    if not expected:
        errors.append(f"{rel_path} deve constar em WRAPPER_EXPECTATIONS do compatibility_contract_guard")
        return
    if expected["canonical_path"] not in text:
        errors.append(f"{rel_path} deve apontar para o caminho canônico `{expected['canonical_path']}`")
    else:
        notes.append(f"OK wrapper aponta para caminho canônico: {rel_path} -> {expected['canonical_path']}")
    if expected["import_line"] not in text:
        errors.append(f"{rel_path} deve importar main via `{expected['import_line']}`")
    else:
        notes.append(f"OK import canônico do wrapper: {rel_path}")
    if not has_raise_system_exit_main(rel_path):
        errors.append(f"{rel_path} deve preservar o código de saída com raise SystemExit(main())")
    else:
        notes.append(f"OK wrapper preserva código de saída: {rel_path}")


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(__doc__.strip())
        return 0

    errors: list[str] = []
    notes: list[str] = []

    for rel_path in CANONICAL_SUCCESS_PROBE_TARGETS:
        validate_canonical(rel_path, runtime_probe=True, notes=notes, errors=errors)
    for rel_path in CANONICAL_STRUCTURAL_ONLY_TARGETS:
        validate_canonical(rel_path, runtime_probe=False, notes=notes, errors=errors)
    for rel_path in WRAPPER_TARGETS:
        validate_wrapper(rel_path, notes=notes, errors=errors)
    for rel_path in DOCS_TO_VALIDATE:
        text = read_text(rel_path)
        for marker in DOC_MARKERS:
            if marker not in text:
                errors.append(f"{rel_path} deve citar `{marker}`")
            else:
                notes.append(f"OK marcador documental presente: {rel_path} -> {marker}")

    print("=== SCRIPT EXIT CODES CONTRACT GUARD | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")

    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nContrato de códigos de saída dos scripts operacionais íntegro.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
