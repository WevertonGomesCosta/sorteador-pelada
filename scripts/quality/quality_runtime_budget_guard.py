#!/usr/bin/env python3
"""Guard leve do orçamento operacional da rotina oficial de checks.

Valida, sem tocar no núcleo funcional do app:
- presença de timeouts explícitos nos runners compostos;
- sincronização da lista oficial de checks entre quality_gate e release_health_report;
- tempo de execução dos guards rápidos sob um orçamento operacional razoável;
- presença do novo contrato nas documentações operacionais oficiais.

Uso:
    python scripts/quality/quality_runtime_budget_guard.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality import quality_gate
from scripts.reports import release_health_report

DEFAULT_FAST_TIMEOUT_SECONDS = 30
FAST_CHECK_BUDGETS: list[tuple[str, str, int]] = [
    ("release_metadata_guard", "scripts/quality/release_metadata_guard.py", 10),
    ("compatibility_contract_guard", "scripts/quality/compatibility_contract_guard.py", 15),
    ("operational_checks_contract_guard", "scripts/quality/operational_checks_contract_guard.py", 15),
    ("canonical_paths_reference_guard", "scripts/quality/canonical_paths_reference_guard.py", 15),
    ("runtime_dependencies_contract_guard", "scripts/quality/runtime_dependencies_contract_guard.py", 15),
    ("documentation_commands_examples_guard", "scripts/quality/documentation_commands_examples_guard.py", 15),
    ("release_manifest_guard", "scripts/quality/release_manifest_guard.py", 15),
]
TOTAL_FAST_CHECKS_BUDGET_SECONDS = 90

REQUIRED_TIMEOUT_OVERRIDES = {
    "smoke_test_base": 180,
    "compileall": 180,
    "release_guard": 180,
    "quality_runtime_budget_guard": 120,
    "script_exit_codes_contract_guard": 120,
}

DOCS_TO_VALIDATE = [
    "README.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
]
DOC_MARKERS = [
    "python scripts/quality/quality_runtime_budget_guard.py",
    "python scripts/quality/script_exit_codes_contract_guard.py",
    "orçamento operacional",
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


def ensure_timeout_contract(module_name: str, module: object, errors: list[str], notes: list[str]) -> None:
    default_timeout = getattr(module, "DEFAULT_CHECK_TIMEOUT_SECONDS", None)
    overrides = getattr(module, "CHECK_TIMEOUT_OVERRIDES", None)
    if default_timeout is None:
        errors.append(f"{module_name} deve expor DEFAULT_CHECK_TIMEOUT_SECONDS")
    else:
        notes.append(f"OK timeout default exposto em {module_name}: {default_timeout}")
    if not isinstance(overrides, dict):
        errors.append(f"{module_name} deve expor CHECK_TIMEOUT_OVERRIDES")
        return
    notes.append(f"OK mapa de timeouts exposto em {module_name}")
    for key, expected in REQUIRED_TIMEOUT_OVERRIDES.items():
        actual = overrides.get(key)
        if actual != expected:
            errors.append(f"{module_name} deve definir timeout {expected}s para `{key}`")
        else:
            notes.append(f"OK timeout explícito em {module_name}: {key} -> {actual}s")


def run_fast_check(rel_path: str, timeout_seconds: int) -> tuple[bool, float, str]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / rel_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=min(timeout_seconds, DEFAULT_FAST_TIMEOUT_SECONDS),
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - started
        return False, elapsed, f"timeout > {timeout_seconds}s"

    elapsed = time.perf_counter() - started
    combined = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    if proc.returncode != 0:
        excerpt = combined.strip().splitlines()[:4]
        return False, elapsed, " | ".join(excerpt) if excerpt else f"returncode {proc.returncode}"
    if elapsed > timeout_seconds:
        return False, elapsed, f"tempo {elapsed:.2f}s acima do orçamento {timeout_seconds}s"
    return True, elapsed, "OK"


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(__doc__.strip())
        return 0

    errors: list[str] = []
    notes: list[str] = []

    ensure_timeout_contract("scripts/quality/quality_gate.py", quality_gate, errors, notes)
    ensure_timeout_contract("scripts/reports/release_health_report.py", release_health_report, errors, notes)

    quality_checks = normalize_checks(quality_gate.CHECKS)
    report_checks = normalize_checks(release_health_report.CHECKS)
    if quality_checks != report_checks:
        errors.append("quality_gate e release_health_report devem permanecer sincronizados na lista oficial de checks")
    else:
        notes.append("OK quality_gate e release_health_report estão sincronizados")

    if "quality_runtime_budget_guard" not in {name for name, _ in quality_checks}:
        errors.append("quality_gate deve incluir quality_runtime_budget_guard na rotina oficial")
    else:
        notes.append("OK quality_runtime_budget_guard integrado à rotina oficial")

    total_elapsed = 0.0
    for name, rel_path, budget in FAST_CHECK_BUDGETS:
        ok, elapsed, detail = run_fast_check(rel_path, budget)
        total_elapsed += elapsed
        if ok:
            notes.append(f"OK check rápido dentro do orçamento: {name} ({elapsed:.2f}s <= {budget}s)")
        else:
            errors.append(f"{name} fora do orçamento operacional: {detail}")

    if total_elapsed > TOTAL_FAST_CHECKS_BUDGET_SECONDS:
        errors.append(
            f"Conjunto de checks rápidos excedeu o orçamento total de {TOTAL_FAST_CHECKS_BUDGET_SECONDS}s ({total_elapsed:.2f}s)"
        )
    else:
        notes.append(
            f"OK orçamento agregado dos checks rápidos: {total_elapsed:.2f}s <= {TOTAL_FAST_CHECKS_BUDGET_SECONDS}s"
        )

    for rel_path in DOCS_TO_VALIDATE:
        text = read_text(rel_path)
        for marker in DOC_MARKERS:
            if marker not in text:
                errors.append(f"{rel_path} deve citar `{marker}`")
            else:
                notes.append(f"OK marcador documental presente: {rel_path} -> {marker}")

    print("=== QUALITY RUNTIME BUDGET GUARD | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")

    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nOrçamento operacional da rotina oficial de checks íntegro.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
