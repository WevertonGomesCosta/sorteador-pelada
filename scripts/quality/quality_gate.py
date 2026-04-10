#!/usr/bin/env python3
"""Runner único dos checks técnicos oficiais da base.

Executa, em sequência:
- scripts/quality/check_base.py
- scripts/validation/smoke_test_base.py
- python -m compileall .
- scripts/quality/release_metadata_guard.py
- scripts/quality/compatibility_contract_guard.py
- scripts/quality/operational_checks_contract_guard.py
- scripts/quality/canonical_paths_reference_guard.py
- scripts/quality/script_cli_contract_guard.py
- scripts/quality/release_artifacts_hygiene_guard.py
- scripts/quality/runtime_dependencies_contract_guard.py
- scripts/quality/documentation_commands_examples_guard.py
- scripts/quality/release_guard.py

Uso:
    python scripts/quality/quality_gate.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHECKS: list[tuple[str, list[str]]] = [
    ("check_base", [sys.executable, str(ROOT / "scripts" / "quality" / "check_base.py")]),
    ("smoke_test_base", [sys.executable, str(ROOT / "scripts" / "validation" / "smoke_test_base.py")]),
    ("compileall", [sys.executable, "-m", "compileall", "."]),
    ("release_metadata_guard", [sys.executable, str(ROOT / "scripts" / "quality" / "release_metadata_guard.py")]),
    ("compatibility_contract_guard", [sys.executable, str(ROOT / "scripts" / "quality" / "compatibility_contract_guard.py")]),
    ("operational_checks_contract_guard", [sys.executable, str(ROOT / "scripts" / "quality" / "operational_checks_contract_guard.py")]),
    ("canonical_paths_reference_guard", [sys.executable, str(ROOT / "scripts" / "quality" / "canonical_paths_reference_guard.py")]),
    ("script_cli_contract_guard", [sys.executable, str(ROOT / "scripts" / "quality" / "script_cli_contract_guard.py")]),
    ("release_artifacts_hygiene_guard", [sys.executable, str(ROOT / "scripts" / "quality" / "release_artifacts_hygiene_guard.py")]),
    ("runtime_dependencies_contract_guard", [sys.executable, str(ROOT / "scripts" / "quality" / "runtime_dependencies_contract_guard.py")]),
    ("documentation_commands_examples_guard", [sys.executable, str(ROOT / "scripts" / "quality" / "documentation_commands_examples_guard.py")]),
    ("release_guard", [sys.executable, str(ROOT / "scripts" / "quality" / "release_guard.py")]),
]


def run_check(name: str, command: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    header = f"=== {name} ==="
    if output.strip():
        return proc.returncode == 0, f"{header}\n{output.strip()}"
    return proc.returncode == 0, header


def main() -> int:
    failures: list[str] = []
    outputs: list[str] = []

    print("=== QUALITY GATE | Sorteador Pelada PRO ===")
    for name, command in CHECKS:
        ok, output = run_check(name, command)
        status = "OK" if ok else "FALHOU"
        print(f"[{status}] {name}")
        outputs.append(output)
        if not ok:
            failures.append(name)

    print("\nResumo:")
    print(f"- Checks executados: {len(CHECKS)}")
    print(f"- Falhas: {len(failures)}")

    if failures:
        print("\nDetalhes:")
        for output in outputs:
            print(output)
            print()
        print("Quality gate interrompido.")
        return 1

    print("\nQuality gate concluído com sucesso.")
    print("Sugestão: depois, faça a validação manual no navegador com CHECKLIST_REGRESSAO.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
