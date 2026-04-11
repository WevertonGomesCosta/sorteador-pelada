#!/usr/bin/env python3
"""Runner único dos checks técnicos oficiais da base.

Executa a rotina oficial registrada em ``scripts/quality/checks_registry.py``.

Uso:
    python scripts/quality/quality_gate.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality.checks_registry import (
    CHECK_TIMEOUT_OVERRIDES,
    DEFAULT_CHECK_TIMEOUT_SECONDS,
    build_check_commands,
)

CHECKS: list[tuple[str, list[str]]] = build_check_commands(ROOT, sys.executable, target="quality_gate")


def run_check(name: str, command: list[str]) -> tuple[bool, str]:
    timeout_seconds = CHECK_TIMEOUT_OVERRIDES.get(name, DEFAULT_CHECK_TIMEOUT_SECONDS)
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        header = f"=== {name} ==="
        if output.strip():
            return proc.returncode == 0, f"{header}\nTimeout configurado: {timeout_seconds}s\n{output.strip()}"
        return proc.returncode == 0, f"{header}\nTimeout configurado: {timeout_seconds}s"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        output = stdout + (("\n" + stderr) if stderr else "")
        detail = output.strip() or "sem saída capturada antes do timeout"
        return False, f"=== {name} ===\nTimeout configurado: {timeout_seconds}s\nTIMEOUT\n{detail}"


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
