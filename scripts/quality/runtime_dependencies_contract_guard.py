#!/usr/bin/env python3
"""Guard leve do contrato mínimo de dependências de runtime local.

Valida a sincronização entre:
- ``requirements.txt``;
- ``scripts/quality/runtime_preflight.py``;
- documentação operacional oficial da base.

Objetivo:
- reduzir deriva entre instalação documentada, preflight local e dependências
  críticas para abrir o app;
- manter a rotina operacional previsível sem tocar no núcleo funcional.

Uso:
    python scripts/quality/runtime_dependencies_contract_guard.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality import runtime_preflight

EXPECTED_RUNTIME_DEPENDENCIES: list[tuple[str, str]] = [
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("pulp", "pulp"),
    ("xlsxwriter", "xlsxwriter"),
    ("openpyxl", "openpyxl"),
]

DOC_MARKERS: dict[str, list[str]] = {
    "README.md": [
        "pip install -r requirements.txt",
        "python scripts/quality/runtime_preflight.py",
        "python scripts/quality/runtime_dependencies_contract_guard.py",
        "streamlit run app.py",
    ],
    "docs/operations/OPERACAO_LOCAL.md": [
        "pip install -r requirements.txt",
        "python scripts/quality/runtime_preflight.py",
        "python scripts/quality/runtime_dependencies_contract_guard.py",
        "streamlit run app.py",
    ],
    "docs/releases/RELEASE_OPERACIONAL.md": [
        "python scripts/quality/runtime_preflight.py",
        "python scripts/quality/runtime_dependencies_contract_guard.py",
    ],
}


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def normalize_package_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def parse_requirements_packages(text: str) -> set[str]:
    packages: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        line = line.split(";", 1)[0].strip()
        name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip()
        if name:
            packages.add(normalize_package_name(name))
    return packages


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []

    requirements_path = ROOT / "requirements.txt"
    if not requirements_path.exists():
        errors.append("requirements.txt ausente na raiz do projeto")
        requirements_packages: set[str] = set()
    else:
        requirements_packages = parse_requirements_packages(requirements_path.read_text(encoding="utf-8"))
        notes.append("OK requirements.txt encontrado e lido com sucesso")

    expected_packages = {normalize_package_name(package) for _, package in EXPECTED_RUNTIME_DEPENDENCIES}
    missing_requirements = sorted(expected_packages - requirements_packages)
    if missing_requirements:
        errors.append(
            "requirements.txt deve conter as dependências críticas de runtime: " + ", ".join(missing_requirements)
        )
    else:
        notes.append("OK requirements.txt contém as dependências críticas de runtime")

    preflight_dependencies = {(module, package) for module, package in runtime_preflight.REQUIRED_MODULES}
    expected_dependencies = set(EXPECTED_RUNTIME_DEPENDENCIES)

    missing_in_preflight = sorted(expected_dependencies - preflight_dependencies)
    unexpected_in_preflight = sorted(preflight_dependencies - expected_dependencies)
    if missing_in_preflight:
        errors.append(
            "scripts/quality/runtime_preflight.py deve verificar as dependências críticas: "
            + ", ".join(f"{module}/{package}" for module, package in missing_in_preflight)
        )
    else:
        notes.append("OK runtime_preflight verifica as dependências críticas esperadas")

    if unexpected_in_preflight:
        errors.append(
            "scripts/quality/runtime_preflight.py contém dependências fora do contrato mínimo: "
            + ", ".join(f"{module}/{package}" for module, package in unexpected_in_preflight)
        )
    else:
        notes.append("OK runtime_preflight não introduz dependências fora do contrato mínimo")

    preflight_packages = {normalize_package_name(package) for _, package in runtime_preflight.REQUIRED_MODULES}
    if preflight_packages != expected_packages:
        errors.append("requirements.txt e runtime_preflight.py devem permanecer sincronizados nas dependências críticas")
    else:
        notes.append("OK requirements.txt e runtime_preflight.py estão sincronizados")

    preflight_text = read_text("scripts/quality/runtime_preflight.py")
    for marker in ("python scripts/quality/runtime_preflight.py", "pip install -r requirements.txt"):
        if marker not in preflight_text:
            errors.append(f"scripts/quality/runtime_preflight.py deve orientar o uso de `{marker}`")
        else:
            notes.append(f"OK runtime_preflight orienta o uso de: {marker}")

    for rel_path, markers in DOC_MARKERS.items():
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f"Documento operacional ausente: {rel_path}")
            continue
        text = read_text(rel_path)
        for marker in markers:
            if marker not in text:
                errors.append(f"{rel_path} deve citar `{marker}`")
            else:
                notes.append(f"OK documentação cita marcador de runtime: {rel_path} -> {marker}")

    print("=== RUNTIME DEPENDENCIES CONTRACT GUARD | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")

    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nContrato mínimo de dependências de runtime íntegro.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
