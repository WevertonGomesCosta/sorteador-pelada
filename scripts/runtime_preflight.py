#!/usr/bin/env python3
"""Pré-checagem leve do ambiente local para rodar o app.

Valida:
- presença dos arquivos essenciais;
- importabilidade das dependências declaradas;
- disponibilidade mínima do runtime para abrir o app com Streamlit.

Uso:
    python scripts/runtime_preflight.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "app.py",
    "requirements.txt",
    "data/repository.py",
    "core/logic.py",
    "core/optimizer.py",
    "ui/primitives.py",
]

REQUIRED_MODULES: list[tuple[str, str]] = [
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("pulp", "pulp"),
    ("xlsxwriter", "xlsxwriter"),
    ("openpyxl", "openpyxl"),
]


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []

    print("=== RUNTIME PREFLIGHT | Sorteador Pelada PRO ===")
    print(f"Python detectado: {sys.version.split()[0]}")

    for rel_path in REQUIRED_FILES:
        if (ROOT / rel_path).exists():
            notes.append(f"OK arquivo essencial: {rel_path}")
        else:
            errors.append(f"Arquivo essencial ausente: {rel_path}")

    for module_name, package_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "sem __version__")
            notes.append(f"OK dependência: {package_name} ({version})")
        except Exception as exc:
            errors.append(f"Dependência indisponível: {package_name} ({exc.__class__.__name__}: {exc})")

    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        print("\nAção sugerida:")
        print("- pip install -r requirements.txt")
        print("- depois execute novamente: python scripts/runtime_preflight.py")
        return 1

    print("\nChecagens concluídas:")
    for note in notes:
        print(f"[OK] {note}")

    print("\nAmbiente pronto para abrir o app.")
    print("Próximo passo sugerido: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
