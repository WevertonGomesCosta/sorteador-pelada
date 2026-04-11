#!/usr/bin/env python3
"""Guard leve de coerência e navegabilidade entre documentos canônicos de governança.

Valida, sem tocar no núcleo funcional do app:
- que os documentos canônicos principais existam;
- que cada documento crítico aponte para os documentos complementares esperados;
- que o contrato de crosslinks use primeiro os caminhos canônicos;
- que nenhum documento central da governança fique órfão.

Uso:
    python scripts/quality/governance_docs_crosslinks_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANONICAL_DOCS: list[str] = [
    "README.md",
    "docs/releases/BASELINE_OFICIAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md",
    "docs/validation/VALIDACAO_MANUAL_GUIA.md",
    "docs/validation/PLANO_SMOKE_TEST_MINIMO.md",
]

EXPECTED_CROSSLINKS: dict[str, list[str]] = {
    "README.md": [
        "docs/releases/BASELINE_OFICIAL.md",
        "docs/releases/RELEASE_OPERACIONAL.md",
        "docs/operations/OPERACAO_LOCAL.md",
        "docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md",
        "docs/validation/VALIDACAO_MANUAL_GUIA.md",
        "docs/validation/PLANO_SMOKE_TEST_MINIMO.md",
        "python scripts/quality/governance_docs_crosslinks_guard.py",
    ],
    "docs/releases/BASELINE_OFICIAL.md": [
        "docs/releases/RELEASE_OPERACIONAL.md",
        "docs/operations/OPERACAO_LOCAL.md",
        "docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md",
        "docs/validation/VALIDACAO_MANUAL_GUIA.md",
        "python scripts/quality/governance_docs_crosslinks_guard.py",
    ],
    "docs/releases/RELEASE_OPERACIONAL.md": [
        "docs/releases/BASELINE_OFICIAL.md",
        "docs/operations/OPERACAO_LOCAL.md",
        "docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md",
        "docs/validation/VALIDACAO_MANUAL_GUIA.md",
        "python scripts/quality/governance_docs_crosslinks_guard.py",
    ],
    "docs/operations/OPERACAO_LOCAL.md": [
        "docs/releases/BASELINE_OFICIAL.md",
        "docs/releases/RELEASE_OPERACIONAL.md",
        "docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md",
        "docs/validation/VALIDACAO_MANUAL_GUIA.md",
        "python scripts/quality/governance_docs_crosslinks_guard.py",
    ],
    "docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md": [
        "docs/releases/BASELINE_OFICIAL.md",
        "docs/releases/RELEASE_OPERACIONAL.md",
        "docs/operations/OPERACAO_LOCAL.md",
        "python scripts/quality/governance_docs_crosslinks_guard.py",
    ],
    "docs/validation/VALIDACAO_MANUAL_GUIA.md": [
        "docs/releases/BASELINE_OFICIAL.md",
        "docs/releases/RELEASE_OPERACIONAL.md",
        "docs/operations/OPERACAO_LOCAL.md",
        "docs/validation/PLANO_SMOKE_TEST_MINIMO.md",
        "python scripts/quality/governance_docs_crosslinks_guard.py",
    ],
    "docs/validation/PLANO_SMOKE_TEST_MINIMO.md": [
        "docs/releases/BASELINE_OFICIAL.md",
        "docs/releases/RELEASE_OPERACIONAL.md",
        "docs/operations/OPERACAO_LOCAL.md",
        "docs/validation/VALIDACAO_MANUAL_GUIA.md",
        "python scripts/quality/governance_docs_crosslinks_guard.py",
    ],
}


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(__doc__.strip())
        return 0

    errors: list[str] = []
    notes: list[str] = []
    texts: dict[str, str] = {}

    for rel_path in CANONICAL_DOCS:
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f"Documento canônico ausente: {rel_path}")
            continue
        texts[rel_path] = read_text(rel_path)
        notes.append(f"OK documento canônico presente: {rel_path}")

    for rel_path, expected_links in EXPECTED_CROSSLINKS.items():
        text = texts.get(rel_path)
        if text is None:
            continue
        for marker in expected_links:
            if marker not in text:
                errors.append(f"{rel_path} deve citar `{marker}`")
            else:
                notes.append(f"OK crosslink presente: {rel_path} -> {marker}")

    incoming_counts = {rel_path: 0 for rel_path in CANONICAL_DOCS}
    for source, text in texts.items():
        for target in CANONICAL_DOCS:
            if source == target:
                continue
            if target in text:
                incoming_counts[target] += 1

    for rel_path, count in incoming_counts.items():
        if rel_path == "README.md":
            continue
        if count == 0:
            errors.append(f"{rel_path} não deve ficar órfão na governança documental canônica")
        else:
            notes.append(f"OK documento referenciado por outros documentos: {rel_path} ({count})")

    print("=== GOVERNANCE DOCS CROSSLINKS GUARD | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")

    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nCrosslinks canônicos de governança íntegros.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
