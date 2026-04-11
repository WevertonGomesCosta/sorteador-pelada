#!/usr/bin/env python3
"""Guard leve do manifesto oficial de hashes do escopo protegido.

Valida, sem tocar no núcleo funcional do app:
- a presença do manifesto oficial de hashes do escopo protegido;
- a integridade dos hashes de ``app.py`` e ``ui/review_view.py``;
- a documentação mínima da política de escopo protegido nos documentos oficiais.

Uso:
    python scripts/quality/protected_scope_hash_guard.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = ROOT / "docs" / "releases" / "PROTECTED_SCOPE_HASHES.json"
PROTECTED_DOCS = [
    "README.md",
    "docs/releases/BASELINE_OFICIAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
    "docs/operations/OPERACAO_LOCAL.md",
]
DOC_MARKERS = [
    "python scripts/quality/protected_scope_hash_guard.py",
    "docs/releases/PROTECTED_SCOPE_HASHES.json",
    "escopo protegido",
]


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(__doc__.strip())
        return 0

    errors: list[str] = []
    notes: list[str] = []

    if not MANIFEST_PATH.exists():
        print("=== PROTECTED SCOPE HASH GUARD | Sorteador Pelada PRO ===")
        print("\nErros encontrados:")
        print(f" - Manifesto ausente: {MANIFEST_PATH.relative_to(ROOT).as_posix()}")
        return 1

    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print("=== PROTECTED SCOPE HASH GUARD | Sorteador Pelada PRO ===")
        print("\nErros encontrados:")
        print(f" - Manifesto inválido: {exc}")
        return 1

    if payload.get("schema_version") != 1:
        errors.append("docs/releases/PROTECTED_SCOPE_HASHES.json deve usar schema_version = 1")
    else:
        notes.append("OK schema_version do manifesto protegido")

    if payload.get("baseline_version") != "v85":
        errors.append("docs/releases/PROTECTED_SCOPE_HASHES.json deve registrar baseline_version = v85")
    else:
        notes.append("OK baseline_version do manifesto protegido")

    protected_map = payload.get("protected_files")
    if not isinstance(protected_map, dict):
        errors.append("docs/releases/PROTECTED_SCOPE_HASHES.json deve expor protected_files como objeto")
        protected_map = {}

    expected_files = {"app.py", "ui/review_view.py"}
    if set(protected_map) != expected_files:
        errors.append("docs/releases/PROTECTED_SCOPE_HASHES.json deve conter exatamente app.py e ui/review_view.py")
    else:
        notes.append("OK manifesto protegido lista exatamente os arquivos congelados")

    for rel_path, meta in protected_map.items():
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f"Arquivo protegido ausente: {rel_path}")
            continue
        if not isinstance(meta, dict):
            errors.append(f"Manifesto inválido para {rel_path}: entrada deve ser objeto")
            continue
        expected_hash = meta.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            errors.append(f"Manifesto inválido para {rel_path}: sha256 ausente ou malformado")
            continue
        current_hash = sha256_file(path)
        if current_hash != expected_hash:
            errors.append(f"Hash divergente em arquivo protegido: {rel_path}")
        else:
            notes.append(f"OK hash protegido íntegro: {rel_path}")
        if meta.get("status") != "frozen":
            errors.append(f"Manifesto de {rel_path} deve registrar status = frozen")
        else:
            notes.append(f"OK status frozen registrado: {rel_path}")

    for rel_path in PROTECTED_DOCS:
        text = read_text(rel_path)
        for marker in DOC_MARKERS:
            if marker not in text:
                errors.append(f"{rel_path} deve citar `{marker}`")
            else:
                notes.append(f"OK marcador documental presente: {rel_path} -> {marker}")

    print("=== PROTECTED SCOPE HASH GUARD | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")

    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nManifesto do escopo protegido íntegro.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
