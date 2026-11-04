#!/usr/bin/env python3
"""Guard leve de higiene dos artefatos do pacote de release.

Valida que a baseline permaneça limpa de resíduos transitórios, arquivos gerados
indevidamente e artefatos locais que não devem seguir dentro do pacote oficial.

Contrato adotado:
- remover caches; tolerar somente pastas ``__pycache__`` comprovadamente vazias\n  que o Windows/OneDrive impeça de excluir; nunca tolerar bytecode residual;
- não permitir diretórios transitórios como ``work_*`` ou ``tmp_*`` dentro da base;
- não permitir arquivos ``.zip`` internos ao repositório;
- manter ``reports/`` como diretório limpo, contendo apenas ``.gitkeep``;
- documentar o check canônico na operação local e no protocolo de release.

Uso:
    python scripts/quality/release_artifacts_hygiene_guard.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_TO_VALIDATE = ["README.md", "docs/operations/OPERACAO_LOCAL.md", "docs/releases/RELEASE_OPERACIONAL.md"]
CANONICAL_DOC_MARKER = "python scripts/quality/release_artifacts_hygiene_guard.py"
ALLOWED_REPORTS = {".gitkeep"}
FORBIDDEN_DIR_PREFIXES = ("work_", "tmp_")
FORBIDDEN_DIR_NAMES = {"build", "dist", ".pytest_cache", ".mypy_cache"}
FORBIDDEN_FILE_SUFFIXES = {".pyc", ".pyo"}
FORBIDDEN_FILE_NAMES = {"Thumbs.db", ".DS_Store"}
EXCLUDED_TOP_LEVEL_DIRS = {".git", ".venv", "venv", ".idea", ".vscode", "node_modules"}


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def iter_project_paths() -> list[Path]:
    paths: list[Path] = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        first = rel.parts[0] if rel.parts else ""
        if first in EXCLUDED_TOP_LEVEL_DIRS:
            continue
        paths.append(path)
    return paths


def is_empty_directory(path: Path) -> bool:
    """Somente pastas verificavelmente vazias podem ser toleradas."""
    try:
        return path.is_dir() and not any(path.iterdir())
    except OSError:
        return False


def remove_bytecode_artifacts(root: Path) -> tuple[int, int]:
    removed_cache = 0
    removed_compiled = 0
    for pycache in sorted(root.rglob("__pycache__"), key=lambda p: len(p.parts), reverse=True):
        if not pycache.is_dir():
            continue
        try:
            for child in pycache.iterdir():
                if child.is_file() and child.suffix in FORBIDDEN_FILE_SUFFIXES:
                    child.unlink()
                    removed_compiled += 1
            pycache.rmdir()
            removed_cache += 1
        except OSError:
            # OneDrive pode bloquear rmdir() até em pasta já vazia.
            # Uma pasta não vazia ou inacessível será reprovada na inspeção.
            pass

    for suffix in FORBIDDEN_FILE_SUFFIXES:
        for compiled in root.rglob(f"*{suffix}"):
            if compiled.exists() and compiled.is_file():
                try:
                    compiled.unlink()
                    removed_compiled += 1
                except OSError:
                    # A auditoria abaixo reprova qualquer bytecode remanescente.
                    pass
    return removed_cache, removed_compiled

def main() -> int:
    errors: list[str] = []
    notes: list[str] = []
    removed_cache, removed_compiled = remove_bytecode_artifacts(ROOT)
    if removed_cache or removed_compiled:
        notes.append(f"OK limpeza pré-checagem: {removed_cache} __pycache__ removidos, {removed_compiled} arquivos compilados removidos")
    all_paths = iter_project_paths()
    all_pycache = [path for path in all_paths if path.is_dir() and path.name == "__pycache__"]
    pycache_dirs = sorted(
        path.relative_to(ROOT).as_posix()
        for path in all_pycache
        if not is_empty_directory(path)
    )
    empty_pycache = sorted(
        path.relative_to(ROOT).as_posix()
        for path in all_pycache
        if is_empty_directory(path)
    )
    if pycache_dirs:
        errors.append("Diretórios __pycache__ não vazios/inacessíveis no pacote: " + ", ".join(pycache_dirs))
    else:
        notes.append("OK sem diretórios __pycache__ com conteúdo")
    if empty_pycache:
        notes.append(
            "AVISO: __pycache__ vazios não excluídos pelo sistema de arquivos: "
            + ", ".join(empty_pycache)
        )
    compiled_files = sorted(str(path.relative_to(ROOT)) for path in all_paths if path.is_file() and path.suffix in FORBIDDEN_FILE_SUFFIXES)
    if compiled_files:
        errors.append("Arquivos compilados temporários não devem existir no pacote: " + ", ".join(compiled_files))
    else:
        notes.append("OK pacote sem arquivos .pyc/.pyo")
    transient_dirs = sorted(str(path.relative_to(ROOT)) for path in all_paths if path.is_dir() and (path.name in FORBIDDEN_DIR_NAMES or any(path.name.startswith(prefix) for prefix in FORBIDDEN_DIR_PREFIXES)))
    if transient_dirs:
        errors.append("Diretórios transitórios não devem existir no pacote: " + ", ".join(transient_dirs))
    else:
        notes.append("OK pacote sem diretórios transitórios de trabalho")
    nested_zips = sorted(str(path.relative_to(ROOT)) for path in all_paths if path.is_file() and path.suffix.lower() == ".zip")
    if nested_zips:
        errors.append("Arquivos .zip internos não devem existir no repositório: " + ", ".join(nested_zips))
    else:
        notes.append("OK pacote sem arquivos .zip internos")
    stray_files = sorted(str(path.relative_to(ROOT)) for path in all_paths if path.is_file() and path.name in FORBIDDEN_FILE_NAMES)
    if stray_files:
        errors.append("Arquivos de sistema transitórios não devem existir no pacote: " + ", ".join(stray_files))
    else:
        notes.append("OK pacote sem arquivos transitórios de sistema")
    reports_dir = ROOT / "reports"
    if not reports_dir.exists():
        errors.append("Diretório reports/ ausente")
    else:
        entries = sorted(path.name for path in reports_dir.iterdir())
        unexpected = [name for name in entries if name not in ALLOWED_REPORTS]
        if unexpected:
            errors.append("reports/ deve permanecer limpo no pacote oficial; encontrados: " + ", ".join(unexpected))
        else:
            notes.append("OK reports/ limpo no pacote oficial (apenas .gitkeep)")
    for rel_path in DOCS_TO_VALIDATE:
        text = read_text(rel_path)
        if CANONICAL_DOC_MARKER not in text:
            errors.append(f"{rel_path} deve citar o comando canônico `{CANONICAL_DOC_MARKER}`")
        else:
            notes.append(f"OK documentação cita comando canônico: {rel_path} -> {CANONICAL_DOC_MARKER}")
    print("=== RELEASE ARTIFACTS HYGIENE GUARD | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")
    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1
    print("\nHigiene dos artefatos de release íntegra.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
