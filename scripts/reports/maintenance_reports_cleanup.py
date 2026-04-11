#!/usr/bin/env python3
"""Higieniza com segurança os artefatos transitórios em reports/.

Objetivo:
- limpar o diretório ``reports/`` antes de revisão final, handoff ou empacotamento da baseline;
- preservar apenas ``.gitkeep`` e não tocar em conteúdo inesperado por padrão;
- mover artefatos conhecidos para um arquivo temporário fora do repositório, com opção explícita de remoção definitiva.

Uso:
    python scripts/reports/maintenance_reports_cleanup.py
    python scripts/reports/maintenance_reports_cleanup.py --dry-run
    python scripts/reports/maintenance_reports_cleanup.py --delete
    python scripts/reports/maintenance_reports_cleanup.py --include-unexpected
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reports import maintenance_snapshot_report

OUTPUT_DIR = ROOT / "reports"
PRESERVED_NAMES = {".gitkeep"}
KNOWN_GENERATED_PREFIXES = (
    "validacao_manual_",
    "release_health_",
    "maintenance_snapshot_",
    "maintenance_handoff_",
    "maintenance_resume_brief_",
    "maintenance_command_journal_",
)


@dataclass(slots=True)
class CleanupPlan:
    preserved: list[Path]
    known_candidates: list[Path]
    unexpected_files: list[Path]
    unexpected_dirs: list[Path]

    @property
    def is_clean(self) -> bool:
        return not self.known_candidates and not self.unexpected_files and not self.unexpected_dirs



def classify_reports_dir() -> CleanupPlan:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preserved: list[Path] = []
    known_candidates: list[Path] = []
    unexpected_files: list[Path] = []
    unexpected_dirs: list[Path] = []

    for path in sorted(OUTPUT_DIR.iterdir(), key=lambda item: item.name):
        if path.name in PRESERVED_NAMES:
            preserved.append(path)
            continue
        if path.is_dir():
            unexpected_dirs.append(path)
            continue
        if path.is_file() and path.name.startswith(KNOWN_GENERATED_PREFIXES):
            known_candidates.append(path)
            continue
        unexpected_files.append(path)

    return CleanupPlan(
        preserved=preserved,
        known_candidates=known_candidates,
        unexpected_files=unexpected_files,
        unexpected_dirs=unexpected_dirs,
    )



def _format_rel_paths(paths: list[Path]) -> list[str]:
    return [str(path.relative_to(ROOT)) for path in paths]



def archive_candidates(paths: list[Path], version: str) -> Path | None:
    if not paths:
        return None
    archive_root = Path(tempfile.mkdtemp(prefix=f"maintenance_reports_cleanup_{version}_"))
    for path in paths:
        target = archive_root / path.name
        shutil.move(str(path), str(target))
    return archive_root



def delete_candidates(paths: list[Path]) -> None:
    for path in paths:
        if path.exists() and path.is_file():
            path.unlink()



def cleanup_reports(*, dry_run: bool = False, delete: bool = False, include_unexpected: bool = False) -> dict[str, object]:
    version = maintenance_snapshot_report.detect_version()
    before = classify_reports_dir()
    selected: list[Path] = list(before.known_candidates)
    if include_unexpected:
        selected.extend(before.unexpected_files)

    archive_path: Path | None = None
    action = "none"

    if not dry_run and selected:
        if delete:
            delete_candidates(selected)
            action = "delete"
        else:
            archive_path = archive_candidates(selected, version)
            action = "archive"

    after = classify_reports_dir()
    return {
        "version": version,
        "dry_run": dry_run,
        "delete": delete,
        "include_unexpected": include_unexpected,
        "action": action,
        "selected": selected,
        "archive_path": archive_path,
        "before": before,
        "after": after,
        "clean_after": after.is_clean,
    }



def build_console_lines(summary: dict[str, object]) -> list[str]:
    before: CleanupPlan = summary["before"]  # type: ignore[assignment]
    after: CleanupPlan = summary["after"]  # type: ignore[assignment]
    selected: list[Path] = summary["selected"]  # type: ignore[assignment]
    archive_path = summary["archive_path"]
    action = str(summary["action"])

    lines: list[str] = []
    lines.append("=== MAINTENANCE REPORTS CLEANUP | Sorteador Pelada PRO ===")
    lines.append(f"Versão detectada: {summary['version']}")
    lines.append("Diretório alvo: reports/")
    if summary["dry_run"]:
        lines.append("Modo: dry-run (somente pré-visualização)")
    elif action == "archive":
        lines.append("Modo: limpeza segura com arquivamento fora do repositório")
    elif action == "delete":
        lines.append("Modo: remoção definitiva dos artefatos selecionados")
    else:
        lines.append("Modo: nenhuma ação aplicada")

    lines.append("")
    lines.append("Conteúdo preservado:")
    preserved_paths = _format_rel_paths(before.preserved)
    if preserved_paths:
        for rel in preserved_paths:
            lines.append(f"- PRESERVADO: {rel}")
    else:
        lines.append("- nenhum arquivo preservado identificado")

    lines.append("")
    lines.append("Artefatos reconhecidos para limpeza:")
    if before.known_candidates:
        for rel in _format_rel_paths(before.known_candidates):
            marker = "SELECIONADO" if any((ROOT / rel) == path for path in selected) else "IGNORADO"
            lines.append(f"- {marker}: {rel}")
    else:
        lines.append("- nenhum artefato gerado conhecido encontrado")

    if before.unexpected_files:
        lines.append("")
        lines.append("Arquivos inesperados não reconhecidos automaticamente:")
        for rel in _format_rel_paths(before.unexpected_files):
            lines.append(f"- ATENÇÃO: {rel}")

    if before.unexpected_dirs:
        lines.append("")
        lines.append("Diretórios inesperados em reports/:")
        for rel in _format_rel_paths(before.unexpected_dirs):
            lines.append(f"- ATENÇÃO: {rel}")

    if archive_path is not None:
        lines.append("")
        lines.append(f"Arquivo temporário criado em: {archive_path}")

    lines.append("")
    if after.is_clean:
        lines.append("Resultado final: reports/ limpo para revisão ou empacotamento (apenas .gitkeep preservado).")
    else:
        lines.append("Resultado final: reports/ ainda exige atenção manual antes do empacotamento oficial.")
        if after.unexpected_files:
            lines.append("- Motivo: arquivos inesperados permanecem no diretório.")
        if after.unexpected_dirs:
            lines.append("- Motivo: diretórios inesperados permanecem no diretório.")

    if summary["dry_run"]:
        lines.append("Nenhuma alteração foi aplicada porque o modo dry-run foi solicitado.")
    elif action == "none" and not selected and after.is_clean:
        lines.append("Nenhuma ação foi necessária.")
    elif action == "none" and not selected:
        lines.append("Nenhuma ação automática foi aplicada para preservar conteúdo inesperado com segurança.")

    return lines



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Limpa com segurança os artefatos transitórios gerados em reports/.")
    parser.add_argument("--dry-run", action="store_true", help="Mostra o plano de limpeza sem alterar o diretório reports/.")
    parser.add_argument("--delete", action="store_true", help="Remove definitivamente os artefatos selecionados em vez de arquivá-los fora do repositório.")
    parser.add_argument(
        "--include-unexpected",
        action="store_true",
        help="Inclui arquivos inesperados no conjunto de limpeza automática. Use apenas quando tiver certeza do conteúdo.",
    )
    return parser.parse_args(argv)



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = cleanup_reports(
        dry_run=args.dry_run,
        delete=args.delete,
        include_unexpected=args.include_unexpected,
    )
    for line in build_console_lines(summary):
        print(line)
    return 0 if bool(summary["clean_after"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
