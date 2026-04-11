#!/usr/bin/env python3
"""Gera um pacote único de handoff operacional somente leitura.

Objetivo:
- consolidar em um único artefato os principais relatórios e referências operacionais da baseline;
- facilitar revisão, auditoria manual e retomada futura sem tocar no núcleo funcional;
- manter o handoff baseado apenas em leitura e cópia de artefatos canônicos.

Uso:
    python scripts/reports/maintenance_handoff_pack.py
"""

from __future__ import annotations

from datetime import datetime
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reports import maintenance_command_journal, maintenance_resume_brief, maintenance_snapshot_report

OUTPUT_DIR = ROOT / "reports"
INCLUDED_REFERENCE_FILES = [
    "README.md",
    "CHANGELOG.md",
    "CHECKLIST_REGRESSAO.md",
    "docs/releases/BASELINE_OFICIAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md",
    "docs/validation/VALIDACAO_MANUAL_GUIA.md",
    "docs/releases/PROTECTED_SCOPE_HASHES.json",
    "scripts/quality/checks_registry.py",
    "scripts/reports/manual_validation_pack.py",
    "scripts/reports/release_health_report.py",
    "scripts/reports/maintenance_snapshot_report.py",
    "scripts/reports/maintenance_handoff_pack.py",
    "scripts/reports/maintenance_resume_brief.py",
    "scripts/reports/maintenance_command_journal.py",
    "scripts/reports/maintenance_reports_cleanup.py",
]
OPTIONAL_REPORT_EXTENSIONS = {".md", ".txt", ".json"}


def collect_local_reports() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    collected: list[Path] = []
    for path in sorted(OUTPUT_DIR.iterdir()):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        if path.suffix.lower() not in OPTIONAL_REPORT_EXTENSIONS:
            continue
        if path.name.startswith("maintenance_handoff_"):
            continue
        collected.append(path)
    return collected


def build_index(version: str, generated_at: datetime, included_reports: list[Path]) -> str:
    timestamp_display = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append(f"# MAINTENANCE_HANDOFF_PACK — {version}")
    lines.append("")
    lines.append("## Metadados")
    lines.append(f"- Versão detectada: `{version}`")
    lines.append(f"- Gerado em: `{timestamp_display}`")
    lines.append("- Modo: somente leitura")
    lines.append("- Estrutura: índice + snapshot + cópias das referências canônicas")
    lines.append("")
    lines.append("## Conteúdo incluído")
    lines.append("- `00_INDEX.md`")
    lines.append("- `01_MAINTENANCE_SNAPSHOT.md`")
    lines.append("- `02_MAINTENANCE_RESUME_BRIEF.md`")
    lines.append("- `03_MAINTENANCE_RESUME_BRIEF.txt`")
    lines.append("- `04_MAINTENANCE_COMMAND_JOURNAL.md`")
    lines.append("- `05_MAINTENANCE_COMMAND_JOURNAL.txt`")
    for rel_path in INCLUDED_REFERENCE_FILES:
        lines.append(f"- `files/{rel_path}`")
    if included_reports:
        for report in included_reports:
            lines.append(f"- `local_reports/{report.name}`")
    else:
        lines.append("- Nenhum relatório local adicional foi encontrado em `reports/`.")
    lines.append("")
    lines.append("## Uso sugerido")
    lines.append("- abrir primeiro `00_INDEX.md`")
    lines.append("- revisar `01_MAINTENANCE_SNAPSHOT.md`")
    lines.append("- usar `02_MAINTENANCE_RESUME_BRIEF.md` ou `03_MAINTENANCE_RESUME_BRIEF.txt` para retomada curta")
    lines.append("- usar `04_MAINTENANCE_COMMAND_JOURNAL.md` ou `05_MAINTENANCE_COMMAND_JOURNAL.txt` para consultar a ordem prática dos comandos")
    lines.append("- consultar as cópias de `docs/`, `scripts/` e arquivos-raiz incluídas em `files/`")
    lines.append("")
    lines.append("## Comandos canônicos relacionados")
    lines.append("- `python scripts/reports/maintenance_snapshot_report.py`")
    lines.append("- `python scripts/reports/maintenance_handoff_pack.py`")
    lines.append("- `python scripts/reports/maintenance_resume_brief.py`")
    lines.append("- `python scripts/reports/maintenance_command_journal.py`")
    lines.append("- `python scripts/reports/maintenance_reports_cleanup.py`")
    lines.append("- `python scripts/reports/release_health_report.py`")
    lines.append("- `python scripts/reports/manual_validation_pack.py`")
    lines.append("")
    lines.append("## Observações")
    lines.append("- Este pacote não altera o repositório nem executa guards compostos.")
    lines.append("- Antes de empacotar a baseline oficial, execute `python scripts/reports/maintenance_reports_cleanup.py` para higienizar `reports/` e preservar apenas `.gitkeep`.")
    return "\n".join(lines) + "\n"


def build_handoff_pack() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    version = maintenance_snapshot_report.detect_version()
    generated_at = datetime.now()
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
    output_base = OUTPUT_DIR / f"maintenance_handoff_{version}_{timestamp}"
    local_reports = collect_local_reports()

    with tempfile.TemporaryDirectory(prefix="maintenance_handoff_pack_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        (temp_dir / "files").mkdir(parents=True, exist_ok=True)
        if local_reports:
            (temp_dir / "local_reports").mkdir(parents=True, exist_ok=True)

        snapshot_text = maintenance_snapshot_report.build_report(version, generated_at)
        resume_md = maintenance_resume_brief.build_markdown(version, generated_at)
        resume_txt = maintenance_resume_brief.build_plain_text(version, generated_at)
        journal_md = maintenance_command_journal.build_markdown(version, generated_at)
        journal_txt = maintenance_command_journal.build_plain_text(version, generated_at)
        (temp_dir / "01_MAINTENANCE_SNAPSHOT.md").write_text(snapshot_text, encoding="utf-8")
        (temp_dir / "02_MAINTENANCE_RESUME_BRIEF.md").write_text(resume_md, encoding="utf-8")
        (temp_dir / "03_MAINTENANCE_RESUME_BRIEF.txt").write_text(resume_txt, encoding="utf-8")
        (temp_dir / "04_MAINTENANCE_COMMAND_JOURNAL.md").write_text(journal_md, encoding="utf-8")
        (temp_dir / "05_MAINTENANCE_COMMAND_JOURNAL.txt").write_text(journal_txt, encoding="utf-8")
        (temp_dir / "00_INDEX.md").write_text(build_index(version, generated_at, local_reports), encoding="utf-8")

        for rel_path in INCLUDED_REFERENCE_FILES:
            source = ROOT / rel_path
            if not source.exists():
                continue
            target = temp_dir / "files" / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        for report in local_reports:
            shutil.copy2(report, temp_dir / "local_reports" / report.name)

        archive_path = shutil.make_archive(str(output_base), "zip", root_dir=temp_dir)
    return Path(archive_path)


def main() -> int:
    output_path = build_handoff_pack()
    print(f"Pacote de handoff gerado em: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
