#!/usr/bin/env python3
"""Gera um índice canônico dos artefatos operacionais mais recentes em reports/.

Objetivo:
- localizar rapidamente os artefatos mais recentes gerados pelos utilitários de manutenção;
- reduzir atrito de revisão, uso e retomada quando houver múltiplos arquivos com timestamp;
- manter a operação em modo somente leitura, sem criar novos guards.

Uso:
    python scripts/reports/maintenance_reports_index.py
"""

from __future__ import annotations

from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reports import maintenance_snapshot_report

OUTPUT_DIR = ROOT / "reports"
INDEX_FILENAME = "MAINTENANCE_REPORTS_INDEX.md"
ARTIFACT_SPECS = [
    {
        "label": "Snapshot operacional",
        "glob": "maintenance_snapshot_*.md",
        "regen": "python scripts/reports/maintenance_snapshot_report.py",
        "reading_order": 1,
    },
    {
        "label": "Resumo curto de retomada (markdown)",
        "glob": "maintenance_resume_brief_*.md",
        "regen": "python scripts/reports/maintenance_resume_brief.py",
        "reading_order": 2,
    },
    {
        "label": "Resumo curto de retomada (texto)",
        "glob": "maintenance_resume_brief_*.txt",
        "regen": "python scripts/reports/maintenance_resume_brief.py",
        "reading_order": 3,
    },
    {
        "label": "Journal de comandos (markdown)",
        "glob": "maintenance_command_journal_*.md",
        "regen": "python scripts/reports/maintenance_command_journal.py",
        "reading_order": 4,
    },
    {
        "label": "Journal de comandos (texto)",
        "glob": "maintenance_command_journal_*.txt",
        "regen": "python scripts/reports/maintenance_command_journal.py",
        "reading_order": 5,
    },
    {
        "label": "Handoff pack (.zip)",
        "glob": "maintenance_handoff_*.zip",
        "regen": "python scripts/reports/maintenance_handoff_pack.py",
        "reading_order": 6,
    },
]


def latest_match(glob_pattern: str) -> Path | None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [path for path in OUTPUT_DIR.glob(glob_pattern) if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))



def build_report(version: str, generated_at: datetime) -> str:
    timestamp_display = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append(f"# MAINTENANCE_REPORTS_INDEX — {version}")
    lines.append("")
    lines.append("## Metadados")
    lines.append(f"- Versão detectada: `{version}`")
    lines.append(f"- Gerado em: `{timestamp_display}`")
    lines.append("- Local fixo: `reports/MAINTENANCE_REPORTS_INDEX.md`")
    lines.append("- Modo: somente leitura")
    lines.append("")
    lines.append("## Artefatos operacionais mais recentes")
    for spec in ARTIFACT_SPECS:
        path = latest_match(str(spec["glob"]))
        if path is None:
            lines.append(f"- `{spec['label']}` — AUSENTE — regenerar com `{spec['regen']}`")
        else:
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(
                f"- `{spec['label']}` — `reports/{path.name}` — atualizado em `{mtime}` — regenerar com `{spec['regen']}`"
            )
    lines.append("")
    lines.append("## Ordem sugerida de leitura")
    lines.append("1. `reports/MAINTENANCE_REPORTS_INDEX.md`")
    for spec in sorted(ARTIFACT_SPECS, key=lambda item: int(item["reading_order"])):
        path = latest_match(str(spec["glob"]))
        if path is None:
            lines.append(f"{int(spec['reading_order']) + 1}. `{spec['label']}` — ausente")
        else:
            lines.append(f"{int(spec['reading_order']) + 1}. `reports/{path.name}`")
    lines.append("")
    lines.append("## Comandos relacionados")
    lines.append("- `python scripts/reports/maintenance_refresh_bundle.py`")
    lines.append("- `python scripts/reports/maintenance_reports_index.py`")
    lines.append("- `python scripts/reports/maintenance_reports_cleanup.py --dry-run`")
    lines.append("")
    lines.append("## Observações")
    lines.append("- Este índice não executa regeneração automática nem substitui a validação real da baseline.")
    lines.append("- Para atualizar em lote os artefatos principais, execute `python scripts/reports/maintenance_refresh_bundle.py` e depois regenere este índice.")
    lines.append("- Antes do empacotamento final, execute `python scripts/reports/maintenance_reports_cleanup.py` para higienizar `reports/` com segurança.")
    return "\n".join(lines) + "\n"



def write_reports_index() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    version = maintenance_snapshot_report.detect_version()
    generated_at = datetime.now()
    output_path = OUTPUT_DIR / INDEX_FILENAME
    output_path.write_text(build_report(version, generated_at), encoding="utf-8")
    return output_path



def main() -> int:
    output_path = write_reports_index()
    print(f"Índice canônico de reports gerado em: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
