#!/usr/bin/env python3
"""Gera um roteiro operacional curto e canônico de comandos da baseline.

Objetivo:
- consolidar em ordem prática os comandos operacionais essenciais da baseline;
- reduzir atrito de revisão, retomada e handoff local;
- manter um artefato somente leitura, sem executar comandos nem alterar o repositório.

Uso:
    python scripts/reports/maintenance_command_journal.py
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

SCENARIOS: list[tuple[str, str, list[str]]] = [
    (
        "Inspeção rápida",
        "Quando precisar entender rapidamente o estado da baseline sem empacotar nada.",
        [
            "python scripts/quality/runtime_preflight.py",
            "python scripts/reports/maintenance_snapshot_report.py",
            "python scripts/reports/maintenance_command_journal.py",
        ],
    ),
    (
        "Retomada curta",
        "Quando for voltar ao projeto após pausa ou abrir um novo chat técnico.",
        [
            "python scripts/reports/maintenance_resume_brief.py",
            "python scripts/reports/maintenance_command_journal.py",
        ],
    ),
    (
        "Revisão e handoff",
        "Quando for reunir referências operacionais para revisão interna ou transferência técnica.",
        [
            "python scripts/reports/maintenance_snapshot_report.py",
            "python scripts/reports/maintenance_resume_brief.py",
            "python scripts/reports/maintenance_command_journal.py",
            "python scripts/reports/maintenance_handoff_pack.py",
        ],
    ),
    (
        "Validação local mínima",
        "Quando precisar confirmar rapidamente a sanidade estrutural da baseline antes de seguir.",
        [
            "python scripts/quality/check_base.py",
            "python scripts/quality/release_guard.py",
        ],
    ),
    (
        "Fechamento operacional de reports/",
        "Quando terminar a geração de artefatos locais e for limpar a baseline antes de empacotar.",
        [
            "python scripts/reports/maintenance_reports_cleanup.py",
        ],
    ),
]

PRACTICAL_ORDER = [
    "python scripts/quality/runtime_preflight.py",
    "python scripts/reports/maintenance_snapshot_report.py",
    "python scripts/reports/maintenance_resume_brief.py",
    "python scripts/reports/maintenance_command_journal.py",
    "python scripts/reports/maintenance_handoff_pack.py",
    "python scripts/quality/check_base.py",
    "python scripts/quality/release_guard.py",
    "python scripts/reports/maintenance_reports_cleanup.py",
]


def build_markdown(version: str, generated_at: datetime) -> str:
    timestamp_display = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append(f"# MAINTENANCE_COMMAND_JOURNAL — {version}")
    lines.append("")
    lines.append("## Metadados")
    lines.append(f"- Versão detectada: `{version}`")
    lines.append(f"- Gerado em: `{timestamp_display}`")
    lines.append("- Modo: somente leitura")
    lines.append("- Objetivo: consolidar a ordem prática dos comandos operacionais essenciais da baseline.")
    lines.append("")
    lines.append("## Ordem prática sugerida")
    for index, command in enumerate(PRACTICAL_ORDER, start=1):
        lines.append(f"{index}. `{command}`")
    lines.append("")
    lines.append("## Cenários de uso")
    for title, description, commands in SCENARIOS:
        lines.append(f"### {title}")
        lines.append(f"- Quando usar: {description}")
        lines.append("- Comandos:")
        for command in commands:
            lines.append(f"  - `{command}`")
        lines.append("")
    lines.append("## Observações")
    lines.append("- Este journal não executa comandos e não substitui a validação real da baseline.")
    lines.append("- Antes do `.zip` final, execute `python scripts/reports/maintenance_reports_cleanup.py`.")
    return "\n".join(lines) + "\n"



def build_plain_text(version: str, generated_at: datetime) -> str:
    timestamp_display = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append(f"MAINTENANCE_COMMAND_JOURNAL — {version}")
    lines.append("")
    lines.append(f"Gerado em: {timestamp_display}")
    lines.append("Modo: somente leitura")
    lines.append("Objetivo: consolidar a ordem prática dos comandos operacionais essenciais da baseline.")
    lines.append("")
    lines.append("Ordem prática sugerida:")
    for index, command in enumerate(PRACTICAL_ORDER, start=1):
        lines.append(f"{index}. {command}")
    lines.append("")
    lines.append("Cenários de uso:")
    for title, description, commands in SCENARIOS:
        lines.append(f"- {title}: {description}")
        for command in commands:
            lines.append(f"  * {command}")
    lines.append("")
    lines.append("Observações:")
    lines.append("- Este journal não executa comandos e não substitui a validação real da baseline.")
    lines.append("- Antes do .zip final, execute python scripts/reports/maintenance_reports_cleanup.py.")
    return "\n".join(lines) + "\n"



def write_command_journal() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    version = maintenance_snapshot_report.detect_version()
    generated_at = datetime.now()
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
    md_path = OUTPUT_DIR / f"maintenance_command_journal_{version}_{timestamp}.md"
    txt_path = OUTPUT_DIR / f"maintenance_command_journal_{version}_{timestamp}.txt"
    md_path.write_text(build_markdown(version, generated_at), encoding="utf-8")
    txt_path.write_text(build_plain_text(version, generated_at), encoding="utf-8")
    return md_path, txt_path



def main() -> int:
    md_path, txt_path = write_command_journal()
    print(f"Journal operacional markdown gerado em: {md_path}")
    print(f"Journal operacional texto gerado em: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
