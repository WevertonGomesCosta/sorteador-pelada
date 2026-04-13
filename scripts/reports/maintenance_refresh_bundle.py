#!/usr/bin/env python3
"""Regenera, em ordem canônica, os artefatos de manutenção já existentes.

Objetivo:
- reduzir o atrito operacional de revisão, retomada e handoff da baseline;
- executar em um único comando os utilitários somente leitura já existentes;
- manter a operação fora do núcleo funcional e sem criar novos guards.

Uso:
    python scripts/reports/maintenance_refresh_bundle.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reports import (
    maintenance_command_journal,
    maintenance_handoff_pack,
    maintenance_reports_index,
    maintenance_resume_brief,
    maintenance_snapshot_report,
)

OUTPUT_DIR = ROOT / "reports"


@dataclass(slots=True)
class RefreshStep:
    name: str
    outputs: list[Path]



def refresh_bundle() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    version = maintenance_snapshot_report.detect_version()
    steps: list[RefreshStep] = []

    snapshot_path = maintenance_snapshot_report.write_snapshot_report()
    steps.append(RefreshStep(name="maintenance_snapshot_report", outputs=[snapshot_path]))

    resume_md, resume_txt = maintenance_resume_brief.write_resume_brief()
    steps.append(RefreshStep(name="maintenance_resume_brief", outputs=[resume_md, resume_txt]))

    journal_md, journal_txt = maintenance_command_journal.write_command_journal()
    steps.append(RefreshStep(name="maintenance_command_journal", outputs=[journal_md, journal_txt]))

    handoff_path = maintenance_handoff_pack.build_handoff_pack()
    steps.append(RefreshStep(name="maintenance_handoff_pack", outputs=[handoff_path]))

    index_path = maintenance_reports_index.write_reports_index()
    steps.append(RefreshStep(name="maintenance_reports_index", outputs=[index_path]))

    outputs: list[Path] = []
    for step in steps:
        outputs.extend(step.outputs)

    return {
        "version": version,
        "steps": steps,
        "outputs": outputs,
        "cleanup_suggestion": "python scripts/reports/maintenance_reports_cleanup.py --dry-run",
    }



def build_console_lines(summary: dict[str, object]) -> list[str]:
    version = str(summary["version"])
    steps: list[RefreshStep] = summary["steps"]  # type: ignore[assignment]
    outputs: list[Path] = summary["outputs"]  # type: ignore[assignment]
    cleanup_suggestion = str(summary["cleanup_suggestion"])

    lines: list[str] = []
    lines.append("=== MAINTENANCE REFRESH BUNDLE | Sorteador Pelada PRO ===")
    lines.append(f"Versão detectada: {version}")
    lines.append("Modo: operação local, somente leitura")
    lines.append("Objetivo: regenerar em ordem canônica os artefatos de manutenção já existentes.")
    lines.append("")
    lines.append("Ordem executada:")
    for index, step in enumerate(steps, start=1):
        lines.append(f"{index}. {step.name}")
        for output in step.outputs:
            lines.append(f"   - {output.relative_to(ROOT)}")
    lines.append("")
    lines.append(f"Total de artefatos gerados: {len(outputs)}")
    lines.append(f"Próximo comando sugerido: {cleanup_suggestion}")
    lines.append("Observação: o cleanup não é executado automaticamente por este bundle.")
    return lines



def main() -> int:
    summary = refresh_bundle()
    for line in build_console_lines(summary):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
