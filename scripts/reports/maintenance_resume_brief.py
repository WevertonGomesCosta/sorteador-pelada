#!/usr/bin/env python3
"""Gera um resumo operacional curto e pronto para retomada da baseline.

Objetivo:
- produzir um artefato curto e reutilizável para retomada do projeto;
- reduzir custo cognitivo de revisão e handoff técnico;
- consolidar em modo somente leitura as restrições, comandos e referências da baseline.

Uso:
    python scripts/reports/maintenance_resume_brief.py
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
FROZEN_SCOPE_ITEMS = [
    "app.py",
    "ui/review_view.py",
    "confirmação/sorteio",
    "lógica central do app",
]
MAINTENANCE_UTILITIES = [
    "python scripts/quality/runtime_preflight.py",
    "python scripts/quality/quality_gate.py",
    "python scripts/reports/release_health_report.py",
    "python scripts/reports/maintenance_snapshot_report.py",
    "python scripts/reports/maintenance_handoff_pack.py",
    "python scripts/reports/maintenance_resume_brief.py",
    "python scripts/reports/maintenance_command_journal.py",
    "python scripts/reports/maintenance_reports_cleanup.py",
    "python scripts/reports/maintenance_refresh_bundle.py",
]
CANONICAL_REFERENCES = [
    "README.md",
    "CHANGELOG.md",
    "CHECKLIST_REGRESSAO.md",
    "docs/releases/BASELINE_OFICIAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md",
    "docs/validation/VALIDACAO_MANUAL_GUIA.md",
    "scripts/quality/checks_registry.py",
    "scripts/reports/maintenance_snapshot_report.py",
    "scripts/reports/maintenance_handoff_pack.py",
    "scripts/reports/maintenance_resume_brief.py",
    "scripts/reports/maintenance_command_journal.py",
    "scripts/reports/maintenance_reports_cleanup.py",
    "scripts/reports/maintenance_refresh_bundle.py",
]


def build_markdown(version: str, generated_at: datetime) -> str:
    timestamp_display = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append(f"# MAINTENANCE_RESUME_BRIEF — {version}")
    lines.append("")
    lines.append("## Estado atual")
    lines.append(f"- Baseline ativa: `{version}`")
    lines.append(f"- Gerado em: `{timestamp_display}`")
    lines.append("- Modo: somente leitura")
    lines.append("- Situação: frente estrutural leve encerrada; base em manutenção pontual fora do núcleo funcional.")
    lines.append("")
    lines.append("## Escopo congelado")
    for item in FROZEN_SCOPE_ITEMS:
        lines.append(f"- `{item}`")
    lines.append("")
    lines.append("## Restrições obrigatórias")
    lines.append("- não reabrir arquitetura ampla")
    lines.append("- não criar novos guards sem evidência operacional concreta")
    lines.append("- não alterar contratos de compatibilidade temporária")
    lines.append("- executar `python scripts/reports/maintenance_refresh_bundle.py` quando quiser regenerar o conjunto de artefatos em ordem canônica")
    lines.append("- executar `python scripts/reports/maintenance_reports_cleanup.py` antes de empacotar a baseline oficial")
    lines.append("")
    lines.append("## Utilitários operacionais disponíveis")
    for command in MAINTENANCE_UTILITIES:
        lines.append(f"- `{command}`")
    lines.append("")
    lines.append("## Referências canônicas para retomada")
    for rel_path in CANONICAL_REFERENCES:
        lines.append(f"- `{rel_path}`")
    lines.append("")
    lines.append("## Próxima ação segura")
    lines.append("- escolher apenas uma micro-etapa fora do núcleo funcional com ganho direto para uso, revisão ou retomada")
    lines.append("")
    lines.append("## Prompt pronto para novo chat")
    lines.append(f"> Usando a baseline {version} como base estável, continue a partir do estado atual apenas com manutenção pontual fora do núcleo funcional, sem criar novos guards e sem alterar contratos de compatibilidade temporária.")
    return "\n".join(lines) + "\n"



def build_plain_text(version: str, generated_at: datetime) -> str:
    timestamp_display = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append(f"MAINTENANCE_RESUME_BRIEF — {version}")
    lines.append("")
    lines.append(f"Gerado em: {timestamp_display}")
    lines.append("Modo: somente leitura")
    lines.append("Estado: frente estrutural leve encerrada; baseline em manutenção pontual fora do núcleo funcional.")
    lines.append("")
    lines.append("Escopo congelado:")
    for item in FROZEN_SCOPE_ITEMS:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Restrições:")
    lines.append("- não reabrir arquitetura ampla")
    lines.append("- não criar novos guards sem evidência concreta")
    lines.append("- não alterar contratos de compatibilidade temporária")
    lines.append("- executar python scripts/reports/maintenance_refresh_bundle.py quando quiser regenerar o conjunto de artefatos em ordem canônica")
    lines.append("- executar python scripts/reports/maintenance_reports_cleanup.py antes de empacotar a baseline oficial")
    lines.append("")
    lines.append("Comandos úteis:")
    for command in MAINTENANCE_UTILITIES:
        lines.append(f"- {command}")
    lines.append("")
    lines.append("Próxima ação segura:")
    lines.append("- escolher uma micro-etapa fora do núcleo funcional com ganho direto para revisão, uso ou retomada")
    lines.append("")
    lines.append("Prompt de continuidade:")
    lines.append(
        f"Usando a baseline {version} como base estável, continue apenas com manutenção pontual fora do núcleo funcional, sem criar novos guards e sem alterar contratos de compatibilidade temporária."
    )
    return "\n".join(lines) + "\n"



def write_resume_brief() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    version = maintenance_snapshot_report.detect_version()
    generated_at = datetime.now()
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
    md_path = OUTPUT_DIR / f"maintenance_resume_brief_{version}_{timestamp}.md"
    txt_path = OUTPUT_DIR / f"maintenance_resume_brief_{version}_{timestamp}.txt"
    md_path.write_text(build_markdown(version, generated_at), encoding="utf-8")
    txt_path.write_text(build_plain_text(version, generated_at), encoding="utf-8")
    return md_path, txt_path



def main() -> int:
    md_path, txt_path = write_resume_brief()
    print(f"Resumo operacional markdown gerado em: {md_path}")
    print(f"Resumo operacional texto gerado em: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
