#!/usr/bin/env python3
"""Gera um registro padronizado para a validação manual local.

Objetivo:
- reduzir atrito ao executar o CHECKLIST_REGRESSAO.md no navegador real;
- criar um arquivo markdown datado para registrar ambiente, checklist e falhas reproduzidas;
- manter o processo manual rastreável sem tocar na lógica do app.

Uso:
    python scripts/reports/manual_validation_pack.py
"""

from __future__ import annotations

from datetime import datetime
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKLIST_PATH = ROOT / "CHECKLIST_REGRESSAO.md"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"
OUTPUT_DIR = ROOT / "reports"


def latest_version() -> str:
    text = CHANGELOG_PATH.read_text(encoding="utf-8")
    match = re.search(r"^##\s+(v\d+)\s+—", text, flags=re.MULTILINE)
    return match.group(1) if match else "versao_desconhecida"


def checklist_items() -> list[str]:
    items: list[str] = []
    for raw_line in CHECKLIST_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("- [ ] "):
            items.append(line[6:].strip())
    return items


def build_report(version: str, generated_at: datetime, items: list[str]) -> str:
    timestamp_display = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append(f"# REGISTRO_VALIDACAO_MANUAL — {version}")
    lines.append("")
    lines.append("## Metadados")
    lines.append(f"- Versão da base: `{version}`")
    lines.append(f"- Gerado em: `{timestamp_display}`")
    lines.append("- Ambiente: ")
    lines.append("- Dispositivo: desktop | mobile")
    lines.append("- Navegador: ")
    lines.append("- Sistema operacional: ")
    lines.append("- Responsável pela validação: ")
    lines.append("")
    lines.append("## Pré-condições sugeridas")
    lines.append("- [ ] `python scripts/quality/runtime_preflight.py`")
    lines.append("- [ ] `python scripts/quality/quality_gate.py`")
    lines.append("- [ ] `streamlit run app.py`")
    lines.append("")
    lines.append("## Checklist manual")
    for item in items:
        lines.append(f"- [ ] {item}")
    lines.append("")
    lines.append("## Falhas reproduzidas")
    lines.append("Registrar apenas falhas reproduzíveis. Duplicar o bloco abaixo para cada falha.")
    lines.append("")
    lines.append("### Falha 1")
    lines.append("- Item do checklist: ")
    lines.append("- Passos executados: ")
    lines.append("- Observado: ")
    lines.append("- Esperado: ")
    lines.append("- Frequência: sempre | intermitente")
    lines.append("- Ambiente: desktop | mobile, navegador, sistema")
    lines.append("- Evidência: print ou descrição curta")
    lines.append("")
    lines.append("## Conclusão")
    lines.append("- Status final: aprovado | aprovado com ressalvas | reprovado")
    lines.append("- Observações finais: ")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    version = latest_version()
    now = datetime.now()
    items = checklist_items()
    timestamp_file = now.strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"validacao_manual_{version}_{timestamp_file}.md"
    output_path.write_text(build_report(version, now, items), encoding="utf-8")

    print("=== MANUAL VALIDATION PACK | Sorteador Pelada PRO ===")
    print(f"Versão detectada: {version}")
    print(f"Checklist carregado: {len(items)} itens")
    print(f"Arquivo gerado: {output_path.relative_to(ROOT)}")
    print("Próximo passo sugerido: abrir o app e preencher este relatório durante a validação manual.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
