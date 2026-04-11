#!/usr/bin/env python3
"""Gera um snapshot operacional somente leitura da baseline.

Objetivo:
- consolidar em um único arquivo uma leitura rápida da baseline ativa;
- reduzir atrito de revisão e handoff sem executar a rotina composta de checks;
- manter um diagnóstico estático e reproduzível fora do núcleo funcional do app.

Uso:
    python scripts/reports/maintenance_snapshot_report.py
"""

from __future__ import annotations

from datetime import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality import compatibility_contract_guard, release_metadata_guard
from scripts.quality.checks_registry import registry_entries

OUTPUT_DIR = ROOT / "reports"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"
BASELINE_PATH = ROOT / "docs" / "releases" / "BASELINE_OFICIAL.md"
PRIMITIVES_PATH = ROOT / "ui" / "primitives.py"
PROTECTED_SCOPE_PATH = ROOT / "docs" / "releases" / "PROTECTED_SCOPE_HASHES.json"
CANONICAL_DOCS = [
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
    "scripts/reports/maintenance_refresh_bundle.py",
]
RECOMMENDED_COMMANDS = [
    "python scripts/quality/runtime_preflight.py",
    "python scripts/quality/quality_gate.py",
    "python scripts/quality/release_guard.py",
    "python scripts/reports/maintenance_snapshot_report.py",
    "python scripts/reports/maintenance_handoff_pack.py",
    "python scripts/reports/maintenance_resume_brief.py",
    "python scripts/reports/maintenance_command_journal.py",
    "python scripts/reports/maintenance_reports_cleanup.py",
    "python scripts/reports/maintenance_refresh_bundle.py",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def detect_version() -> str:
    footer = release_metadata_guard.extract_footer_version(read_text(PRIMITIVES_PATH)) if PRIMITIVES_PATH.exists() else None
    changelog = release_metadata_guard.extract_changelog_version(read_text(CHANGELOG_PATH)) if CHANGELOG_PATH.exists() else None
    baseline = release_metadata_guard.extract_baseline_version(read_text(BASELINE_PATH)) if BASELINE_PATH.exists() else None
    for value in (footer, changelog, baseline):
        if value:
            return value
    return "versao_desconhecida"


def registry_summary() -> dict[str, object]:
    entries = registry_entries()
    categories = sorted({str(item["category"]) for item in entries})
    return {
        "entries": entries,
        "categories": categories,
    }


def wrappers_status() -> list[tuple[str, str, bool]]:
    rows: list[tuple[str, str, bool]] = []
    for rel_path, spec in compatibility_contract_guard.WRAPPER_EXPECTATIONS.items():
        ok = (ROOT / rel_path).exists() and (ROOT / spec["canonical_path"]).exists()
        rows.append((rel_path, spec["canonical_path"], ok))
    return rows


def doc_bridges_status() -> list[tuple[str, str, bool]]:
    rows: list[tuple[str, str, bool]] = []
    for rel_path, canonical in compatibility_contract_guard.DOC_BRIDGE_EXPECTATIONS.items():
        ok = (ROOT / rel_path).exists() and (ROOT / canonical).exists()
        rows.append((rel_path, canonical, ok))
    return rows


def protected_scope_summary() -> tuple[int, list[str]]:
    if not PROTECTED_SCOPE_PATH.exists():
        return 0, []
    data = json.loads(PROTECTED_SCOPE_PATH.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = sorted(str(key) for key in data.keys())
    elif isinstance(data, list):
        items = sorted(str(item) for item in data)
    else:
        items = [str(type(data).__name__)]
    return len(items), items


def reports_dir_state() -> list[str]:
    if not OUTPUT_DIR.exists():
        return ["<ausente>"]
    return sorted(path.name for path in OUTPUT_DIR.iterdir())


def build_report(version: str, generated_at: datetime) -> str:
    timestamp_display = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    registry = registry_summary()
    wrappers = wrappers_status()
    doc_bridges = doc_bridges_status()
    protected_count, protected_items = protected_scope_summary()
    reports_state = reports_dir_state()

    lines: list[str] = []
    lines.append(f"# MAINTENANCE_SNAPSHOT_REPORT — {version}")
    lines.append("")
    lines.append("## Metadados")
    lines.append(f"- Versão detectada: `{version}`")
    lines.append(f"- Gerado em: `{timestamp_display}`")
    lines.append("- Modo: somente leitura")
    lines.append("- Fonte do versionamento: `ui/primitives.py`, `CHANGELOG.md` e `docs/releases/BASELINE_OFICIAL.md`")
    lines.append("")
    lines.append("## Resumo executivo")
    lines.append(f"- Checks canônicos cadastrados: `{len(registry['entries'])}`")
    lines.append(f"- Categorias cobertas: `{len(registry['categories'])}`")
    lines.append(f"- Wrappers históricos preservados: `{len(wrappers)}`")
    lines.append(f"- Arquivos-ponte preservados: `{len(doc_bridges)}`")
    lines.append(f"- Itens protegidos no manifesto de hashes: `{protected_count}`")
    lines.append("- Observação: este snapshot não executa `quality_gate` nem substitui a validação manual.")
    lines.append("")
    lines.append("## Checks oficiais cadastrados")
    for spec in registry["entries"]:
        lines.append(
            f"- `{spec['name']}` — categoria=`{spec['category']}` — kind=`{spec['kind']}` — quality_gate=`{bool(spec['enabled_in_quality_gate'])}` — release_health_report=`{bool(spec['enabled_in_release_health_report'])}`"
        )
    lines.append("")
    lines.append("## Compatibilidade temporária")
    lines.append("### Wrappers históricos")
    for rel_path, canonical, ok in wrappers:
        status = "OK" if ok else "PENDENTE"
        lines.append(f"- `{status}` — `{rel_path}` -> `{canonical}`")
    lines.append("")
    lines.append("### Arquivos-ponte em docs/")
    for rel_path, canonical, ok in doc_bridges:
        status = "OK" if ok else "PENDENTE"
        lines.append(f"- `{status}` — `{rel_path}` -> `{canonical}`")
    lines.append("")
    lines.append("## Escopo protegido")
    if protected_items:
        for item in protected_items:
            lines.append(f"- `{item}`")
    else:
        lines.append("- Manifesto indisponível ou vazio.")
    lines.append("")
    lines.append("## Documentos e referências canônicas")
    for rel_path in CANONICAL_DOCS:
        status = "OK" if (ROOT / rel_path).exists() else "AUSENTE"
        lines.append(f"- `{status}` — `{rel_path}`")
    lines.append("")
    lines.append("## Estado atual de reports/")
    for entry in reports_state:
        lines.append(f"- `{entry}`")
    lines.append("")
    lines.append("## Comandos recomendados")
    for command in RECOMMENDED_COMMANDS:
        lines.append(f"- `{command}`")
    lines.append("")
    lines.append("## Fechamento")
    lines.append("- Este artefato é voltado a triagem, revisão rápida e handoff técnico.")
    lines.append("- Para regenerar em um único comando os artefatos operacionais principais, execute `python scripts/reports/maintenance_refresh_bundle.py`.")
    lines.append("- Antes de empacotar a baseline oficial, execute `python scripts/reports/maintenance_reports_cleanup.py` para fazer a higiene segura de `reports/` e voltar a conter apenas `.gitkeep`.")
    return "\n".join(lines) + "\n"


def write_snapshot_report() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    version = detect_version()
    generated_at = datetime.now()
    filename = f"maintenance_snapshot_{version}_{generated_at.strftime('%Y%m%d_%H%M%S')}.md"
    output_path = OUTPUT_DIR / filename
    output_path.write_text(build_report(version, generated_at), encoding="utf-8")
    return output_path


def main() -> int:
    output_path = write_snapshot_report()
    print(f"Snapshot gerado em: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
