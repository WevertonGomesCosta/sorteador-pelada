#!/usr/bin/env python3
"""Gera um relatório padronizado da saúde operacional da release.

Objetivo:
- consolidar em um único arquivo a evidência operacional dos checks e guards da release;
- reduzir dispersão entre múltiplas saídas de console;
- manter rastreabilidade leve da baseline sem tocar no núcleo funcional do app.

Uso:
    python scripts/reports/release_health_report.py
"""

from __future__ import annotations

from datetime import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality import compatibility_contract_guard, release_metadata_guard
from scripts.quality.checks_registry import (
    CHECK_TIMEOUT_OVERRIDES,
    DEFAULT_CHECK_TIMEOUT_SECONDS,
    build_check_commands,
)

OUTPUT_DIR = ROOT / "reports"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"
BASELINE_PATH = ROOT / "docs" / "releases" / "BASELINE_OFICIAL.md"
PRIMITIVES_PATH = ROOT / "ui" / "primitives.py"

CHECKS: list[tuple[str, list[str]]] = build_check_commands(ROOT, sys.executable, target="release_health_report")

CANONICAL_PATHS = [
    "scripts/quality/checks_registry.py",
    "scripts/quality/checks_registry_contract_guard.py",
    "scripts/quality/checks_registry_schema_guard.py",
    "scripts/quality/checks_registry_consumers_guard.py",
    "scripts/quality/quality_gate_composition_guard.py",
    "scripts/quality/check_base.py",
    "scripts/validation/smoke_test_base.py",
    "scripts/quality/release_metadata_guard.py",
    "scripts/quality/compatibility_contract_guard.py",
    "scripts/quality/operational_checks_contract_guard.py",
    "scripts/quality/canonical_paths_reference_guard.py",
    "scripts/quality/script_cli_contract_guard.py",
    "scripts/quality/release_artifacts_hygiene_guard.py",
    "scripts/quality/runtime_dependencies_contract_guard.py",
    "scripts/quality/documentation_commands_examples_guard.py",
    "scripts/quality/release_manifest_guard.py",
    "scripts/quality/quality_runtime_budget_guard.py",
    "scripts/quality/script_exit_codes_contract_guard.py",
    "scripts/quality/governance_docs_crosslinks_guard.py",
    "scripts/quality/protected_scope_hash_guard.py",
    "scripts/quality/release_guard.py",
    "scripts/quality/quality_gate.py",
    "scripts/reports/manual_validation_pack.py",
    "scripts/reports/release_health_report.py",
    "scripts/reports/maintenance_snapshot_report.py",
    "scripts/reports/maintenance_handoff_pack.py",
    "scripts/reports/maintenance_resume_brief.py",
    "scripts/reports/maintenance_command_journal.py",
    "scripts/reports/maintenance_reports_cleanup.py",
    "scripts/reports/maintenance_refresh_bundle.py",
    "docs/releases/BASELINE_OFICIAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
    "docs/releases/PROTECTED_SCOPE_HASHES.json",
    "docs/operations/OPERACAO_LOCAL.md",
    "scripts/quality/runtime_preflight.py",
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


def run_check(name: str, command: list[str]) -> dict[str, object]:
    timeout_seconds = CHECK_TIMEOUT_OVERRIDES.get(name, DEFAULT_CHECK_TIMEOUT_SECONDS)
    try:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout_seconds)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        combined = stdout + (("\n" + stderr) if stderr else "")
        excerpt_lines = [line.rstrip() for line in combined.strip().splitlines()[:20] if line.strip()]
        return {
            "name": name,
            "command": " ".join(command),
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "excerpt": excerpt_lines,
            "timeout_seconds": timeout_seconds,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        combined = stdout + (("\n" + stderr) if stderr else "")
        excerpt_lines = [line.rstrip() for line in combined.strip().splitlines()[:20] if line.strip()]
        return {
            "name": name,
            "command": " ".join(command),
            "returncode": 124,
            "ok": False,
            "excerpt": excerpt_lines or ["TIMEOUT"],
            "timeout_seconds": timeout_seconds,
        }


def inventory_summary() -> dict[str, int]:
    wrappers = len(compatibility_contract_guard.WRAPPER_EXPECTATIONS)
    doc_bridges = len(compatibility_contract_guard.DOC_BRIDGE_EXPECTATIONS)
    aggregator = 1 if (ROOT / compatibility_contract_guard.AGGREGATOR_PATH).exists() else 0
    return {
        "wrappers": wrappers,
        "doc_bridges": doc_bridges,
        "compatibility_aggregators": aggregator,
    }


def build_report(version: str, generated_at: datetime, results: list[dict[str, object]]) -> str:
    timestamp_display = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    inventory = inventory_summary()
    ok_count = sum(1 for item in results if item["ok"])
    fail_count = len(results) - ok_count
    lines: list[str] = []
    lines.append(f"# RELEASE_HEALTH_REPORT — {version}")
    lines.append("")
    lines.append("## Metadados")
    lines.append(f"- Versão detectada: `{version}`")
    lines.append(f"- Gerado em: `{timestamp_display}`")
    lines.append("- Baseline oficial: `docs/releases/BASELINE_OFICIAL.md`")
    lines.append("- Changelog: `CHANGELOG.md`")
    lines.append("- Rodapé: `ui/primitives.py`")
    lines.append("")
    lines.append("## Resumo executivo")
    lines.append(f"- Checks executados: `{len(results)}`")
    lines.append(f"- Checks aprovados: `{ok_count}`")
    lines.append(f"- Checks com falha: `{fail_count}`")
    lines.append("- Composite gate complementar: `python scripts/quality/quality_gate.py`")
    lines.append("- Fonte única de verdade dos checks: `scripts/quality/checks_registry.py`")
    lines.append("- Schema canônico do registro: `scripts/quality/checks_registry_schema_guard.py`")
    lines.append("- Composição determinística do quality_gate: `scripts/quality/quality_gate_composition_guard.py`")
    lines.append("")
    lines.append("## Status dos checks")
    for item in results:
        status = "OK" if item["ok"] else "FALHOU"
        lines.append(
            f"- **{item['name']}** — `{status}` — `timeout={item.get('timeout_seconds', 'n/d')}s` — `{item['command']}`"
        )
        for excerpt in item["excerpt"]:
            lines.append(f"  - {excerpt}")
    lines.append("")
    lines.append("## Inventário resumido de compatibilidade temporária")
    lines.append(f"- Wrappers históricos: `{inventory['wrappers']}`")
    lines.append(f"- Arquivos-ponte em docs/: `{inventory['doc_bridges']}`")
    lines.append(f"- Agregadores compatíveis: `{inventory['compatibility_aggregators']}`")
    lines.append("")
    lines.append("## Caminhos canônicos principais")
    for path in CANONICAL_PATHS:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Fechamento")
    if fail_count:
        lines.append("- Resultado geral: **pendente de correção**")
    else:
        lines.append("- Resultado geral: **aprovado tecnicamente**")
    lines.append("- Observação: este relatório é complementar e não substitui a validação manual no navegador.")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    version = detect_version()
    generated_at = datetime.now()
    results = [run_check(name, command) for name, command in CHECKS]
    report = build_report(version, generated_at, results)
    filename = f"release_health_{version}_{generated_at.strftime('%Y%m%d_%H%M%S')}.md"
    output_path = OUTPUT_DIR / filename
    output_path.write_text(report, encoding="utf-8")
    print(f"Relatório gerado em: {output_path}")
    return 0 if all(item["ok"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
