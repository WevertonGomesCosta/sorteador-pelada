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

OUTPUT_DIR = ROOT / "reports"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"
BASELINE_PATH = ROOT / "docs" / "releases" / "BASELINE_OFICIAL.md"
PRIMITIVES_PATH = ROOT / "ui" / "primitives.py"

CHECKS: list[tuple[str, list[str]]] = [
    ("check_base", [sys.executable, str(ROOT / "scripts" / "quality" / "check_base.py")]),
    ("smoke_test_base", [sys.executable, str(ROOT / "scripts" / "validation" / "smoke_test_base.py")]),
    ("compileall", [sys.executable, "-m", "compileall", "."]),
    ("release_metadata_guard", [sys.executable, str(ROOT / "scripts" / "quality" / "release_metadata_guard.py")]),
    ("compatibility_contract_guard", [sys.executable, str(ROOT / "scripts" / "quality" / "compatibility_contract_guard.py")]),
    ("release_guard", [sys.executable, str(ROOT / "scripts" / "quality" / "release_guard.py")]),
]

CANONICAL_PATHS = [
    "scripts/quality/check_base.py",
    "scripts/validation/smoke_test_base.py",
    "scripts/quality/release_metadata_guard.py",
    "scripts/quality/compatibility_contract_guard.py",
    "scripts/quality/release_guard.py",
    "scripts/quality/quality_gate.py",
    "scripts/reports/manual_validation_pack.py",
    "scripts/reports/release_health_report.py",
    "docs/releases/BASELINE_OFICIAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
    "docs/operations/OPERACAO_LOCAL.md",
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
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
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
    lines.append("")
    lines.append("## Status dos checks")
    for item in results:
        status = "OK" if item["ok"] else "FALHOU"
        lines.append(f"- **{item['name']}** — `{status}` — `{item['command']}`")
    lines.append("")
    lines.append("## Inventário resumido de compatibilidade temporária")
    lines.append(f"- Wrappers históricos em `scripts/`: `{inventory['wrappers']}`")
    lines.append(f"- Arquivos-ponte históricos em `docs/`: `{inventory['doc_bridges']}`")
    lines.append(f"- Agregador compatível em `tests/`: `{inventory['compatibility_aggregators']}`")
    lines.append("")
    lines.append("## Caminhos canônicos principais")
    for rel_path in CANONICAL_PATHS:
        lines.append(f"- `{rel_path}`")
    lines.append("")
    lines.append("## Evidência resumida por check")
    for item in results:
        lines.append(f"### {item['name']}")
        lines.append(f"- Status: `{'OK' if item['ok'] else 'FALHOU'}`")
        lines.append(f"- Comando: `{item['command']}`")
        lines.append(f"- Return code: `{item['returncode']}`")
        excerpt = item["excerpt"]
        if excerpt:
            lines.append("- Saída resumida:")
            lines.append("```text")
            lines.extend(excerpt)
            lines.append("```")
        else:
            lines.append("- Saída resumida: sem conteúdo relevante.")
        lines.append("")
    lines.append("## Conclusão")
    lines.append(f"- Status final: `{'aprovado' if fail_count == 0 else 'requer atenção'}`")
    lines.append("- Observação: este relatório é complementar ao `quality_gate` e não altera a política de compatibilidade temporária.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    version = detect_version()
    now = datetime.now()
    results = [run_check(name, command) for name, command in CHECKS]
    timestamp_file = now.strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"release_health_{version}_{timestamp_file}.md"
    output_path.write_text(build_report(version, now, results), encoding="utf-8")

    print("=== RELEASE HEALTH REPORT | Sorteador Pelada PRO ===")
    print(f"Versão detectada: {version}")
    print(f"Checks executados: {len(results)}")
    print(f"Arquivo gerado: {output_path.relative_to(ROOT)}")
    failures = [item["name"] for item in results if not item["ok"]]
    if failures:
        print("Checks com falha: " + ", ".join(failures))
        return 1
    print("Release health report gerado com todos os checks aprovados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
