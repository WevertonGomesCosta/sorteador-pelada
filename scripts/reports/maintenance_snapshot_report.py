#!/usr/bin/env python3
"""Gera um snapshot operacional somente leitura da baseline atual.

Objetivo:
- consolidar em um único arquivo um diagnóstico operacional estático da versão;
- reduzir a triagem manual dispersa entre documentação, manifesto e compatibilidade;
- apoiar manutenção pontual sem criar novo guard nem tocar no núcleo funcional.

Uso:
    python scripts/reports/maintenance_snapshot_report.py
"""

from __future__ import annotations

from datetime import datetime
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality import compatibility_contract_guard, protected_scope_hash_guard, release_metadata_guard
from scripts.quality.checks_registry import expected_official_checks

OUTPUT_DIR = ROOT / "reports"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"
BASELINE_PATH = ROOT / "docs" / "releases" / "BASELINE_OFICIAL.md"
PRIMITIVES_PATH = ROOT / "ui" / "primitives.py"
MANIFEST_PATH = ROOT / "docs" / "releases" / "PROTECTED_SCOPE_HASHES.json"

GOVERNANCE_DOCS = [
    "README.md",
    "docs/releases/BASELINE_OFICIAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/operations/MANUTENCAO_OPERACIONAL.md",
    "docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md",
    "docs/validation/VALIDACAO_MANUAL_GUIA.md",
]

RECOMMENDED_COMMANDS = [
    "python scripts/quality/runtime_preflight.py",
    "python scripts/quality/quality_gate.py",
    "python scripts/reports/manual_validation_pack.py",
    "python scripts/reports/release_health_report.py",
    "python scripts/reports/maintenance_snapshot_report.py",
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


def governance_docs_snapshot() -> list[dict[str, object]]:
    snapshot: list[dict[str, object]] = []
    for rel_path in GOVERNANCE_DOCS:
        path = ROOT / rel_path
        snapshot.append({
            "rel_path": rel_path,
            "exists": path.exists(),
        })
    return snapshot


def wrappers_snapshot() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for wrapper_path, expected in compatibility_contract_guard.WRAPPER_EXPECTATIONS.items():
        canonical_path = expected["canonical_path"]
        rows.append(
            {
                "wrapper_path": wrapper_path,
                "wrapper_exists": (ROOT / wrapper_path).exists(),
                "canonical_path": canonical_path,
                "canonical_exists": (ROOT / canonical_path).exists(),
            }
        )
    return rows


def doc_bridges_snapshot() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bridge_path, canonical_path in compatibility_contract_guard.DOC_BRIDGE_EXPECTATIONS.items():
        rows.append(
            {
                "bridge_path": bridge_path,
                "bridge_exists": (ROOT / bridge_path).exists(),
                "canonical_path": canonical_path,
                "canonical_exists": (ROOT / canonical_path).exists(),
            }
        )
    return rows


def protected_scope_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "manifest_exists": MANIFEST_PATH.exists(),
        "schema_version": None,
        "baseline_version": None,
        "files": [],
    }
    if not MANIFEST_PATH.exists():
        return snapshot

    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    snapshot["schema_version"] = payload.get("schema_version")
    snapshot["baseline_version"] = payload.get("baseline_version")

    protected_files = payload.get("protected_files", {})
    if not isinstance(protected_files, dict):
        return snapshot

    file_rows: list[dict[str, object]] = []
    for rel_path, meta in protected_files.items():
        path = ROOT / rel_path
        exists = path.exists()
        expected_hash = meta.get("sha256") if isinstance(meta, dict) else None
        current_hash = protected_scope_hash_guard.sha256_file(path) if exists else None
        file_rows.append(
            {
                "rel_path": rel_path,
                "exists": exists,
                "status": meta.get("status") if isinstance(meta, dict) else None,
                "expected_hash": expected_hash,
                "current_hash": current_hash,
                "hash_ok": bool(exists and isinstance(expected_hash, str) and current_hash == expected_hash),
            }
        )
    snapshot["files"] = file_rows
    return snapshot


def reports_dir_snapshot() -> dict[str, object]:
    entries = []
    if OUTPUT_DIR.exists():
        entries = sorted(path.name for path in OUTPUT_DIR.iterdir())
    return {
        "exists": OUTPUT_DIR.exists(),
        "entries": entries,
        "is_clean_package_state": entries == [".gitkeep"],
    }


def build_snapshot(generated_at: datetime) -> dict[str, object]:
    official_checks = expected_official_checks(target="quality_gate")
    return {
        "version": detect_version(),
        "generated_at": generated_at,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "root": ROOT.as_posix(),
        "official_checks": official_checks,
        "wrappers": wrappers_snapshot(),
        "doc_bridges": doc_bridges_snapshot(),
        "protected_scope": protected_scope_snapshot(),
        "governance_docs": governance_docs_snapshot(),
        "reports_dir": reports_dir_snapshot(),
        "recommended_commands": RECOMMENDED_COMMANDS,
    }


def build_report(snapshot: dict[str, object]) -> str:
    version = str(snapshot["version"])
    generated_at = snapshot["generated_at"]
    assert isinstance(generated_at, datetime)
    timestamp_display = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    official_checks = snapshot["official_checks"]
    wrappers = snapshot["wrappers"]
    doc_bridges = snapshot["doc_bridges"]
    protected_scope = snapshot["protected_scope"]
    governance_docs = snapshot["governance_docs"]
    reports_dir = snapshot["reports_dir"]

    wrapper_ok = sum(1 for item in wrappers if item["wrapper_exists"] and item["canonical_exists"])
    bridge_ok = sum(1 for item in doc_bridges if item["bridge_exists"] and item["canonical_exists"])
    protected_files = protected_scope.get("files", []) if isinstance(protected_scope, dict) else []
    protected_ok = sum(1 for item in protected_files if item.get("hash_ok"))
    docs_ok = sum(1 for item in governance_docs if item["exists"])

    lines: list[str] = []
    lines.append(f"# MAINTENANCE_SNAPSHOT_REPORT — {version}")
    lines.append("")
    lines.append("## Metadados")
    lines.append(f"- Versão detectada: `{version}`")
    lines.append(f"- Gerado em: `{timestamp_display}`")
    lines.append(f"- Python: `{snapshot['python_version']}`")
    lines.append(f"- Plataforma: `{snapshot['platform']}`")
    lines.append(f"- Raiz inspecionada: `{snapshot['root']}`")
    lines.append("")
    lines.append("## Resumo executivo")
    lines.append(f"- Checks oficiais cadastrados: `{len(official_checks)}`")
    lines.append(f"- Wrappers temporários íntegros por presença: `{wrapper_ok}/{len(wrappers)}`")
    lines.append(f"- Arquivos-ponte presentes por presença: `{bridge_ok}/{len(doc_bridges)}`")
    lines.append(f"- Arquivos protegidos com hash íntegro: `{protected_ok}/{len(protected_files)}`")
    lines.append(f"- Documentos canônicos presentes: `{docs_ok}/{len(governance_docs)}`")
    lines.append(f"- Estado atual de `reports/`: `{'limpo' if reports_dir['is_clean_package_state'] else 'com artefatos locais'}`")
    lines.append("- Natureza do snapshot: **somente leitura**, sem executar o quality gate e sem alterar contratos temporários.")
    lines.append("")
    lines.append("## Rotina oficial de checks cadastrada")
    for name, command in official_checks:
        lines.append(f"- `{name}` — `{command}`")
    lines.append("")
    lines.append("## Compatibilidade temporária")
    lines.append("### Wrappers históricos")
    for item in wrappers:
        status = "OK" if item["wrapper_exists"] and item["canonical_exists"] else "PENDENTE"
        lines.append(
            f"- `{item['wrapper_path']}` → `{item['canonical_path']}` — `{status}`"
        )
    lines.append("")
    lines.append("### Arquivos-ponte em docs/")
    for item in doc_bridges:
        status = "OK" if item["bridge_exists"] and item["canonical_exists"] else "PENDENTE"
        lines.append(
            f"- `{item['bridge_path']}` → `{item['canonical_path']}` — `{status}`"
        )
    lines.append("")
    lines.append("## Escopo protegido")
    lines.append(f"- Manifesto: `{MANIFEST_PATH.relative_to(ROOT).as_posix()}`")
    lines.append(f"- Manifesto presente: `{'sim' if protected_scope.get('manifest_exists') else 'não'}`")
    lines.append(f"- schema_version: `{protected_scope.get('schema_version')}`")
    lines.append(f"- baseline_version no manifesto: `{protected_scope.get('baseline_version')}`")
    for item in protected_files:
        status = "OK" if item.get("hash_ok") else "PENDENTE"
        lines.append(
            f"- `{item['rel_path']}` — status manifesto=`{item.get('status')}` — hash=`{status}`"
        )
    lines.append("")
    lines.append("## Documentação canônica presente")
    for item in governance_docs:
        status = "OK" if item["exists"] else "PENDENTE"
        lines.append(f"- `{item['rel_path']}` — `{status}`")
    lines.append("")
    lines.append("## Estado do diretório reports/")
    lines.append(f"- Diretório presente: `{'sim' if reports_dir['exists'] else 'não'}`")
    lines.append(f"- Estado limpo para pacote oficial: `{'sim' if reports_dir['is_clean_package_state'] else 'não'}`")
    if reports_dir["entries"]:
        for entry in reports_dir["entries"]:
            lines.append(f"- `{entry}`")
    else:
        lines.append("- sem entradas detectadas")
    lines.append("")
    lines.append("## Comandos recomendados")
    for command in snapshot["recommended_commands"]:
        lines.append(f"- `{command}`")
    lines.append("")
    lines.append("## Fechamento")
    lines.append("- Este snapshot consolida o estado operacional da baseline sem introduzir novo guard.")
    lines.append("- O artefato é de apoio a triagem, handoff e manutenção pontual fora do núcleo funcional.")
    return "\n".join(lines) + "\n"


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(__doc__.strip())
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now()
    snapshot = build_snapshot(generated_at)
    report = build_report(snapshot)
    version = str(snapshot["version"])
    output_path = OUTPUT_DIR / f"maintenance_snapshot_{version}_{generated_at.strftime('%Y%m%d_%H%M%S')}.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"Snapshot gerado em: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
