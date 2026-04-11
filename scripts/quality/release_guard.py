#!/usr/bin/env python3
"""Gate operacional de release da base do Sorteador Pelada PRO.

Valida:
- integridade estrutural via scripts/quality/check_base.py;
- sincronização entre versão do rodapé e versão mais recente do CHANGELOG.md;
- sincronização adicional entre rodapé, changelog e baseline oficial via release_metadata_guard;
- presença dos artefatos mínimos de governança e release;
- higiene do pacote (sem __pycache__ e sem .pyc) ao final da execução.

Uso:
    python scripts/quality/release_guard.py
"""

from __future__ import annotations

from datetime import datetime
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_RELEASE_FILES = [
    "README.md",
    "CHANGELOG.md",
    "CHECKLIST_REGRESSAO.md",
    "docs/README.md",
    "docs/ARQUITETURA_BASE.md",
    "docs/MANUTENCAO_OPERACIONAL.md",
    "docs/RELEASE_OPERACIONAL.md",
    "docs/BASELINE_OFICIAL.md",
    "docs/PLANO_SMOKE_TEST_MINIMO.md",
    "docs/VALIDACAO_MANUAL_GUIA.md",
    "docs/OPERACAO_LOCAL.md",
    "docs/POLITICA_COMPATIBILIDADE_TEMPORARIA.md",
    "docs/VALIDACAO_UX_MOBILE_2026-04-09.md",
    "docs/architecture/ARQUITETURA_BASE.md",
    "docs/operations/MANUTENCAO_OPERACIONAL.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md",
    "docs/releases/BASELINE_OFICIAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
    "docs/releases/PROTECTED_SCOPE_HASHES.json",
    "docs/validation/PLANO_SMOKE_TEST_MINIMO.md",
    "docs/validation/VALIDACAO_MANUAL_GUIA.md",
    "scripts/check_base.py",
    "scripts/release_guard.py",
    "scripts/smoke_test_base.py",
    "scripts/quality_gate.py",
    "scripts/runtime_preflight.py",
    "scripts/release_metadata_guard.py",
    "scripts/compatibility_contract_guard.py",
    "scripts/operational_checks_contract_guard.py",
    "scripts/canonical_paths_reference_guard.py",
    "scripts/runtime_dependencies_contract_guard.py",
    "scripts/documentation_commands_examples_guard.py",
    "scripts/manual_validation_pack.py",
    "scripts/release_health_report.py",
    "scripts/quality/check_base.py",
    "scripts/quality/release_guard.py",
    "scripts/quality/quality_gate.py",
    "scripts/quality/runtime_preflight.py",
    "scripts/quality/release_metadata_guard.py",
    "scripts/quality/compatibility_contract_guard.py",
    "scripts/quality/operational_checks_contract_guard.py",
    "scripts/quality/canonical_paths_reference_guard.py",
    "scripts/quality/runtime_dependencies_contract_guard.py",
    "scripts/quality/documentation_commands_examples_guard.py",
    "scripts/validation/smoke_test_base.py",
    "scripts/reports/manual_validation_pack.py",
    "scripts/reports/release_health_report.py",
    "scripts/reports/maintenance_snapshot_report.py",
    "tests/test_smoke_base.py",
    "tests/test_core_smoke.py",
    "tests/test_state_smoke.py",
    "tests/test_ui_safe_smoke.py",
    "tests/test_scripts_smoke.py",
    "reports/.gitkeep",
    "ui/primitives.py",
]


def remove_bytecode_artifacts(root: Path) -> tuple[int, int]:
    pycache_count = 0
    pyc_count = 0
    for cache_dir in sorted(root.rglob("__pycache__")):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)
            pycache_count += 1
    for pyc_file in sorted(root.rglob("*.pyc")):
        if pyc_file.exists():
            pyc_file.unlink(missing_ok=True)
            pyc_count += 1
    return pycache_count, pyc_count


def find_artifacts(root: Path) -> tuple[list[Path], list[Path]]:
    caches = [p for p in root.rglob("__pycache__") if p.is_dir()]
    pycs = [p for p in root.rglob("*.pyc") if p.is_file()]
    return caches, pycs


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def footer_version() -> str | None:
    text = read_text("ui/primitives.py")
    match = re.search(r'versao\s*:\s*str\s*=\s*"(v\d+)"', text)
    return match.group(1) if match else None


def latest_changelog_version() -> str | None:
    text = read_text("CHANGELOG.md")
    match = re.search(r"^##\s+(v\d+)\s+—", text, flags=re.MULTILINE)
    return match.group(1) if match else None


def latest_changelog_date() -> str | None:
    text = read_text("CHANGELOG.md")
    match = re.search(r"^##\s+v\d+\s+—\s+([0-9]{4}-[0-9]{2}-[0-9]{2})", text, flags=re.MULTILINE)
    return match.group(1) if match else None


def latest_project_update() -> str:
    latest = max(
        p.stat().st_mtime
        for p in ROOT.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    return datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M:%S")


def run_check_base() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "quality" / "check_base.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return proc.returncode == 0, output.strip()


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []

    for rel_path in REQUIRED_RELEASE_FILES:
        if not (ROOT / rel_path).exists():
            errors.append(f"Arquivo obrigatório ausente para release: {rel_path}")
        else:
            notes.append(f"OK artefato de release: {rel_path}")

    check_ok, check_output = run_check_base()
    if check_ok:
        notes.append("OK check_base executado com sucesso")
    else:
        errors.append("Falha em scripts/quality/check_base.py")

    footer = footer_version()
    changelog = latest_changelog_version()
    if not footer:
        errors.append("Não foi possível identificar a versão do rodapé em ui/primitives.py")
    else:
        notes.append(f"OK versão do rodapé detectada: {footer}")
    if not changelog:
        errors.append("Não foi possível identificar a versão mais recente em CHANGELOG.md")
    else:
        notes.append(f"OK versão mais recente no changelog: {changelog}")
    if footer and changelog and footer != changelog:
        errors.append(f"Divergência entre rodapé ({footer}) e changelog ({changelog})")
    elif footer and changelog:
        notes.append("OK sincronização entre rodapé e changelog")

    release_doc = read_text("docs/releases/RELEASE_OPERACIONAL.md") if (ROOT / "docs/releases/RELEASE_OPERACIONAL.md").exists() else ""
    if "scripts/quality/release_guard.py" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar scripts/quality/release_guard.py")
    else:
        notes.append("OK protocolo de release cita o release_guard")

    if "scripts/quality/release_metadata_guard.py" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar scripts/quality/release_metadata_guard.py")
    else:
        notes.append("OK protocolo de release cita o release_metadata_guard")

    if "scripts/quality/compatibility_contract_guard.py" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar scripts/quality/compatibility_contract_guard.py")
    else:
        notes.append("OK protocolo de release cita o compatibility_contract_guard")

    if "scripts/quality/operational_checks_contract_guard.py" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar scripts/quality/operational_checks_contract_guard.py")
    else:
        notes.append("OK protocolo de release cita o operational_checks_contract_guard")

    if "scripts/quality/canonical_paths_reference_guard.py" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar scripts/quality/canonical_paths_reference_guard.py")
    else:
        notes.append("OK protocolo de release cita o canonical_paths_reference_guard")

    if "scripts/quality/runtime_dependencies_contract_guard.py" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar scripts/quality/runtime_dependencies_contract_guard.py")
    else:
        notes.append("OK protocolo de release cita o runtime_dependencies_contract_guard")

    if "scripts/quality/documentation_commands_examples_guard.py" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar scripts/quality/documentation_commands_examples_guard.py")
    else:
        notes.append("OK protocolo de release cita o documentation_commands_examples_guard")

    if "scripts/quality/release_manifest_guard.py" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar scripts/quality/release_manifest_guard.py")
    else:
        notes.append("OK protocolo de release cita o release_manifest_guard")

    if "scripts/quality/checks_registry_contract_guard.py" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar scripts/quality/checks_registry_contract_guard.py")
    else:
        notes.append("OK protocolo de release cita o checks_registry_contract_guard")

    if "scripts/quality/checks_registry_schema_guard.py" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar scripts/quality/checks_registry_schema_guard.py")
    else:
        notes.append("OK protocolo de release cita o checks_registry_schema_guard")

    if "scripts/quality/checks_registry_consumers_guard.py" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar scripts/quality/checks_registry_consumers_guard.py")
    else:
        notes.append("OK protocolo de release cita o checks_registry_consumers_guard")

    if "scripts/quality/quality_gate_composition_guard.py" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar scripts/quality/quality_gate_composition_guard.py")
    else:
        notes.append("OK protocolo de release cita o quality_gate_composition_guard")

    if "scripts/quality/script_exit_codes_contract_guard.py" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar scripts/quality/script_exit_codes_contract_guard.py")
    else:
        notes.append("OK protocolo de release cita o script_exit_codes_contract_guard")

    if "scripts/reports/maintenance_snapshot_report.py" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar scripts/reports/maintenance_snapshot_report.py")
    else:
        notes.append("OK protocolo de release cita o maintenance_snapshot_report")

    if "scripts/quality/protected_scope_hash_guard.py" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar scripts/quality/protected_scope_hash_guard.py")
    else:
        notes.append("OK protocolo de release cita o protected_scope_hash_guard")

    if "docs/releases/PROTECTED_SCOPE_HASHES.json" not in release_doc:
        errors.append("docs/releases/RELEASE_OPERACIONAL.md deve mencionar docs/releases/PROTECTED_SCOPE_HASHES.json")
    else:
        notes.append("OK protocolo de release cita o manifesto de hashes protegido")

    readme = read_text("README.md") if (ROOT / "README.md").exists() else ""
    if "python scripts/quality/release_guard.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/release_guard.py")
    else:
        notes.append("OK README orienta o uso do release_guard")

    if "python scripts/quality/quality_gate.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/quality_gate.py")
    else:
        notes.append("OK README orienta o uso do quality_gate")

    if "python scripts/quality/runtime_preflight.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/runtime_preflight.py")
    else:
        notes.append("OK README orienta o uso do runtime_preflight")

    if "python scripts/quality/release_metadata_guard.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/release_metadata_guard.py")
    else:
        notes.append("OK README orienta o uso do release_metadata_guard")

    if "python scripts/quality/compatibility_contract_guard.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/compatibility_contract_guard.py")
    else:
        notes.append("OK README orienta o uso do compatibility_contract_guard")

    if "python scripts/quality/operational_checks_contract_guard.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/operational_checks_contract_guard.py")
    else:
        notes.append("OK README orienta o uso do operational_checks_contract_guard")

    if "python scripts/quality/canonical_paths_reference_guard.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/canonical_paths_reference_guard.py")
    else:
        notes.append("OK README orienta o uso do canonical_paths_reference_guard")

    if "python scripts/quality/script_cli_contract_guard.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/script_cli_contract_guard.py")
    else:
        notes.append("OK README orienta o uso do script_cli_contract_guard")

    if "python scripts/quality/runtime_dependencies_contract_guard.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/runtime_dependencies_contract_guard.py")
    else:
        notes.append("OK README orienta o uso do runtime_dependencies_contract_guard")

    if "python scripts/quality/documentation_commands_examples_guard.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/documentation_commands_examples_guard.py")
    else:
        notes.append("OK README orienta o uso do documentation_commands_examples_guard")

    if "python scripts/quality/release_manifest_guard.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/release_manifest_guard.py")
    else:
        notes.append("OK README orienta o uso do release_manifest_guard")

    if "python scripts/quality/script_exit_codes_contract_guard.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/script_exit_codes_contract_guard.py")
    else:
        notes.append("OK README orienta o uso do script_exit_codes_contract_guard")

    if "python scripts/quality/protected_scope_hash_guard.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/protected_scope_hash_guard.py")
    else:
        notes.append("OK README orienta o uso do protected_scope_hash_guard")

    if "python scripts/quality/checks_registry_contract_guard.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/checks_registry_contract_guard.py")
    else:
        notes.append("OK README orienta o uso do checks_registry_contract_guard")

    if "python scripts/quality/checks_registry_schema_guard.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/checks_registry_schema_guard.py")
    else:
        notes.append("OK README orienta o uso do checks_registry_schema_guard")

    if "python scripts/quality/checks_registry_consumers_guard.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/quality/checks_registry_consumers_guard.py")
    else:
        notes.append("OK README orienta o uso do checks_registry_consumers_guard")

    if "python scripts/reports/manual_validation_pack.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/reports/manual_validation_pack.py")
    else:
        notes.append("OK README orienta o uso do manual_validation_pack")

    if "python scripts/reports/maintenance_snapshot_report.py" not in readme:
        errors.append("README.md deve orientar o uso de python scripts/reports/maintenance_snapshot_report.py")
    else:
        notes.append("OK README orienta o uso do maintenance_snapshot_report")

    compatibility_policy = read_text("docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md") if (ROOT / "docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md").exists() else ""
    if "2 releases oficiais estáveis completas após a v70" not in compatibility_policy:
        errors.append("docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md deve definir critério objetivo de janela mínima de estabilidade")
    else:
        notes.append("OK política de compatibilidade define janela mínima de estabilidade")

    if ("scripts/quality/check_base.py" not in compatibility_policy or
        "scripts/quality/release_guard.py" not in compatibility_policy or
        "scripts/quality/compatibility_contract_guard.py" not in compatibility_policy):
        errors.append("docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md deve citar os gates canônicos envolvidos na futura remoção do legado")
    else:
        notes.append("OK política de compatibilidade cita os gates canônicos")

    removed_cache, removed_pyc = remove_bytecode_artifacts(ROOT)
    notes.append(f"OK limpeza pós-checagem: {removed_cache} __pycache__ removidos, {removed_pyc} .pyc removidos")

    caches, pycs = find_artifacts(ROOT)
    if caches:
        errors.append("Persistem diretórios __pycache__: " + ", ".join(str(p.relative_to(ROOT)) for p in caches[:10]))
    else:
        notes.append("OK sem diretórios __pycache__ no pacote final")
    if pycs:
        errors.append("Persistem arquivos .pyc: " + ", ".join(str(p.relative_to(ROOT)) for p in pycs[:10]))
    else:
        notes.append("OK sem arquivos .pyc no pacote final")

    print("=== RELEASE GUARD | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")

    version_label = footer or changelog or "desconhecida"
    changelog_date = latest_changelog_date() or "desconhecida"
    check_status = "OK" if check_ok else "FALHOU"
    print("\nResumo da release:")
    print(f"- Versão: {version_label}")
    print(f"- Data do changelog: {changelog_date}")
    print(f"- Última atualização detectada no projeto: {latest_project_update()}")
    print(f"- Artefatos obrigatórios: {len(REQUIRED_RELEASE_FILES)} itens")
    print(f"- Status check_base: {check_status}")

    if errors:
        print("\nSaída do check_base:")
        if check_output:
            print(check_output)
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nRelease operacional íntegra.")
    print("Sugestão: use CHECKLIST_REGRESSAO.md para a validação funcional final no navegador/mobile.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
