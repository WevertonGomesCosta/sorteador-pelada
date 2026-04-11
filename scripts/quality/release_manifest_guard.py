#!/usr/bin/env python3
"""Guard leve do inventário estrutural obrigatório da release.

Valida que a baseline mantenha um manifesto mínimo e íntegro dos artefatos
operacionais esperados da release, sem tocar no núcleo funcional do app.

Uso:
    python scripts/quality/release_manifest_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality import compatibility_contract_guard


REQUIRED_FILES: list[str] = [
    'README.md',
    'CHANGELOG.md',
    'requirements.txt',
    'app.py',
    'scripts/quality/check_base.py',
    'scripts/quality/release_guard.py',
    'scripts/quality/quality_gate.py',
    'scripts/quality/runtime_preflight.py',
    'scripts/quality/release_metadata_guard.py',
    'scripts/quality/compatibility_contract_guard.py',
    'scripts/quality/operational_checks_contract_guard.py',
    'scripts/quality/canonical_paths_reference_guard.py',
    'scripts/quality/script_cli_contract_guard.py',
    'scripts/quality/release_artifacts_hygiene_guard.py',
    'scripts/quality/runtime_dependencies_contract_guard.py',
    'scripts/quality/documentation_commands_examples_guard.py',
    'scripts/quality/release_manifest_guard.py',
    'scripts/quality/quality_runtime_budget_guard.py',
    'scripts/quality/checks_registry.py',
    'scripts/quality/checks_registry_contract_guard.py',
    'scripts/quality/checks_registry_schema_guard.py',
    'scripts/quality/checks_registry_consumers_guard.py',
    'scripts/quality/quality_gate_composition_guard.py',
    'scripts/reports/release_health_report.py',
    'scripts/reports/manual_validation_pack.py',
    'scripts/reports/maintenance_snapshot_report.py',
    'scripts/validation/smoke_test_base.py',
    'docs/releases/BASELINE_OFICIAL.md',
    'docs/releases/RELEASE_OPERACIONAL.md',
    'docs/operations/OPERACAO_LOCAL.md',
    'docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md',
    'reports/.gitkeep',
]

REQUIRED_DIRS: list[str] = [
    'docs',
    'docs/operations',
    'docs/releases',
    'docs/validation',
    'scripts',
    'scripts/quality',
    'scripts/reports',
    'scripts/validation',
    'tests',
    'reports',
]


DOC_MARKERS: dict[str, list[str]] = {
    'README.md': [
        'python scripts/quality/release_manifest_guard.py',
        'python scripts/quality/quality_runtime_budget_guard.py',
        'python scripts/quality/governance_docs_crosslinks_guard.py',
        'python scripts/quality/checks_registry_contract_guard.py',
        'python scripts/quality/checks_registry_schema_guard.py',
        'python scripts/quality/checks_registry_consumers_guard.py',
        'python scripts/quality/quality_gate_composition_guard.py',
        'python scripts/reports/maintenance_snapshot_report.py',
        'docs/releases/BASELINE_OFICIAL.md',
        'docs/releases/RELEASE_OPERACIONAL.md',
    ],
    'docs/operations/OPERACAO_LOCAL.md': [
        'python scripts/quality/release_manifest_guard.py',
        'python scripts/quality/quality_runtime_budget_guard.py',
        'python scripts/quality/governance_docs_crosslinks_guard.py',
        'python scripts/quality/checks_registry_contract_guard.py',
        'python scripts/quality/checks_registry_schema_guard.py',
        'python scripts/quality/checks_registry_consumers_guard.py',
        'python scripts/quality/quality_gate_composition_guard.py',
        'python scripts/reports/maintenance_snapshot_report.py',
        'inventário estrutural',
    ],
    'docs/releases/RELEASE_OPERACIONAL.md': [
        'python scripts/quality/release_manifest_guard.py',
        'python scripts/quality/quality_runtime_budget_guard.py',
        'python scripts/quality/governance_docs_crosslinks_guard.py',
        'python scripts/quality/checks_registry_contract_guard.py',
        'python scripts/quality/checks_registry_schema_guard.py',
        'python scripts/quality/checks_registry_consumers_guard.py',
        'python scripts/reports/maintenance_snapshot_report.py',
        'inventário estrutural obrigatório da release',
    ],
    'docs/releases/BASELINE_OFICIAL.md': [
        'python scripts/quality/release_manifest_guard.py',
        'python scripts/quality/quality_runtime_budget_guard.py',
        'python scripts/quality/governance_docs_crosslinks_guard.py',
        'python scripts/quality/checks_registry_contract_guard.py',
        'python scripts/quality/checks_registry_schema_guard.py',
        'python scripts/quality/checks_registry_consumers_guard.py',
        'python scripts/quality/quality_gate_composition_guard.py',
        'python scripts/reports/maintenance_snapshot_report.py',
    ],
}



def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding='utf-8')


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []

    for rel_path in REQUIRED_FILES:
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f'Arquivo obrigatório ausente no manifesto da release: {rel_path}')
        else:
            notes.append(f'OK arquivo obrigatório presente: {rel_path}')

    for rel_path in REQUIRED_DIRS:
        path = ROOT / rel_path
        if not path.exists() or not path.is_dir():
            errors.append(f'Diretório obrigatório ausente no manifesto da release: {rel_path}')
        else:
            notes.append(f'OK diretório obrigatório presente: {rel_path}')

    for rel_path, expected in compatibility_contract_guard.WRAPPER_EXPECTATIONS.items():
        if not (ROOT / rel_path).exists():
            errors.append(f'Wrapper histórico obrigatório ausente do pacote: {rel_path}')
        else:
            notes.append(f'OK wrapper histórico presente: {rel_path}')
        if not (ROOT / expected['canonical_path']).exists():
            errors.append(f'Caminho canônico esperado pelo wrapper não existe: {expected["canonical_path"]}')

    for rel_path, canonical in compatibility_contract_guard.DOC_BRIDGE_EXPECTATIONS.items():
        if not (ROOT / rel_path).exists():
            errors.append(f'Arquivo-ponte histórico obrigatório ausente: {rel_path}')
        else:
            notes.append(f'OK arquivo-ponte presente: {rel_path}')
        if not (ROOT / canonical).exists():
            errors.append(f'Documento canônico esperado pela ponte não existe: {canonical}')

    aggregator = ROOT / compatibility_contract_guard.AGGREGATOR_PATH
    if not aggregator.exists():
        errors.append(f'Agregador de compatibilidade ausente do pacote: {compatibility_contract_guard.AGGREGATOR_PATH}')
    else:
        notes.append(f'OK agregador de compatibilidade presente: {compatibility_contract_guard.AGGREGATOR_PATH}')

    for rel_path, markers in DOC_MARKERS.items():
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f'Documento obrigatório ausente para validação de manifesto: {rel_path}')
            continue
        text = read_text(rel_path)
        for marker in markers:
            if marker not in text:
                errors.append(f'{rel_path} deve citar `{marker}` para manter o manifesto operacional coerente')
            else:
                notes.append(f'OK marcador do manifesto presente: {rel_path} -> {marker}')

    print('=== RELEASE MANIFEST GUARD | Sorteador Pelada PRO ===')
    for note in notes:
        print(f'[OK] {note}')

    if errors:
        print('\nErros encontrados:')
        for error in errors:
            print(f' - {error}')
        return 1

    print('\nManifesto estrutural obrigatório da release íntegro.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
