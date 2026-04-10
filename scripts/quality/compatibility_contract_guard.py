#!/usr/bin/env python3
"""Guard leve do contrato de compatibilidade temporária.

Valida a integridade do legado temporário mantido durante a transição estável:
- wrappers históricos em ``scripts/``;
- arquivos-ponte históricos na raiz de ``docs/``;
- agregador compatível ``tests/test_smoke_base.py``.

Uso:
    python scripts/quality/compatibility_contract_guard.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

WRAPPER_EXPECTATIONS: dict[str, dict[str, str]] = {
    'scripts/check_base.py': {
        'canonical_path': 'scripts/quality/check_base.py',
        'import_line': 'from scripts.quality.check_base import main',
    },
    'scripts/release_guard.py': {
        'canonical_path': 'scripts/quality/release_guard.py',
        'import_line': 'from scripts.quality.release_guard import main',
    },
    'scripts/quality_gate.py': {
        'canonical_path': 'scripts/quality/quality_gate.py',
        'import_line': 'from scripts.quality.quality_gate import main',
    },
    'scripts/runtime_preflight.py': {
        'canonical_path': 'scripts/quality/runtime_preflight.py',
        'import_line': 'from scripts.quality.runtime_preflight import main',
    },
    'scripts/release_metadata_guard.py': {
        'canonical_path': 'scripts/quality/release_metadata_guard.py',
        'import_line': 'from scripts.quality.release_metadata_guard import main',
    },
    'scripts/compatibility_contract_guard.py': {
        'canonical_path': 'scripts/quality/compatibility_contract_guard.py',
        'import_line': 'from scripts.quality.compatibility_contract_guard import main',
    },
    'scripts/operational_checks_contract_guard.py': {
        'canonical_path': 'scripts/quality/operational_checks_contract_guard.py',
        'import_line': 'from scripts.quality.operational_checks_contract_guard import main',
    },
    'scripts/canonical_paths_reference_guard.py': {
        'canonical_path': 'scripts/quality/canonical_paths_reference_guard.py',
        'import_line': 'from scripts.quality.canonical_paths_reference_guard import main',
    },
    'scripts/script_cli_contract_guard.py': {
        'canonical_path': 'scripts/quality/script_cli_contract_guard.py',
        'import_line': 'from scripts.quality.script_cli_contract_guard import main',
    },
    'scripts/release_artifacts_hygiene_guard.py': {
        'canonical_path': 'scripts/quality/release_artifacts_hygiene_guard.py',
        'import_line': 'from scripts.quality.release_artifacts_hygiene_guard import main',
    },
    'scripts/runtime_dependencies_contract_guard.py': {
        'canonical_path': 'scripts/quality/runtime_dependencies_contract_guard.py',
        'import_line': 'from scripts.quality.runtime_dependencies_contract_guard import main',
    },
    'scripts/documentation_commands_examples_guard.py': {
        'canonical_path': 'scripts/quality/documentation_commands_examples_guard.py',
        'import_line': 'from scripts.quality.documentation_commands_examples_guard import main',
    },
    'scripts/release_manifest_guard.py': {
        'canonical_path': 'scripts/quality/release_manifest_guard.py',
        'import_line': 'from scripts.quality.release_manifest_guard import main',
    },
    'scripts/manual_validation_pack.py': {
        'canonical_path': 'scripts/reports/manual_validation_pack.py',
        'import_line': 'from scripts.reports.manual_validation_pack import main',
    },
    'scripts/release_health_report.py': {
        'canonical_path': 'scripts/reports/release_health_report.py',
        'import_line': 'from scripts.reports.release_health_report import main',
    },
    'scripts/smoke_test_base.py': {
        'canonical_path': 'scripts/validation/smoke_test_base.py',
        'import_line': 'from scripts.validation.smoke_test_base import main',
    },
}

DOC_BRIDGE_EXPECTATIONS: dict[str, str] = {
    'docs/ARQUITETURA_BASE.md': 'docs/architecture/ARQUITETURA_BASE.md',
    'docs/BASELINE_OFICIAL.md': 'docs/releases/BASELINE_OFICIAL.md',
    'docs/MANUTENCAO_OPERACIONAL.md': 'docs/operations/MANUTENCAO_OPERACIONAL.md',
    'docs/OPERACAO_LOCAL.md': 'docs/operations/OPERACAO_LOCAL.md',
    'docs/PLANO_SMOKE_TEST_MINIMO.md': 'docs/validation/PLANO_SMOKE_TEST_MINIMO.md',
    'docs/POLITICA_COMPATIBILIDADE_TEMPORARIA.md': 'docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md',
    'docs/RELEASE_OPERACIONAL.md': 'docs/releases/RELEASE_OPERACIONAL.md',
    'docs/VALIDACAO_MANUAL_GUIA.md': 'docs/validation/VALIDACAO_MANUAL_GUIA.md',
    'docs/VALIDACAO_UX_MOBILE_2026-04-09.md': 'docs/validation/VALIDACAO_UX_MOBILE_2026-04-09.md',
}

COMPATIBILITY_POLICY_PATH = 'docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md'
AGGREGATOR_PATH = 'tests/test_smoke_base.py'
AGGREGATOR_IMPORTS = [
    'from tests.test_core_smoke import *',
    'from tests.test_state_smoke import *',
    'from tests.test_ui_safe_smoke import *',
    'from tests.test_scripts_smoke import *',
]


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding='utf-8')


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []

    for rel_path, expected in WRAPPER_EXPECTATIONS.items():
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f'Wrapper histórico ausente: {rel_path}')
            continue

        text = read_text(rel_path)
        if 'Wrapper temporário de compatibilidade' not in text:
            errors.append(f'{rel_path} deve se declarar como wrapper temporário de compatibilidade')
        else:
            notes.append(f'OK wrapper identificado: {rel_path}')

        canonical_path = expected['canonical_path']
        if canonical_path not in text:
            errors.append(f'{rel_path} deve apontar explicitamente para {canonical_path}')
        else:
            notes.append(f'OK wrapper aponta para caminho canônico: {rel_path} -> {canonical_path}')

        import_line = expected['import_line']
        if import_line not in text:
            errors.append(f'{rel_path} deve importar main via `{import_line}`')
        else:
            notes.append(f'OK import canônico do wrapper: {rel_path}')

    for rel_path, canonical_path in DOC_BRIDGE_EXPECTATIONS.items():
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f'Arquivo-ponte histórico ausente: {rel_path}')
            continue

        text = read_text(rel_path)
        if 'compatibilidade' not in text.lower():
            errors.append(f'{rel_path} deve indicar explicitamente que é ponte de compatibilidade')
        else:
            notes.append(f'OK ponte de compatibilidade identificada: {rel_path}')

        if canonical_path not in text:
            errors.append(f'{rel_path} deve apontar explicitamente para {canonical_path}')
        else:
            notes.append(f'OK ponte aponta para documento canônico: {rel_path} -> {canonical_path}')

    aggregator = ROOT / AGGREGATOR_PATH
    if not aggregator.exists():
        errors.append(f'Agregador compatível ausente: {AGGREGATOR_PATH}')
    else:
        text = read_text(AGGREGATOR_PATH)
        if 'Agregador de compatibilidade' not in text:
            errors.append(f'{AGGREGATOR_PATH} deve se declarar como agregador de compatibilidade')
        else:
            notes.append(f'OK agregador compatível identificado: {AGGREGATOR_PATH}')

        for import_line in AGGREGATOR_IMPORTS:
            if import_line not in text:
                errors.append(f'{AGGREGATOR_PATH} deve agregar `{import_line}`')
            else:
                notes.append(f'OK agregador inclui: {import_line}')

    policy_text = read_text(COMPATIBILITY_POLICY_PATH) if (ROOT / COMPATIBILITY_POLICY_PATH).exists() else ''
    required_policy_markers = [
        'Wrappers em `scripts/`',
        'Arquivos-ponte na raiz de `docs/`',
        '`tests/test_smoke_base.py`',
        'python scripts/quality/compatibility_contract_guard.py',
    ]
    for marker in required_policy_markers:
        if marker not in policy_text:
            errors.append(f'{COMPATIBILITY_POLICY_PATH} deve citar {marker}')
        else:
            notes.append(f'OK política cita: {marker}')

    print('=== COMPATIBILITY CONTRACT GUARD | Sorteador Pelada PRO ===')
    for note in notes:
        print(f'[OK] {note}')

    if errors:
        print('\nErros encontrados:')
        for error in errors:
            print(f' - {error}')
        return 1

    print('\nContrato de compatibilidade temporária íntegro.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
