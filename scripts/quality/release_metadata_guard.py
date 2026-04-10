#!/usr/bin/env python3
"""Guard leve de consistência de metadados de release.

Valida a sincronização entre:
- versão do rodapé em ``ui/primitives.py``;
- versão mais recente do ``CHANGELOG.md``;
- versão oficial vigente em ``docs/releases/BASELINE_OFICIAL.md``.

Também verifica se os documentos operacionais centrais já orientam o uso do
check canônico desta etapa.

Uso:
    python scripts/quality/release_metadata_guard.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = [
    'ui/primitives.py',
    'CHANGELOG.md',
    'docs/releases/BASELINE_OFICIAL.md',
    'README.md',
    'docs/operations/OPERACAO_LOCAL.md',
    'docs/releases/RELEASE_OPERACIONAL.md',
]


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding='utf-8')


def extract_footer_version(text: str) -> str | None:
    match = re.search(r'versao\s*:\s*str\s*=\s*"(v\d+)"', text)
    return match.group(1) if match else None


def extract_changelog_version(text: str) -> str | None:
    match = re.search(r'^##\s+(v\d+)\s+—', text, flags=re.MULTILINE)
    return match.group(1) if match else None


def extract_baseline_version(text: str) -> str | None:
    match = re.search(r'baseline oficial vigente desta base é\s+\*\*(v\d+)\*\*', text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []

    for rel_path in REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            errors.append(f'Arquivo obrigatório ausente para metadados de release: {rel_path}')
        else:
            notes.append(f'OK artefato presente: {rel_path}')

    footer = extract_footer_version(read_text('ui/primitives.py')) if (ROOT / 'ui/primitives.py').exists() else None
    changelog = extract_changelog_version(read_text('CHANGELOG.md')) if (ROOT / 'CHANGELOG.md').exists() else None
    baseline = extract_baseline_version(read_text('docs/releases/BASELINE_OFICIAL.md')) if (ROOT / 'docs/releases/BASELINE_OFICIAL.md').exists() else None

    if not footer:
        errors.append('Não foi possível identificar a versão do rodapé em ui/primitives.py')
    else:
        notes.append(f'OK versão do rodapé detectada: {footer}')

    if not changelog:
        errors.append('Não foi possível identificar a versão mais recente em CHANGELOG.md')
    else:
        notes.append(f'OK versão mais recente no changelog: {changelog}')

    if not baseline:
        errors.append('Não foi possível identificar a versão oficial em docs/releases/BASELINE_OFICIAL.md')
    else:
        notes.append(f'OK versão oficial na baseline: {baseline}')

    versions = {
        'rodapé': footer,
        'changelog': changelog,
        'baseline': baseline,
    }
    non_null_versions = {value for value in versions.values() if value}
    if len(non_null_versions) > 1:
        details = ', '.join(f'{name}={value}' for name, value in versions.items())
        errors.append('Divergência entre metadados de release: ' + details)
    elif len(non_null_versions) == 1:
        version = next(iter(non_null_versions))
        notes.append(f'OK metadados sincronizados em {version}')

    command = 'python scripts/quality/release_metadata_guard.py'
    for rel_path in ['README.md', 'docs/operations/OPERACAO_LOCAL.md', 'docs/releases/RELEASE_OPERACIONAL.md']:
        text = read_text(rel_path) if (ROOT / rel_path).exists() else ''
        if command not in text:
            errors.append(f'{rel_path} deve orientar o uso de {command}')
        else:
            notes.append(f'OK documentação cita {command}: {rel_path}')

    print('=== RELEASE METADATA GUARD | Sorteador Pelada PRO ===')
    for note in notes:
        print(f'[OK] {note}')

    if errors:
        print('\nErros encontrados:')
        for error in errors:
            print(f' - {error}')
        return 1

    print('\nMetadados de release sincronizados.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
