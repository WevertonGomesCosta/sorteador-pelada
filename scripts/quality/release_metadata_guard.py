#!/usr/bin/env python3
"""Guard leve de consistência de metadados automáticos do app.

Valida se o rodapé passou a usar metadados derivados do Git:
- versão automática baseada na contagem de commits;
- data automática baseada na data do último commit;
- fallbacks explícitos para ambientes sem Git disponível.

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


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []

    for rel_path in REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            errors.append(f'Arquivo obrigatório ausente para metadados de release: {rel_path}')
        else:
            notes.append(f'OK artefato presente: {rel_path}')

    primitives = read_text('ui/primitives.py') if (ROOT / 'ui/primitives.py').exists() else ''

    required_markers = {
        'helper de versão automática': '_versao_atual_projeto',
        'helper de data automática': '_data_ultima_atualizacao_projeto',
        'contagem Git de commits': 'rev-list", "--count", "HEAD"',
        'data Git do último commit': 'log", "-1", "--format=%cs", "HEAD"',
        'fallback de versão': 'APP_VERSION_FALLBACK',
        'fallback de data': 'APP_LAST_UPDATED_FALLBACK',
        'override de versão por ambiente': 'SORTEADOR_APP_VERSION',
        'override de data por ambiente': 'SORTEADOR_APP_LAST_UPDATED',
    }

    for description, marker in required_markers.items():
        if marker not in primitives:
            errors.append(f'Rodapé sem {description}: marcador esperado {marker!r}')
        else:
            notes.append(f'OK rodapé contém {description}')

    if re.search(r'^APP_VERSION\s*=\s*"v\d+"', primitives, flags=re.MULTILINE):
        errors.append('Rodapé ainda define APP_VERSION estático; use APP_VERSION_FALLBACK + Git')
    else:
        notes.append('OK sem APP_VERSION estático no rodapé')

    if re.search(r'versao\s*:\s*str\s*=\s*"v\d+"', primitives):
        errors.append('render_app_meta_footer ainda possui versão fixa no argumento padrão')
    else:
        notes.append('OK render_app_meta_footer sem versão fixa como padrão')

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

    print('\nMetadados automáticos de release configurados.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
