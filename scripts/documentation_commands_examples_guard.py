#!/usr/bin/env python3
"""Wrapper temporário de compatibilidade para o guard de exemplos de comandos na documentação.

Padrão oficial atual: `scripts/quality/documentation_commands_examples_guard.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality.documentation_commands_examples_guard import main


if __name__ == '__main__':
    raise SystemExit(main())
