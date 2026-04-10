#!/usr/bin/env python3
"""Wrapper temporário de compatibilidade para o caminho histórico `scripts/script_cli_contract_guard.py`.

Padrão oficial atual: `scripts/quality/script_cli_contract_guard.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality.script_cli_contract_guard import main


if __name__ == "__main__":
    raise SystemExit(main())
