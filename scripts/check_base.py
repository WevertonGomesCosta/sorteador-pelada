#!/usr/bin/env python3
"""Wrapper temporário de compatibilidade para o caminho histórico `scripts/check_base.py`.

Padrão oficial atual: `scripts/quality/check_base.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality.check_base import main


if __name__ == "__main__":
    raise SystemExit(main())
