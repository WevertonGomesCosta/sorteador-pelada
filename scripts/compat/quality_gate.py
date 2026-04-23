#!/usr/bin/env python3
"""Wrapper temporário de compatibilidade para o caminho histórico `scripts/compat/quality_gate.py`.

Padrão oficial atual: `scripts/quality/quality_gate.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality.quality_gate import main


if __name__ == "__main__":
    raise SystemExit(main())
