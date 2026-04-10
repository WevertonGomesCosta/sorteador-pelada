#!/usr/bin/env python3
"""Runner do smoke test funcional mínimo da base.

Objetivo:
- complementar a validação estrutural de ``scripts/check_base.py``;
- validar cenários leves e seguros dos módulos neutros do app;
- evitar testes interativos de Streamlit ou fluxos congelados.

Uso:
    python scripts/smoke_test_base.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_smoke_base.py")
    resultado = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if resultado.wasSuccessful() else 1)
