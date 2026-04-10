#!/usr/bin/env python3
"""Wrapper temporário de compatibilidade para o guard de higiene de artefatos.

Padrão oficial atual:
    python scripts/quality/release_artifacts_hygiene_guard.py

Este wrapper histórico permanece disponível apenas como compatibilidade temporária.
"""

from scripts.quality.release_artifacts_hygiene_guard import main


if __name__ == "__main__":
    raise SystemExit(main())
