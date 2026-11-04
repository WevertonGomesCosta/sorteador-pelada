"""Agregador de compatibilidade do smoke test leve.

Mantém o caminho histórico `tests/test_smoke_base.py` apenas como agregador
de compatibilidade, enquanto a suíte canônica oficial permanece distribuída
em módulos menores dentro de `tests/`.
"""

from tests.test_core_smoke import *  # noqa: F401,F403
from tests.test_state_smoke import *  # noqa: F401,F403
from tests.test_ui_safe_smoke import *  # noqa: F401,F403
from tests.test_scripts_smoke import *  # noqa: F401,F403

import unittest


def load_tests(
    loader: unittest.TestLoader,
    standard_tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Evita duplicacao na descoberta; preserva a execucao direta do alias.

    O runner canonico descobre os quatro modulos diretamente. Quando o modulo
    historico e invocado isoladamente, conserva toda a suite agregada.
    """
    if pattern is not None:
        return unittest.TestSuite()
    return standard_tests
