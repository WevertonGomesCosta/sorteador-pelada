from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

sys.modules.setdefault("streamlit", SimpleNamespace(session_state={}))

import core.flow_guard as flow_guard
import state.keys as K
from core import validators
from core.base_summary import resumo_inconsistencias_base, total_inconsistencias_base
from core.flow_guard import (
    construir_assinatura_entrada_sorteio,
    construir_gate_pre_sorteio,
    contar_duplicados_base_atual,
    invalidar_resultado_se_entrada_mudou,
)
from state import criteria_state, view_models
from ui import panels, summary_strings


class DummyLogic:
    def __init__(self, jogadores: list[str], inconsistencias: dict | None = None) -> None:
        self._jogadores = jogadores
        self._inconsistencias = inconsistencias or {}

    def processar_lista(self, lista_texto: str, return_metadata: bool = True, emit_warning: bool = False) -> dict:
        return {"jogadores": list(self._jogadores)}

    def diagnosticar_inconsistencias_base(self, df_base: pd.DataFrame) -> dict:
        return dict(self._inconsistencias)
