"""Estado e resumos neutros dos critérios ativos do sorteio."""

from __future__ import annotations

import streamlit as st


def obter_criterios_ativos() -> dict:
    return {
        "pos": bool(st.session_state.get("criterio_posicao", True)),
        "nota": bool(st.session_state.get("criterio_nota", True)),
        "vel": bool(st.session_state.get("criterio_velocidade", True)),
        "mov": bool(st.session_state.get("criterio_movimentacao", True)),
    }



def resumo_criterios_ativos() -> str:
    criterios = obter_criterios_ativos()
    ativos = []

    if criterios["pos"]:
        ativos.append("Posição")
    if criterios["nota"]:
        ativos.append("Nota")
    if criterios["vel"]:
        ativos.append("Velocidade")
    if criterios["mov"]:
        ativos.append("Movimentação")

    if len(ativos) == 4:
        return "Padrão · Posição, Nota, Velocidade e Movimentação"
    if not ativos:
        return "Personalizado · Nenhum critério ativo"

    return "Personalizado · " + ", ".join(ativos)
