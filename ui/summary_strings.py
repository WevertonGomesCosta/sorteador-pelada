"""Resumos e textos auxiliares da interface do Sorteador Pelada PRO."""

from __future__ import annotations

import streamlit as st

from ui.primitives import _titulo_expander


def resumo_expander_configuracao(nome_pelada_adm: str) -> str:
    base_admin_carregada = bool(
        st.session_state.get("base_admin_carregada", False)
        and st.session_state.get("is_admin", False)
    )
    base_upload_carregada = bool(st.session_state.get("ultimo_arquivo")) and not st.session_state.get("is_admin", False)
    fluxo_lista = st.session_state.get("grupo_origem_fluxo") == "lista"
    busca_status = st.session_state.get("grupo_busca_status", "idle")

    if base_admin_carregada:
        status = "Base do grupo carregada"
    elif base_upload_carregada:
        status = "Excel próprio carregado"
    elif fluxo_lista:
        status = "Somente lista"
    elif busca_status == "found":
        status = "Grupo encontrado"
    elif busca_status == "not_found":
        status = "Grupo não encontrado"
    else:
        status = "Sem base"

    return _titulo_expander("⚙️ Grupo e base", status)



def _qtd_adicoes_manuais() -> int:
    return int(st.session_state.get("qtd_jogadores_adicionados_manualmente", 0))



def resumo_expander_cadastro_manual() -> str:
    cadastro_guiado_ativo = bool(st.session_state.get("cadastro_guiado_ativo", False))
    cadastro_guiado_concluido = bool(
        st.session_state.get("revisao_pendente_pos_cadastro", False)
        and len(st.session_state.get("faltantes_cadastrados_na_rodada", [])) > 0
        and not cadastro_guiado_ativo
    )
    qtd_manual = _qtd_adicoes_manuais()

    if cadastro_guiado_ativo:
        status = "Cadastro guiado ativo"
    elif cadastro_guiado_concluido:
        status = "Faltantes cadastrados"
    elif qtd_manual > 0:
        status = f"{qtd_manual} adicionados"
    else:
        status = "Opcional"

    return _titulo_expander("📝 Cadastro manual", status)



def obter_criterios_ativos() -> dict:
    return {
        "pos": bool(st.session_state.get("criterio_posicao", True)),
        "nota": bool(st.session_state.get("criterio_nota", True)),
        "vel": bool(st.session_state.get("criterio_velocidade", True)),
        "mov": bool(st.session_state.get("criterio_movimentacao", True)),
    }



def _criterios_estao_no_padrao() -> bool:
    criterios = obter_criterios_ativos()
    return (
        criterios["pos"],
        criterios["nota"],
        criterios["vel"],
        criterios["mov"],
    ) == (True, True, True, True)



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



def resumo_expander_criterios() -> str:
    status = "Padrão" if _criterios_estao_no_padrao() else "Personalizado"
    return _titulo_expander("⚙️ Critérios", status)
