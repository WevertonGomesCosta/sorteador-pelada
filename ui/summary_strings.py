"""Resumos e textos auxiliares da interface do Sorteador Pelada PRO."""

from __future__ import annotations

import streamlit as st

from state.criteria_state import obter_criterios_ativos
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



def _qtd_criterios_ativos() -> int:
    return sum(obter_criterios_ativos().values())



def _criterios_estao_no_padrao() -> bool:
    criterios = obter_criterios_ativos()
    return (
        criterios["pos"],
        criterios["nota"],
        criterios["vel"],
        criterios["mov"],
    ) == (True, True, True, True)



def _garantir_parametros_opcionais() -> None:
    if "sortear_capitao" not in st.session_state:
        st.session_state["sortear_capitao"] = False
    if "sortear_goleiros" not in st.session_state:
        st.session_state["sortear_goleiros"] = False



def render_parametro_goleiros_pre_revisao(qtd_goleiros_lidos: int, n_times: int) -> None:
    _garantir_parametros_opcionais()
    if not hasattr(st, "checkbox") or not hasattr(st, "caption"):
        return

    qtd_goleiros_lidos = int(qtd_goleiros_lidos or 0)
    n_times = int(n_times or 0)
    goleiros_compativeis = qtd_goleiros_lidos > 0 and qtd_goleiros_lidos == n_times

    if not goleiros_compativeis:
        st.session_state["sortear_goleiros"] = False
        if qtd_goleiros_lidos > 0:
            st.caption(
                f"Goleiros detectados: {qtd_goleiros_lidos}. A opção de incluir goleiros aparece apenas quando a quantidade de goleiros é igual ao número de times."
            )
        return

    st.checkbox("Incluir goleiros no sorteio", key="sortear_goleiros")
    st.caption(
        f"Foram detectados {qtd_goleiros_lidos} goleiro(s) para {n_times} time(s). Se ativo, eles entram na revisão e podem receber nota, posição G, velocidade e movimentação."
    )



def render_parametro_capitao_pos_confirmacao() -> None:
    _garantir_parametros_opcionais()
    if not hasattr(st, "checkbox") or not hasattr(st, "caption"):
        return

    st.checkbox("Sortear Capitão", key="sortear_capitao")
    st.caption("Quando ativo, o app sorteia aleatoriamente um capitão para cada time após montar os times.")



def render_parametros_opcionais_pre_revisao() -> None:
    _garantir_parametros_opcionais()


def resumo_expander_criterios() -> str:
    status = "Padrão" if _criterios_estao_no_padrao() else "Personalizado"
    return _titulo_expander("⚙️ Critérios", status)
