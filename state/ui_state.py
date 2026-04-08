"""Helpers de estado local da interface."""

import streamlit as st


def ensure_local_session_state():
    if "base_admin_carregada" not in st.session_state:
        st.session_state.base_admin_carregada = False
    if "base_inconsistencias_carregamento" not in st.session_state:
        st.session_state.base_inconsistencias_carregamento = {}
    if "base_registros_inconsistentes_carregamento" not in st.session_state:
        st.session_state.base_registros_inconsistentes_carregamento = []
    if "senha_admin_confirmada" not in st.session_state:
        st.session_state.senha_admin_confirmada = False
    if "ultima_senha_digitada" not in st.session_state:
        st.session_state.ultima_senha_digitada = ""
    if "qtd_jogadores_adicionados_manualmente" not in st.session_state:
        st.session_state.qtd_jogadores_adicionados_manualmente = 0
    if "cadastro_manual_expanded" not in st.session_state:
        st.session_state.cadastro_manual_expanded = False
    if "cadastro_manual_nome_existente" not in st.session_state:
        st.session_state.cadastro_manual_nome_existente = ""
    if "criterio_posicao" not in st.session_state:
        st.session_state.criterio_posicao = True
    if "criterio_nota" not in st.session_state:
        st.session_state.criterio_nota = True
    if "criterio_velocidade" not in st.session_state:
        st.session_state.criterio_velocidade = True
    if "criterio_movimentacao" not in st.session_state:
        st.session_state.criterio_movimentacao = True
    if "scroll_para_resultado" not in st.session_state:
        st.session_state.scroll_para_resultado = False
    if "scroll_para_lista" not in st.session_state:
        st.session_state.scroll_para_lista = False
    if "scroll_para_revisao" not in st.session_state:
        st.session_state.scroll_para_revisao = False
    if "scroll_destino_revisao" not in st.session_state:
        st.session_state.scroll_destino_revisao = "top"
    if "scroll_para_sorteio" not in st.session_state:
        st.session_state.scroll_para_sorteio = False
    if "scroll_para_confirmar_senha" not in st.session_state:
        st.session_state.scroll_para_confirmar_senha = False
    if "resultado_assinatura" not in st.session_state:
        st.session_state.resultado_assinatura = None
    if "resultado_invalidado_msg" not in st.session_state:
        st.session_state.resultado_invalidado_msg = False
    if "manual_section_visible" not in st.session_state:
        st.session_state.manual_section_visible = False


def abrir_expander_cadastro_manual():
    st.session_state.cadastro_manual_expanded = True
    st.session_state.manual_section_visible = True
