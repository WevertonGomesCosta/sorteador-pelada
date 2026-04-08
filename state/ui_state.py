"""Helpers de estado local da interface."""

import streamlit as st

import state.keys as K

def ensure_local_session_state():
    if K.BASE_ADMIN_CARREGADA not in st.session_state:
        st.session_state[K.BASE_ADMIN_CARREGADA] = False
    if K.BASE_INCONSISTENCIAS_CARREGAMENTO not in st.session_state:
        st.session_state[K.BASE_INCONSISTENCIAS_CARREGAMENTO] = {}
    if K.BASE_REGISTROS_INCONSISTENTES_CARREGAMENTO not in st.session_state:
        st.session_state[K.BASE_REGISTROS_INCONSISTENTES_CARREGAMENTO] = []
    if K.SENHA_ADMIN_CONFIRMADA not in st.session_state:
        st.session_state[K.SENHA_ADMIN_CONFIRMADA] = False
    if K.ULTIMA_SENHA_DIGITADA not in st.session_state:
        st.session_state[K.ULTIMA_SENHA_DIGITADA] = ""
    if K.QTD_JOGADORES_ADICIONADOS_MANUALMENTE not in st.session_state:
        st.session_state[K.QTD_JOGADORES_ADICIONADOS_MANUALMENTE] = 0
    if K.CADASTRO_MANUAL_EXPANDED not in st.session_state:
        st.session_state[K.CADASTRO_MANUAL_EXPANDED] = False
    if K.CADASTRO_MANUAL_NOME_EXISTENTE not in st.session_state:
        st.session_state[K.CADASTRO_MANUAL_NOME_EXISTENTE] = ""
    if K.CRITERIO_POSICAO not in st.session_state:
        st.session_state[K.CRITERIO_POSICAO] = True
    if K.CRITERIO_NOTA not in st.session_state:
        st.session_state[K.CRITERIO_NOTA] = True
    if K.CRITERIO_VELOCIDADE not in st.session_state:
        st.session_state[K.CRITERIO_VELOCIDADE] = True
    if K.CRITERIO_MOVIMENTACAO not in st.session_state:
        st.session_state[K.CRITERIO_MOVIMENTACAO] = True
    if K.SCROLL_PARA_RESULTADO not in st.session_state:
        st.session_state[K.SCROLL_PARA_RESULTADO] = False
    if K.SCROLL_PARA_LISTA not in st.session_state:
        st.session_state[K.SCROLL_PARA_LISTA] = False
    if K.SCROLL_PARA_REVISAO not in st.session_state:
        st.session_state[K.SCROLL_PARA_REVISAO] = False
    if K.SCROLL_DESTINO_REVISAO not in st.session_state:
        st.session_state[K.SCROLL_DESTINO_REVISAO] = "top"
    if K.SCROLL_PARA_SORTEIO not in st.session_state:
        st.session_state[K.SCROLL_PARA_SORTEIO] = False
    if K.SCROLL_PARA_CONFIRMAR_SENHA not in st.session_state:
        st.session_state[K.SCROLL_PARA_CONFIRMAR_SENHA] = False
    if K.RESULTADO_ASSINATURA not in st.session_state:
        st.session_state[K.RESULTADO_ASSINATURA] = None
    if K.RESULTADO_INVALIDADO_MSG not in st.session_state:
        st.session_state[K.RESULTADO_INVALIDADO_MSG] = False
    if K.MANUAL_SECTION_VISIBLE not in st.session_state:
        st.session_state[K.MANUAL_SECTION_VISIBLE] = False


def abrir_expander_cadastro_manual():
    st.session_state[K.CADASTRO_MANUAL_EXPANDED] = True
    st.session_state[K.MANUAL_SECTION_VISIBLE] = True