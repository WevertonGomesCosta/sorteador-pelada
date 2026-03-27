import streamlit as st


def init_session_state(logic):
    if 'df_base' not in st.session_state:
        st.session_state.df_base = logic.criar_base_vazia()

    if 'novos_jogadores' not in st.session_state:
        st.session_state.novos_jogadores = []

    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False

    if 'aviso_sem_planilha' not in st.session_state:
        st.session_state.aviso_sem_planilha = False
