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

    if 'diagnostico_lista' not in st.session_state:
        st.session_state.diagnostico_lista = None

    if 'lista_revisada' not in st.session_state:
        st.session_state.lista_revisada = None

    if 'lista_revisada_confirmada' not in st.session_state:
        st.session_state.lista_revisada_confirmada = False

    if 'lista_texto_revisado' not in st.session_state:
        st.session_state.lista_texto_revisado = ""

    if 'revisao_lista_expandida' not in st.session_state:
        st.session_state.revisao_lista_expandida = False

    if "faltantes_revisao" not in st.session_state:
        st.session_state.faltantes_revisao = []

    if "cadastro_guiado_ativo" not in st.session_state:
        st.session_state.cadastro_guiado_ativo = False

    if "faltantes_cadastrados_na_rodada" not in st.session_state:
        st.session_state.faltantes_cadastrados_na_rodada = []

    if "revisao_pendente_pos_cadastro" not in st.session_state:
        st.session_state.revisao_pendente_pos_cadastro = False
