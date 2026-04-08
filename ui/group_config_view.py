"""Etapa de entrada e configuração inicial do Sorteador Pelada PRO."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

import state.keys as K

from state.session import registrar_base_carregada_no_estado
from ui.summary_strings import resumo_expander_configuracao



def abrir_expander_grupo():
    st.session_state[K.GRUPO_CONFIG_EXPANDED] = True



def grupo_config_deve_abrir() -> bool:
    if K.GRUPO_CONFIG_EXPANDED not in st.session_state:
        st.session_state[K.GRUPO_CONFIG_EXPANDED] = True
    return bool(
        st.session_state.get(K.GRUPO_CONFIG_EXPANDED, True)
        or str(st.session_state.get(K.GRUPO_NOME_PELADA, "")).strip()
        or str(st.session_state.get(K.GRUPO_SENHA_ADMIN, "")).strip()
        or st.session_state.get(K.SENHA_ADMIN_CONFIRMADA, False)
    )



def ativar_fluxo_somente_lista(logic):
    st.session_state[K.DF_BASE] = logic.criar_base_vazia()
    st.session_state[K.NOVOS_JOGADORES] = []
    st.session_state[K.IS_ADMIN] = False
    st.session_state[K.BASE_ADMIN_CARREGADA] = False
    st.session_state[K.ULTIMO_ARQUIVO] = None
    st.session_state[K.QTD_JOGADORES_ADICIONADOS_MANUALMENTE] = 0
    st.session_state[K.SENHA_ADMIN_CONFIRMADA] = False
    st.session_state[K.BASE_INCONSISTENCIAS_CARREGAMENTO] = {}
    st.session_state[K.BASE_REGISTROS_INCONSISTENTES_CARREGAMENTO] = []
    st.session_state[K.GRUPO_NOME_PELADA] = ""
    st.session_state[K.GRUPO_SENHA_ADMIN] = ""
    st.session_state[K.GRUPO_ORIGEM_FLUXO] = "lista"
    st.session_state[K.GRUPO_CONFIG_EXPANDED] = False
    st.session_state[K.SCROLL_PARA_LISTA] = True
    st.session_state[K.LISTA_REVISADA_CONFIRMADA] = False
    st.session_state[K.LISTA_REVISADA] = None
    st.session_state[K.DIAGNOSTICO_LISTA] = None
    st.rerun()



def render_group_config_expander(logic, nome_pelada_adm: str, senha_adm: str) -> str:
    st.session_state.setdefault(K.GRUPO_CONFIG_EXPANDED, True)
    st.session_state.setdefault(K.GRUPO_ORIGEM_FLUXO, None)
    st.session_state.setdefault(K.GRUPO_BUSCA_STATUS, "idle")
    st.session_state.setdefault(K.GRUPO_NOME_ULTIMA_BUSCA, "")

    with st.expander(
        resumo_expander_configuracao(nome_pelada_adm),
        expanded=grupo_config_deve_abrir(),
    ):
        st.markdown("**Como deseja iniciar o sorteio?**")
        col_lista, col_admin, col_excel = st.columns(3)
        with col_lista:
            if st.button("🎲 Apenas sorteio com lista", key="grupo_escolher_lista"):
                ativar_fluxo_somente_lista(logic)
        with col_admin:
            if st.button("🗂️ Carregar base do grupo", key="grupo_escolher_admin"):
                st.session_state[K.GRUPO_ORIGEM_FLUXO] = "admin"
                st.session_state[K.GRUPO_CONFIG_EXPANDED] = True
                st.rerun()
        with col_excel:
            if st.button("📄 Usar Excel próprio", key="grupo_escolher_excel"):
                st.session_state[K.GRUPO_ORIGEM_FLUXO] = "excel"
                st.session_state[K.GRUPO_CONFIG_EXPANDED] = True
                st.rerun()

        if K.GRUPO_NOME_PELADA_PENDING in st.session_state:
            st.session_state[K.GRUPO_NOME_PELADA] = st.session_state.pop(K.GRUPO_NOME_PELADA_PENDING)
        if K.GRUPO_SENHA_ADMIN_PENDING in st.session_state:
            st.session_state[K.GRUPO_SENHA_ADMIN] = st.session_state.pop(K.GRUPO_SENHA_ADMIN_PENDING)

        origem_fluxo = st.session_state.get(K.GRUPO_ORIGEM_FLUXO)
        nome_pelada = str(st.session_state.get(K.GRUPO_NOME_PELADA, "")).strip()
        uploaded_file = None
        base_grupo_carregada = bool(st.session_state[K.BASE_ADMIN_CARREGADA] and st.session_state[K.IS_ADMIN])

        if origem_fluxo == "admin":
            st.markdown("---")
            st.markdown("**🗂️ Carregar base do grupo**")
            nome_digitado = st.text_input(
                "Nome da pelada:",
                placeholder="Ex: Pelada de Domingo",
                key="grupo_nome_pelada",
            ).strip()
            nome_pelada = nome_digitado

            ultima_busca = str(st.session_state.get(K.GRUPO_NOME_ULTIMA_BUSCA, "")).strip()
            busca_status = st.session_state.get(K.GRUPO_BUSCA_STATUS, "idle")
            if nome_digitado != ultima_busca and not base_grupo_carregada:
                busca_status = "idle"
                st.session_state[K.GRUPO_BUSCA_STATUS] = "idle"
                st.session_state[K.SENHA_ADMIN_CONFIRMADA] = False

            if st.button("🔎 Buscar grupo", key="grupo_buscar_nome"):
                st.session_state[K.GRUPO_NOME_ULTIMA_BUSCA] = nome_digitado
                if nome_digitado and nome_digitado.upper() == str(nome_pelada_adm).upper():
                    st.session_state[K.GRUPO_BUSCA_STATUS] = "found"
                    st.session_state[K.SCROLL_PARA_CONFIRMAR_SENHA] = True
                else:
                    st.session_state[K.GRUPO_BUSCA_STATUS] = "not_found" if nome_digitado else "idle"
                st.rerun()

            busca_status = st.session_state.get(K.GRUPO_BUSCA_STATUS, "idle")
            senha_atual = st.session_state.get(K.GRUPO_SENHA_ADMIN, "")
            if st.session_state[K.ULTIMA_SENHA_DIGITADA] != senha_atual:
                st.session_state[K.SENHA_ADMIN_CONFIRMADA] = False
                st.session_state[K.ULTIMA_SENHA_DIGITADA] = senha_atual

            if base_grupo_carregada:
                st.success("Base do grupo carregada com sucesso.")
            elif busca_status == "found":
                st.success("Base encontrada para esse grupo.")
                st.caption("Informe a senha para carregar a base.")
            elif busca_status == "not_found":
                st.warning("Grupo não encontrado. Confira o nome informado ou escolha a opção de Excel próprio.")
            else:
                st.info("Informe o nome da pelada e clique em **Buscar grupo** para localizar a base.")

            if busca_status == "found" and not base_grupo_carregada:
                st.markdown('<div id="confirmar-senha-anchor"></div>', unsafe_allow_html=True)
                if st.session_state.get(K.SCROLL_PARA_CONFIRMAR_SENHA, False):
                    components.html(
                        """
import state.keys as K
                        <script>
                        const parentDoc = window.parent.document;
                        const anchor = parentDoc.getElementById("confirmar-senha-anchor");
                        if (anchor) {
                            anchor.scrollIntoView({ behavior: "smooth", block: "center" });
                        }
                        </script>
                        """,
                        height=0,
                    )
                    st.session_state[K.SCROLL_PARA_CONFIRMAR_SENHA] = False
                senha = st.text_input(
                    "Senha:",
                    type="password",
                    key="grupo_senha_admin",
                )
                if st.button(
                    "📥 Carregar base de dados",
                    key="grupo_confirmar_senha",
                ):
                    if senha != str(senha_adm):
                        st.session_state[K.SENHA_ADMIN_CONFIRMADA] = False
                        st.session_state[K.ULTIMA_SENHA_DIGITADA] = senha
                        st.session_state[K.IS_ADMIN] = False
                        st.session_state[K.BASE_ADMIN_CARREGADA] = False
                        st.error("Senha incorreta")
                    else:
                        st.session_state[K.SENHA_ADMIN_CONFIRMADA] = True
                        st.session_state[K.ULTIMA_SENHA_DIGITADA] = senha
                        registrar_base_carregada_no_estado(
                            logic,
                            logic.carregar_dados_originais(),
                            is_admin=True,
                            ultimo_arquivo=None,
                        )
                        st.session_state[K.GRUPO_CONFIG_EXPANDED] = False
                        st.success(f"Base carregada: {len(st.session_state[K.DF_BASE])} jogadores.")
                        st.rerun()

        elif origem_fluxo == "excel":
            st.markdown("---")
            st.markdown("**📄 Usar Excel próprio**")
            st.caption("Envie sua planilha e depois toque em **Carregar base de dados**.")
            uploaded_file = st.file_uploader(
                "Enviar planilha Excel",
                type=["xlsx"],
                label_visibility="collapsed",
                key="grupo_upload_planilha",
            )

            if st.button(
                "📥 Carregar base de dados",
                key="grupo_carregar_base",
            ):
                if uploaded_file is None:
                    st.info("Você ainda não selecionou uma planilha própria para carregar.")
                else:
                    df_novo = logic.processar_upload(uploaded_file)
                    if df_novo is not None:
                        registrar_base_carregada_no_estado(
                            logic,
                            df_novo,
                            is_admin=False,
                            ultimo_arquivo=uploaded_file.name,
                        )
                        st.session_state[K.SENHA_ADMIN_CONFIRMADA] = False
                        st.session_state[K.GRUPO_CONFIG_EXPANDED] = False
                        st.success("Arquivo carregado!")
                        st.rerun()
        elif origem_fluxo == "lista":
            st.markdown("---")
            st.markdown("**🎲 Apenas sorteio com lista**")
            st.info("Neste modo, você não precisa carregar base nem Excel. O app usará apenas os nomes informados na lista para um sorteio aleatório.")
            st.caption("Você será levado diretamente para a seção da lista. Se houver nomes repetidos, revise e corrija cada ocorrência antes do sorteio.")
        else:
            st.caption("Escolha uma opção para começar: usar a base do grupo, enviar um Excel próprio ou seguir direto para o sorteio apenas com lista.")

        if (
            not st.session_state[K.DF_BASE].empty
            or st.session_state[K.NOVOS_JOGADORES]
            or st.session_state[K.IS_ADMIN]
        ):
            with st.expander("Ações secundárias", expanded=False):
                st.caption("Use a limpeza apenas quando quiser reiniciar a base atual.")
                if st.button("🗑 Limpar base atual", key="grupo_limpar_base_atual"):
                    st.session_state[K.DF_BASE] = logic.criar_base_vazia()
                    st.session_state[K.NOVOS_JOGADORES] = []
                    st.session_state[K.IS_ADMIN] = False
                    st.session_state[K.BASE_ADMIN_CARREGADA] = False
                    st.session_state[K.ULTIMO_ARQUIVO] = None
                    st.session_state[K.QTD_JOGADORES_ADICIONADOS_MANUALMENTE] = 0
                    st.session_state[K.SENHA_ADMIN_CONFIRMADA] = False
                    st.session_state[K.BASE_INCONSISTENCIAS_CARREGAMENTO] = {}
                    st.session_state[K.BASE_REGISTROS_INCONSISTENTES_CARREGAMENTO] = []
                    st.session_state[K.GRUPO_BUSCA_STATUS] = "idle"
                    st.session_state[K.GRUPO_NOME_ULTIMA_BUSCA] = ""
                    st.session_state[K.GRUPO_ORIGEM_FLUXO] = None
                    st.session_state[K.GRUPO_CONFIG_EXPANDED] = True
                    st.session_state[K.GRUPO_NOME_PELADA_PENDING] = ""
                    st.session_state[K.GRUPO_SENHA_ADMIN_PENDING] = ""
                    st.rerun()

    return nome_pelada