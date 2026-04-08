"""Seções de configuração inicial do Sorteador Pelada PRO."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from state.session import registrar_base_carregada_no_estado
from ui.summary_strings import resumo_expander_configuracao

def abrir_expander_grupo():
    st.session_state.grupo_config_expanded = True


def grupo_config_deve_abrir() -> bool:
    if "grupo_config_expanded" not in st.session_state:
        st.session_state.grupo_config_expanded = True
    return bool(
        st.session_state.get("grupo_config_expanded", True)
        or str(st.session_state.get("grupo_nome_pelada", "")).strip()
        or str(st.session_state.get("grupo_senha_admin", "")).strip()
        or st.session_state.get("senha_admin_confirmada", False)
    )


def ativar_fluxo_somente_lista(logic):
    st.session_state.df_base = logic.criar_base_vazia()
    st.session_state.novos_jogadores = []
    st.session_state.is_admin = False
    st.session_state.base_admin_carregada = False
    st.session_state.ultimo_arquivo = None
    st.session_state.qtd_jogadores_adicionados_manualmente = 0
    st.session_state.senha_admin_confirmada = False
    st.session_state.base_inconsistencias_carregamento = {}
    st.session_state.base_registros_inconsistentes_carregamento = []
    st.session_state.grupo_nome_pelada = ""
    st.session_state.grupo_senha_admin = ""
    st.session_state.grupo_origem_fluxo = "lista"
    st.session_state.grupo_config_expanded = False
    st.session_state.scroll_para_lista = True
    st.session_state.lista_revisada_confirmada = False
    st.session_state.lista_revisada = None
    st.session_state.diagnostico_lista = None
    st.rerun()


def render_group_config_expander(logic, nome_pelada_adm: str, senha_adm: str) -> str:
    st.session_state.setdefault("grupo_config_expanded", True)
    st.session_state.setdefault("grupo_origem_fluxo", None)
    st.session_state.setdefault("grupo_busca_status", "idle")
    st.session_state.setdefault("grupo_nome_ultima_busca", "")

    with st.expander(
        resumo_expander_configuracao(nome_pelada_adm),
        expanded=True,
    ):
        st.markdown("**Como deseja iniciar o sorteio?**")
        col_lista, col_admin, col_excel = st.columns(3)
        with col_lista:
            if st.button("🎲 Apenas sorteio com lista", key="grupo_escolher_lista"):
                ativar_fluxo_somente_lista(logic)
        with col_admin:
            if st.button("🗂️ Carregar base do grupo", key="grupo_escolher_admin"):
                st.session_state.grupo_origem_fluxo = "admin"
                st.session_state.grupo_config_expanded = True
                st.rerun()
        with col_excel:
            if st.button("📄 Usar Excel próprio", key="grupo_escolher_excel"):
                st.session_state.grupo_origem_fluxo = "excel"
                st.session_state.grupo_config_expanded = True
                st.rerun()

        if "grupo_nome_pelada__pending" in st.session_state:
            st.session_state["grupo_nome_pelada"] = st.session_state.pop("grupo_nome_pelada__pending")
        if "grupo_senha_admin__pending" in st.session_state:
            st.session_state["grupo_senha_admin"] = st.session_state.pop("grupo_senha_admin__pending")

        origem_fluxo = st.session_state.get("grupo_origem_fluxo")
        nome_pelada = str(st.session_state.get("grupo_nome_pelada", "")).strip()
        uploaded_file = None
        base_grupo_carregada = bool(st.session_state.base_admin_carregada and st.session_state.is_admin)

        if origem_fluxo == "admin":
            st.markdown("---")
            st.markdown("**🗂️ Carregar base do grupo**")
            nome_digitado = st.text_input(
                "Nome da pelada:",
                placeholder="Ex: Pelada de Domingo",
                key="grupo_nome_pelada",
            ).strip()
            nome_pelada = nome_digitado

            ultima_busca = str(st.session_state.get("grupo_nome_ultima_busca", "")).strip()
            busca_status = st.session_state.get("grupo_busca_status", "idle")
            if nome_digitado != ultima_busca and not base_grupo_carregada:
                busca_status = "idle"
                st.session_state.grupo_busca_status = "idle"
                st.session_state.senha_admin_confirmada = False

            if st.button("🔎 Buscar grupo", key="grupo_buscar_nome"):
                st.session_state.grupo_nome_ultima_busca = nome_digitado
                if nome_digitado and nome_digitado.upper() == str(nome_pelada_adm).upper():
                    st.session_state.grupo_busca_status = "found"
                    st.session_state.scroll_para_confirmar_senha = True
                else:
                    st.session_state.grupo_busca_status = "not_found" if nome_digitado else "idle"
                st.rerun()

            busca_status = st.session_state.get("grupo_busca_status", "idle")
            senha_atual = st.session_state.get("grupo_senha_admin", "")
            if st.session_state.ultima_senha_digitada != senha_atual:
                st.session_state.senha_admin_confirmada = False
                st.session_state.ultima_senha_digitada = senha_atual

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
                if st.session_state.get("scroll_para_confirmar_senha", False):
                    components.html(
                        """
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
                    st.session_state.scroll_para_confirmar_senha = False
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
                        st.session_state.senha_admin_confirmada = False
                        st.session_state.ultima_senha_digitada = senha
                        st.session_state.is_admin = False
                        st.session_state.base_admin_carregada = False
                        st.error("Senha incorreta")
                    else:
                        st.session_state.senha_admin_confirmada = True
                        st.session_state.ultima_senha_digitada = senha
                        registrar_base_carregada_no_estado(
                            logic,
                            logic.carregar_dados_originais(),
                            is_admin=True,
                            ultimo_arquivo=None,
                        )
                        st.session_state.grupo_config_expanded = False
                        st.success(f"Base carregada: {len(st.session_state.df_base)} jogadores.")
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
                        st.session_state.senha_admin_confirmada = False
                        st.session_state.grupo_config_expanded = False
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
            not st.session_state.df_base.empty
            or st.session_state.novos_jogadores
            or st.session_state.is_admin
        ):
            with st.expander("Ações secundárias", expanded=False):
                st.caption("Use a limpeza apenas quando quiser reiniciar a base atual.")
                if st.button("🗑 Limpar base atual", key="grupo_limpar_base_atual"):
                    st.session_state.df_base = logic.criar_base_vazia()
                    st.session_state.novos_jogadores = []
                    st.session_state.is_admin = False
                    st.session_state.base_admin_carregada = False
                    st.session_state.ultimo_arquivo = None
                    st.session_state.qtd_jogadores_adicionados_manualmente = 0
                    st.session_state.senha_admin_confirmada = False
                    st.session_state.base_inconsistencias_carregamento = {}
                    st.session_state.base_registros_inconsistentes_carregamento = []
                    st.session_state.grupo_busca_status = "idle"
                    st.session_state.grupo_nome_ultima_busca = ""
                    st.session_state.grupo_origem_fluxo = None
                    st.session_state.grupo_config_expanded = True
                    st.session_state["grupo_nome_pelada__pending"] = ""
                    st.session_state["grupo_senha_admin__pending"] = ""
                    st.rerun()

    return nome_pelada
