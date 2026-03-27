import streamlit as st


def render_sidebar(logic, nome_pelada_adm: str, senha_adm: str) -> str:
    with st.sidebar:
        st.header("🔐 Configuração do Grupo")
        nome_pelada = st.text_input("Nome da Pelada:", placeholder="Ex: Pelada de Domingo")

        # VERIFICAÇÃO COM DADOS DO SECRETS
        if nome_pelada.strip().upper() == str(nome_pelada_adm).upper():
            st.success("Grupo identificado!")
            opcao = st.radio(
                "Selecione a ação:",
                ["Acessar Base Original (Admin)", "Criar Nova Lista (Limpar)"]
            )

            if opcao == "Acessar Base Original (Admin)":
                senha = st.text_input("Senha de Acesso:", type="password")

                if senha == str(senha_adm):
                    st.session_state.is_admin = True
                    st.success("🔓 Acesso liberado")
                else:
                    st.session_state.is_admin = False
                    if senha:
                        st.error("Senha incorreta")
            else:
                st.session_state.is_admin = False
                if st.button("🗑 Confirmar Limpeza"):
                    st.session_state.df_base = logic.criar_base_vazia()
                    st.session_state.novos_jogadores = []
                    st.rerun()
        else:
            st.session_state.is_admin = False
            if st.button("🗑 Limpar / Começar do Zero"):
                st.session_state.df_base = logic.criar_base_vazia()
                st.session_state.novos_jogadores = []
                st.rerun()

        st.markdown("---")
        st.subheader("📂 Banco de Dados")

        # Ações ADMIN
        if st.session_state.is_admin:
            if st.button("🔔 Carregar Planilha Original"):
                st.session_state.df_base = logic.carregar_dados_originais()
                st.session_state.novos_jogadores = []
                st.success(f"Base carregada: {len(st.session_state.df_base)} jogadores.")
                st.rerun()

        # upload
        st.write("Substituir por Excel Próprio:")

        uploaded_file = st.file_uploader(
            "Enviar planilha Excel",
            type=["xlsx"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            if (
                "ultimo_arquivo" not in st.session_state
                or st.session_state.ultimo_arquivo != uploaded_file.name
            ):
                df_novo = logic.processar_upload(uploaded_file)
                if df_novo is not None:
                    st.session_state.df_base = df_novo
                    st.session_state.novos_jogadores = []
                    st.session_state.ultimo_arquivo = uploaded_file.name
                    st.success("Arquivo carregado!")
                    st.rerun()

    return nome_pelada
