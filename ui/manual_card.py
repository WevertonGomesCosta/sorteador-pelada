"""
AVISO TEMPORÁRIO DE REPOSITÓRIO

Este arquivo está temporariamente desatualizado e NÃO é a fonte oficial de verdade
do cadastro manual no estado atual do projeto.

A implementação atualmente vigente e mais completa de cadastro manual está no
`app.py`. Este módulo será substituído em uma próxima rodada de reorganização
do repositório, quando a implementação atual do `app.py` for migrada para cá.

Até essa substituição, evite evoluir a lógica principal de cadastro manual neste
arquivo para não criar divergência com o fluxo ativo do app.
"""

import pandas as pd
import streamlit as st


def render_manual_card(logic, nome_pelada: str):
    with st.expander("📝 Adicionar Jogador Manualmente", expanded=False):
        st.caption("Se preferir montar ou editar a base fora do app, baixe o modelo abaixo.")
        df_exemplo = logic.criar_exemplo()
        excel_exemplo = logic.converter_df_para_excel(df_exemplo)
        st.download_button(
            label="📥 Baixar Modelo de Planilha",
            data=excel_exemplo,
            file_name="modelo_pelada.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Baixe este arquivo para ver como preencher os dados corretamente."
        )

        with st.form("form_add_manual"):
            col_a, col_b = st.columns(2)
            nome_m = col_a.text_input("Nome")
            p_m = col_b.selectbox("Posição", ["M", "A", "D"])
            n_m = st.slider("Nota", 1.0, 10.0, 6.0, 0.5)
            v_m = st.slider("Velocidade", 1, 5, 3)
            mv_m = st.slider("Movimentação", 1, 5, 3)
            if st.form_submit_button("Adicionar à Base"):
                if nome_m:
                    novo_nome = logic.formatar_nome_visual(nome_m)
                    novo = {'Nome': novo_nome, 'Nota': n_m, 'Posição': p_m, 'Velocidade': v_m, 'Movimentação': mv_m}
                    st.session_state.df_base.loc[len(st.session_state.df_base)] = novo
                    st.success(f"{novo_nome} salvo!")
                else:
                    st.error("Digite um nome.")

        st.markdown("---")
        if not st.session_state.df_base.empty:
            st.caption("Baixe a planilha atual com os jogadores já adicionados à base.")
            if st.session_state.is_admin:
                st.info("🔒 O download da Base Mestra é bloqueado por segurança.")
            else:
                nome_arquivo = nome_pelada.strip()
                if not nome_arquivo:
                    nome_arquivo = "minha_pelada"
                if not nome_arquivo.endswith(".xlsx"):
                    nome_arquivo += ".xlsx"
                excel_data = logic.converter_df_para_excel(st.session_state.df_base)
                st.download_button(
                    label="💾 Baixar Minha Planilha",
                    data=excel_data,
                    file_name=nome_arquivo,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            if not st.session_state.is_admin:
                st.info("Adicione jogadores para baixar a planilha.")
