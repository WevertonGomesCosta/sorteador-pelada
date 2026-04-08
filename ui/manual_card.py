import pandas as pd
import streamlit as st

from core.validators import normalizar_nome_comparacao
from ui.summary_strings import resumo_expander_cadastro_manual


def render_manual_card(
    logic,
    nome_pelada: str,
    on_open_expander,
    render_inline_correction,
):
    """Render the manual player registration section.

    This is the current source of truth for the manual registration flow.
    The callbacks are injected by app.py to avoid circular imports while the
    repository is still being reorganized.
    """
    with st.expander(
        resumo_expander_cadastro_manual(),
        expanded=st.session_state.get("cadastro_manual_expanded", False),
    ):
        st.caption(
            "Use esta etapa para montar sua base do zero ou complementar a base atual com novos jogadores."
        )
        st.caption("Quer montar ou editar a base fora do app? Baixe o modelo de planilha abaixo.")

        df_exemplo = logic.criar_exemplo()
        excel_exemplo = logic.converter_df_para_excel(df_exemplo)
        st.download_button(
            label="📥 Baixar planilha modelo",
            data=excel_exemplo,
            file_name="modelo_pelada.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Baixe este arquivo para ver como preencher a planilha no formato correto.",
            key="manual_baixar_modelo_planilha",
        )

        with st.form("form_add_manual"):
            col_a, col_b = st.columns(2)
            nome_m = col_a.text_input("Nome")
            p_m = col_b.selectbox("Posição", ["M", "A", "D"])
            n_m = st.slider("Nota", 1, 10, 6)
            v_m = st.slider("Velocidade", 1, 5, 3)
            mv_m = st.slider("Movimentação", 1, 5, 3)
            submit_manual = st.form_submit_button(
                "Adicionar à Base",
                on_click=on_open_expander,
            )
            if submit_manual:
                if nome_m:
                    novo_nome = logic.formatar_nome_visual(nome_m)
                    nomes_existentes = {
                        normalizar_nome_comparacao(nome)
                        for nome in st.session_state.df_base["Nome"].astype(str).tolist()
                    }
                    nomes_existentes.update(
                        {
                            normalizar_nome_comparacao(nome)
                            for nome in pd.Series(st.session_state.get("novos_jogadores", []))
                            .apply(lambda x: x.get("Nome") if isinstance(x, dict) else None)
                            .dropna()
                            .tolist()
                        }
                    )

                    if normalizar_nome_comparacao(novo_nome) in nomes_existentes:
                        st.session_state.cadastro_manual_expanded = True
                        st.session_state.cadastro_manual_nome_existente = novo_nome
                        st.error(
                            "Esse nome já existe na base atual. Revise a grafia ou edite o registro existente antes de adicionar novamente."
                        )
                    else:
                        novo = {
                            "Nome": novo_nome,
                            "Nota": n_m,
                            "Posição": p_m,
                            "Velocidade": v_m,
                            "Movimentação": mv_m,
                        }
                        st.session_state.df_base.loc[len(st.session_state.df_base)] = novo
                        st.session_state.qtd_jogadores_adicionados_manualmente += 1
                        st.session_state.cadastro_manual_expanded = False
                        st.session_state.cadastro_manual_nome_existente = ""
                        st.success(f"{novo_nome} salvo!")
                else:
                    st.session_state.cadastro_manual_expanded = True
                    st.session_state.cadastro_manual_nome_existente = ""
                    st.error("Digite um nome.")

        nome_existente = st.session_state.get("cadastro_manual_nome_existente", "")
        if nome_existente:
            st.warning("Você pode editar ou remover esse registro existente sem sair da etapa 3.")
            render_inline_correction(
                logic,
                st.session_state.get("lista_texto_revisado", ""),
                [{"nome": nome_existente, "motivos": ["nome já existente na base"]}],
            )

        if (
            not st.session_state.cadastro_guiado_ativo
            and st.session_state.revisao_pendente_pos_cadastro
            and st.session_state.faltantes_cadastrados_na_rodada
        ):
            st.success(
                "Todos os faltantes desta revisão foram cadastrados. Agora revise a lista novamente para liberar o sorteio."
            )
            st.caption(
                f"Cadastrados nesta rodada: {', '.join(st.session_state.faltantes_cadastrados_na_rodada)}"
            )

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
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        else:
            if not st.session_state.is_admin:
                st.info("Sem base carregada? Você pode adicionar jogadores aqui e montar sua base manualmente.")
