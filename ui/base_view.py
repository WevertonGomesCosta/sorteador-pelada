"""Visualização, resumo e integridade da base do Sorteador Pelada PRO."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import state.keys as K

from core.base_summary import resumo_inconsistencias_base, total_inconsistencias_base
from core.validators import (
    normalizar_nome_comparacao,
    registro_valido_para_sorteio,
    valor_slider_corrigir,
)
from ui.primitives import render_section_header


def formatar_df_visual_numeros_inteiros(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df_fmt = df.copy()
    for col in ["Nota", "Velocidade", "Movimentação"]:
        if col in df_fmt.columns:
            def _formatar_valor_visual(v):
                try:
                    if pd.isna(v):
                        return v
                except Exception:
                    pass
                try:
                    num = float(v)
                    if num.is_integer():
                        return int(num)
                    return round(num, 2)
                except Exception:
                    return v
            df_fmt[col] = df_fmt[col].apply(_formatar_valor_visual)
    return df_fmt


def estilo_celulas_inconsistentes(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(index=getattr(df, "index", []), columns=getattr(df, "columns", []))

    estilos = pd.DataFrame("", index=df.index, columns=df.columns)
    destaque = "font-weight: 700;"

    if "Nome" in df.columns:
        nomes = df["Nome"].fillna("").astype(str).str.strip()
        estilos.loc[nomes.eq(""), "Nome"] = destaque

    if "Posição" in df.columns:
        posicoes = df["Posição"].fillna("").astype(str).str.strip().str.upper()
        estilos.loc[~posicoes.isin(["D", "M", "A", "G"]), "Posição"] = destaque

    if "Nota" in df.columns:
        nota = pd.to_numeric(df["Nota"], errors="coerce")
        estilos.loc[nota.isna() | (nota < 0) | (nota > 10), "Nota"] = destaque

    if "Velocidade" in df.columns:
        velocidade = pd.to_numeric(df["Velocidade"], errors="coerce")
        estilos.loc[velocidade.isna() | (velocidade < 0) | (velocidade > 10), "Velocidade"] = destaque

    if "Movimentação" in df.columns:
        movimentacao = pd.to_numeric(df["Movimentação"], errors="coerce")
        estilos.loc[movimentacao.isna() | (movimentacao < 0) | (movimentacao > 10), "Movimentação"] = destaque

    return estilos


def render_base_summary():
    df_base = st.session_state[K.DF_BASE]
    qtd_jogadores = len(df_base)

    if st.session_state[K.IS_ADMIN]:
        origem = "Grupo"
    elif qtd_jogadores == 0:
        origem = "Vazia"
    else:
        origem = "Sua base"

    modo = "Base do grupo" if st.session_state[K.IS_ADMIN] else "Público"

    if df_base.empty:
        posicoes = "—"
    else:
        cont_pos = df_base["Posição"].value_counts()
        posicoes = " / ".join(
            [
                f"D {cont_pos.get('D', 0)}",
                f"M {cont_pos.get('M', 0)}",
                f"A {cont_pos.get('A', 0)}",
            ]
        )

    st.markdown(
        f"""
import state.keys as K
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-label">⚽ Modo</div>
                <div class="summary-value">{modo}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">👥 Jogadores</div>
                <div class="summary-value">{qtd_jogadores} jogadores</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">📋 Base</div>
                <div class="summary-value">{origem}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">🧩 D / M / A</div>
                <div class="summary-value">{posicoes}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_base_inconsistencias_expander(
    logic=None,
    *,
    atualizar_integridade_base_no_estado=None,
    diagnosticar_lista_no_estado=None,
    render_action_button=None,
):
    registros = st.session_state.get(K.BASE_REGISTROS_INCONSISTENTES_CARREGAMENTO, [])
    if not registros:
        return

    df_inconsistentes = pd.DataFrame(registros)
    if df_inconsistentes.empty:
        return

    with st.expander("⚠️ Registros com inconsistências", expanded=False):
        st.caption("Revise apenas os registros inconsistentes. Você pode remover a linha ou, se preferir, corrigi-la no próprio bloco.")
        df_inconsistentes_display = df_inconsistentes.copy()
        styler = df_inconsistentes_display.style.apply(estilo_celulas_inconsistentes, axis=None)
        st.dataframe(
            styler,
            width="stretch",
            hide_index=True,
        )

        if logic is None or atualizar_integridade_base_no_estado is None or render_action_button is None:
            return

        st.markdown("---")
        st.caption("A exclusão é a ação mais rápida quando a linha está errada. Se quiser manter o registro, use a opção de correção.")

        for _, row in df_inconsistentes.iterrows():
            nome = str(row.get("Nome", "")).strip()
            posicao = str(row.get("Posição", "")).strip()
            nota = row.get("Nota", "")
            velocidade = row.get("Velocidade", "")
            movimentacao = row.get("Movimentação", "")

            candidatos = st.session_state[K.DF_BASE].copy()
            if nome:
                candidatos = candidatos[candidatos["Nome"].astype(str).str.strip() == nome]
            else:
                candidatos = candidatos[candidatos["Nome"].fillna("").astype(str).str.strip() == ""]

            candidatos = candidatos.reset_index().rename(columns={"index": "_orig_index"})

            idx_original = None
            for _, cand in candidatos.iterrows():
                if (
                    str(cand.get("Posição", "")) == posicao
                    and str(cand.get("Nota", "")) == str(nota)
                    and str(cand.get("Velocidade", "")) == str(velocidade)
                    and str(cand.get("Movimentação", "")) == str(movimentacao)
                ):
                    idx_original = int(cand["_orig_index"])
                    break
            if idx_original is None and not candidatos.empty:
                idx_original = int(candidatos.iloc[0]["_orig_index"])
            if idx_original is None:
                continue

            with st.expander(f"🧾 Registro inconsistente: {nome or '(sem nome)'}", expanded=False):
                col_info1, col_info2, col_info3, col_info4, col_info5 = st.columns(5)
                col_info1.markdown(f"**Nome**\n\n{nome}")
                col_info2.markdown(f"**Posição**\n\n{posicao}")
                col_info3.markdown(f"**Nota**\n\n{nota}")
                col_info4.markdown(f"**Velocidade**\n\n{velocidade}")
                col_info5.markdown(f"**Movimentação**\n\n{movimentacao}")

                if render_action_button(
                    "🗑️ Excluir esta linha",
                    key=f"base_inconsistente_excluir_{idx_original}",
                    role="secondary",
                ):
                    st.session_state[K.DF_BASE] = (
                        st.session_state[K.DF_BASE].drop(index=idx_original).reset_index(drop=True)
                    )
                    st.session_state[K.REVISAO_FOCO_BLOQUEIO_NOME] = nome
                    atualizar_integridade_base_no_estado(logic)
                    if diagnosticar_lista_no_estado is not None and st.session_state.get(K.LISTA_TEXTO_REVISADO, ""):
                        diagnosticar_lista_no_estado(logic, st.session_state.get(K.LISTA_TEXTO_REVISADO, ""))
                        st.session_state[K.REVISAO_LISTA_EXPANDIDA] = True
                    st.rerun()

                with st.expander("✏️ Corrigir este registro", expanded=False):
                    with st.form(f"base_inconsistente_corrigir_{idx_original}"):
                        nome_corr = st.text_input(
                            "Nome",
                            value=nome,
                            key=f"base_inconsistente_nome_{idx_original}",
                        )
                        posicoes_formulario = ["D", "M", "A", "G"]
                        posicao_atual = posicao.upper() if posicao.upper() in posicoes_formulario else "M"
                        pos_corr = st.selectbox(
                            "Posição",
                            posicoes_formulario,
                            index=posicoes_formulario.index(posicao_atual),
                            key=f"base_inconsistente_pos_{idx_original}",
                        )
                        nota_corr = st.slider(
                            "Nota", 0.0, 10.0, valor_slider_corrigir(nota, 0.0, 10.0, 6.0), 0.5,
                            key=f"base_inconsistente_nota_{idx_original}",
                        )
                        vel_corr = st.slider(
                            "Velocidade", 0.0, 10.0, valor_slider_corrigir(velocidade, 0.0, 10.0, 3.0), 0.5,
                            key=f"base_inconsistente_vel_{idx_original}",
                        )
                        mov_corr = st.slider(
                            "Movimentação", 0.0, 10.0, valor_slider_corrigir(movimentacao, 0.0, 10.0, 3.0), 0.5,
                            key=f"base_inconsistente_mov_{idx_original}",
                        )
                        salvar = st.form_submit_button("💾 Salvar correção")

                        if salvar:
                            nome_corrigido = str(nome_corr).strip()
                            if hasattr(logic, "formatar_nome_visual") and nome_corrigido:
                                nome_corrigido = logic.formatar_nome_visual(nome_corrigido)
                            st.session_state[K.DF_BASE].loc[idx_original, "Nome"] = nome_corrigido
                            st.session_state[K.DF_BASE].loc[idx_original, "Posição"] = pos_corr
                            st.session_state[K.DF_BASE].loc[idx_original, "Nota"] = nota_corr
                            st.session_state[K.DF_BASE].loc[idx_original, "Velocidade"] = vel_corr
                            st.session_state[K.DF_BASE].loc[idx_original, "Movimentação"] = mov_corr
                            atualizar_integridade_base_no_estado(logic)
                            if diagnosticar_lista_no_estado is not None and st.session_state.get(K.LISTA_TEXTO_REVISADO, ""):
                                diagnosticar_lista_no_estado(logic, st.session_state.get(K.LISTA_TEXTO_REVISADO, ""))
                                st.session_state[K.REVISAO_LISTA_EXPANDIDA] = True
                            st.rerun()


def render_base_integrity_alert():
    df_base = st.session_state[K.DF_BASE]

    if df_base.empty:
        return

    inconsistencias = st.session_state.get(K.BASE_INCONSISTENCIAS_CARREGAMENTO, {})
    total_inconsistencias = total_inconsistencias_base(inconsistencias)
    resumo_inconsistencias = resumo_inconsistencias_base(inconsistencias)

    nomes_normalizados = df_base["Nome"].astype(str).apply(normalizar_nome_comparacao)
    duplicados = nomes_normalizados[nomes_normalizados.duplicated(keep=False)]

    if not duplicados.empty:
        qtd_nomes_duplicados = duplicados.nunique()
        mensagem = (
            f"A base carregada tem {qtd_nomes_duplicados} nome(s) duplicado(s). "
            "Nomes duplicados bloqueiam o sorteio até a correção."
        )
        st.warning(mensagem)

    if total_inconsistencias > 0:
        detalhe = f": {resumo_inconsistencias}" if resumo_inconsistencias else "."
        st.warning(
            f"A base carregada tem {total_inconsistencias} inconsistência(s){detalhe} "
            "Esses registros podem bloquear jogadores até a correção."
        )


def render_base_preview():
    df_base = st.session_state[K.DF_BASE]
    if df_base.empty:
        return

    st.markdown("**Base atual**")
    df_visual = formatar_df_visual_numeros_inteiros(df_base)
    st.dataframe(df_visual, width="stretch", hide_index=True)
