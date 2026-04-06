"""Camada de dados da aplicação.

Centraliza operações de base e upload sem alterar a API pública usada por
`PeladaLogic`.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

DataFrameCleaner = Callable[[pd.DataFrame], pd.DataFrame]


def criar_base_vazia() -> pd.DataFrame:
    return pd.DataFrame(columns=["Nome", "Nota", "Posição", "Velocidade", "Movimentação"])


def criar_exemplo() -> pd.DataFrame:
    dados_exemplo = [
        {"Nome": "Exemplo Atacante", "Nota": 8.5, "Posição": "A", "Velocidade": 5, "Movimentação": 4},
        {"Nome": "Exemplo Meio", "Nota": 6.0, "Posição": "M", "Velocidade": 3, "Movimentação": 3},
        {"Nome": "Exemplo Zagueiro", "Nota": 7.0, "Posição": "D", "Velocidade": 2, "Movimentação": 2},
    ]
    return pd.DataFrame(dados_exemplo)


def converter_df_para_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Notas pelada")
    return output.getvalue()


def carregar_dados_originais(url_padrao: str, limpar_df_fn: DataFrameCleaner) -> pd.DataFrame:
    try:
        df = pd.read_excel(url_padrao, sheet_name="Notas pelada")
        return limpar_df_fn(df)
    except Exception as e:
        st.error(f"Erro ao conectar com Google Sheets: {e}")
        return criar_base_vazia()


def processar_upload(arquivo_upload: Any, limpar_df_fn: DataFrameCleaner) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(arquivo_upload)
        df = limpar_df_fn(df)
        return df
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        return None


def limpar_df(
    df: pd.DataFrame | None,
    *,
    diagnosticar_inconsistencias_base_fn: Callable[[pd.DataFrame], dict],
    listar_registros_inconsistentes_fn: Callable[[pd.DataFrame], pd.DataFrame],
    formatar_nome_visual_fn: Callable[[str], str],
) -> pd.DataFrame:
    cols = ["Nome", "Nota", "Posição", "Velocidade", "Movimentação"]
    if df is None or df.empty:
        st.session_state["base_inconsistencias_carregamento"] = {}
        st.session_state["base_registros_inconsistentes_carregamento"] = []
        return criar_base_vazia()

    inconsistencias = diagnosticar_inconsistencias_base_fn(df)
    st.session_state["base_inconsistencias_carregamento"] = inconsistencias
    st.session_state["base_registros_inconsistentes_carregamento"] = (
        listar_registros_inconsistentes_fn(df).to_dict("records")
    )

    df_limpo = df.copy()
    for col in cols:
        if col not in df_limpo.columns:
            df_limpo[col] = 0 if col not in ["Nome", "Posição"] else ""

    df_limpo = df_limpo[cols]
    df_limpo["Posição"] = df_limpo["Posição"].fillna("").astype(str)
    df_limpo = df_limpo[df_limpo["Posição"].str.upper() != "G"].reset_index(drop=True)
    df_limpo = df_limpo.dropna(subset=["Nota"])
    df_limpo["Nome"] = df_limpo["Nome"].apply(formatar_nome_visual_fn)

    # Duplicidade na base não deve bloquear o carregamento.
    # A interface do app fará apenas o diagnóstico visual desses casos.
    return df_limpo.reset_index(drop=True)
