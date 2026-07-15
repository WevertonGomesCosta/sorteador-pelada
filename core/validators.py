"""Funções de validação e preparação de dados do Sorteador Pelada PRO."""

from __future__ import annotations

import pandas as pd
import re
import unicodedata


POSICOES_VALIDAS_SORTEIO = ["D", "M", "A", "G"]
VALOR_MINIMO_ATRIBUTO = 0
VALOR_MAXIMO_ATRIBUTO = 10


def normalizar_nome_comparacao(nome: str) -> str:
    nome = unicodedata.normalize("NFKD", str(nome))
    nome = "".join(ch for ch in nome if not unicodedata.combining(ch))
    nome = " ".join(nome.split())
    return nome.strip().upper()



def normalizar_nome_duplicado_lista(nome: str) -> str:
    nome = normalizar_nome_comparacao(nome)
    if not nome:
        return ""

    nome = re.sub(r"\s*\(\s*", "(", nome)
    nome = re.sub(r"\s*\)\s*", ")", nome)
    nome = re.sub(r"\s*[-–—]\s*", "-", nome)
    nome = " ".join(nome.split())
    return nome.strip()



def _atributo_invalido(valor) -> bool:
    valor_num = pd.to_numeric(pd.Series([valor]), errors="coerce").iloc[0]
    return bool(
        pd.isna(valor_num)
        or valor_num < VALOR_MINIMO_ATRIBUTO
        or valor_num > VALOR_MAXIMO_ATRIBUTO
    )



def registro_valido_para_sorteio(row: pd.Series) -> bool:
    nome = str(row.get("Nome", "")).strip()
    posicao = str(row.get("Posição", "")).strip().upper()

    if not nome:
        return False
    if posicao not in POSICOES_VALIDAS_SORTEIO:
        return False
    if _atributo_invalido(row.get("Nota")):
        return False
    if _atributo_invalido(row.get("Velocidade")):
        return False
    if _atributo_invalido(row.get("Movimentação")):
        return False

    return True



def diagnosticar_nomes_bloqueados_para_sorteio(df_base: pd.DataFrame, nomes_confirmados: list[str]) -> list[dict]:
    if df_base is None or df_base.empty:
        return [{"nome": nome, "motivos": ["sem registro na base atual"]} for nome in nomes_confirmados]

    bloqueios = []
    for nome in nomes_confirmados:
        df_nome = df_base[df_base["Nome"] == nome].copy()
        if df_nome.empty:
            bloqueios.append({"nome": nome, "motivos": ["sem registro na base atual"]})
            continue

        total_registros = len(df_nome)
        registros_validos = int(df_nome.apply(registro_valido_para_sorteio, axis=1).sum())
        registros_invalidos = total_registros - registros_validos

        motivos = []
        if total_registros > 1:
            motivos.append("duplicado na base")
        if registros_invalidos > 0:
            motivos.append("com inconsistência na base")
        if registros_validos == 0:
            motivos.append("sem registro válido para sorteio")

        if motivos:
            bloqueios.append({"nome": nome, "motivos": motivos})

    return bloqueios



def preparar_df_sorteio(df_base: pd.DataFrame, nomes_confirmados: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    bloqueios = diagnosticar_nomes_bloqueados_para_sorteio(df_base, nomes_confirmados)
    if bloqueios:
        return pd.DataFrame(), bloqueios

    if df_base is None or df_base.empty:
        return pd.DataFrame(), [{"nome": nome, "motivos": ["sem registro na base atual"]} for nome in nomes_confirmados]

    df_lista = df_base[df_base["Nome"].isin(nomes_confirmados)].copy()
    if df_lista.empty:
        return pd.DataFrame(), [{"nome": nome, "motivos": ["sem registro na base atual"]} for nome in nomes_confirmados]

    df_validos = df_lista[df_lista.apply(registro_valido_para_sorteio, axis=1)].copy()
    df_validos = df_validos.drop_duplicates(subset=["Nome"], keep="last")

    return df_validos.reset_index(drop=True), []



def valor_slider_corrigir(v, minimo: float, maximo: float, fallback: float) -> float:
    num = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    if pd.isna(num):
        return fallback
    return max(minimo, min(maximo, float(num)))



def listar_bloqueios_base_atual(df_base: pd.DataFrame) -> list[dict]:
    if df_base is None or df_base.empty:
        return []

    nomes = [
        nome for nome in df_base["Nome"].astype(str).tolist()
        if str(nome).strip()
    ]
    nomes_unicos = list(dict.fromkeys(nomes))
    return diagnosticar_nomes_bloqueados_para_sorteio(df_base, nomes_unicos)
