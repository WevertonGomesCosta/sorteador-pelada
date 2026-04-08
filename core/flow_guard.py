"""Helpers de fluxo e prontidão pré-sorteio."""

import hashlib
import json
import random

import pandas as pd
import streamlit as st

from core.validators import normalizar_nome_comparacao
from ui.base_view import resumo_inconsistencias_base, total_inconsistencias_base
from ui.summary_strings import obter_criterios_ativos, resumo_criterios_ativos


def construir_assinatura_entrada_sorteio(lista_texto: str, n_times: int) -> str:
    cols = ["Nome", "Nota", "Posição", "Velocidade", "Movimentação"]

    df_base = st.session_state.get("df_base")
    if df_base is None or df_base.empty:
        base_json = "[]"
    else:
        df_base_ref = df_base.copy()
        for col in cols:
            if col not in df_base_ref.columns:
                df_base_ref[col] = ""
        base_json = (
            df_base_ref[cols]
            .astype(str)
            .to_json(orient="split", force_ascii=False)
        )

    novos_jogadores = st.session_state.get("novos_jogadores", [])
    if novos_jogadores:
        df_novos = pd.DataFrame(novos_jogadores)
        for col in cols:
            if col not in df_novos.columns:
                df_novos[col] = ""
        novos_json = (
            df_novos[cols]
            .astype(str)
            .to_json(orient="split", force_ascii=False)
        )
    else:
        novos_json = "[]"

    payload = {
        "lista_texto": lista_texto or "",
        "n_times": int(n_times),
        "criterios": obter_criterios_ativos(),
        "base_json": base_json,
        "novos_json": novos_json,
    }
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def extrair_nomes_unicos_da_lista(logic, lista_texto: str) -> tuple[list[str], int]:
    processamento = logic.processar_lista(
        lista_texto,
        return_metadata=True,
        emit_warning=False,
    )
    nomes_lidos = processamento.get("jogadores", [])
    nomes_unicos = list(dict.fromkeys(nomes_lidos))
    qtd_duplicados = max(0, len(nomes_lidos) - len(nomes_unicos))
    return nomes_unicos, qtd_duplicados


def sortear_times_aleatorios_por_lista(nomes: list[str], n_times: int) -> list[list[list]]:
    nomes_embaralhados = nomes.copy()
    random.shuffle(nomes_embaralhados)

    times = [[] for _ in range(n_times)]
    for idx, nome in enumerate(nomes_embaralhados):
        time_idx = idx % n_times
        times[time_idx].append([nome, None, "", None, None])

    return times


def invalidar_resultado_se_entrada_mudou(lista_texto: str, n_times: int):
    if "resultado" not in st.session_state:
        return

    assinatura_anterior = st.session_state.get("resultado_assinatura")
    if not assinatura_anterior:
        return

    assinatura_atual = construir_assinatura_entrada_sorteio(lista_texto, n_times)
    if assinatura_atual == assinatura_anterior:
        return

    st.session_state.pop("resultado", None)
    st.session_state.pop("resultado_contexto", None)
    st.session_state.resultado_assinatura = None
    st.session_state.scroll_para_resultado = False
    st.session_state.resultado_invalidado_msg = True


def contar_duplicados_base_atual(df_base: pd.DataFrame) -> int:
    if df_base is None or df_base.empty or "Nome" not in df_base.columns:
        return 0

    nomes = (
        df_base["Nome"]
        .fillna("")
        .astype(str)
        .apply(normalizar_nome_comparacao)
    )
    nomes = nomes[nomes.ne("")]
    if nomes.empty:
        return 0
    return int(nomes[nomes.duplicated(keep=False)].nunique())


def construir_gate_pre_sorteio(logic, lista_texto: str, qtd_nomes_informados: int, n_times: int) -> dict:
    df_base = st.session_state.get("df_base", pd.DataFrame())
    diagnostico_atual = st.session_state.get("diagnostico_lista") or {}
    lista_texto_revisado = st.session_state.get("lista_texto_revisado", "")
    lista_revisada_atual = bool(diagnostico_atual) and lista_texto == lista_texto_revisado
    lista_confirmada_atual = bool(
        st.session_state.get("lista_revisada_confirmada")
        and st.session_state.get("lista_revisada")
        and lista_texto == lista_texto_revisado
    )
    cadastro_guiado_ativo = bool(st.session_state.get("cadastro_guiado_ativo", False))
    base_pronta = bool(not df_base.empty or st.session_state.get("novos_jogadores"))

    nomes_lista_unicos, qtd_duplicados_lista = extrair_nomes_unicos_da_lista(logic, lista_texto)
    qtd_nomes_unicos = len(nomes_lista_unicos)
    sorteio_aleatorio_lista = bool(not base_pronta and qtd_nomes_unicos > 0)

    if hasattr(logic, "diagnosticar_inconsistencias_base"):
        inconsistencias_base = logic.diagnosticar_inconsistencias_base(df_base)
    else:
        inconsistencias_base = st.session_state.get("base_inconsistencias_carregamento", {})

    total_inconsistencias = total_inconsistencias_base(inconsistencias_base)
    resumo_incons = resumo_inconsistencias_base(inconsistencias_base)
    qtd_duplicados_base = contar_duplicados_base_atual(df_base)

    faltantes = len(diagnostico_atual.get("nao_encontrados", [])) if lista_revisada_atual else 0
    bloqueios_base = len(diagnostico_atual.get("nomes_bloqueados_base", [])) if lista_revisada_atual else 0

    pendencias = []
    avisos = []

    if qtd_nomes_informados == 0:
        pendencias.append("nenhum nome foi informado na lista")

    if sorteio_aleatorio_lista:
        if qtd_nomes_unicos < 2:
            pendencias.append("o modo aleatório exige pelo menos 2 nomes únicos na lista")
        if qtd_nomes_unicos < n_times:
            pendencias.append(f"há apenas {qtd_nomes_unicos} nome(s) único(s) para {n_times} time(s)")

        avisos.append("Modo aleatório por lista ativo: nenhuma base foi carregada.")
        avisos.append("Os critérios de equilíbrio, métricas e odds não serão aplicados neste modo.")
        avisos.append(f"O sorteio usará {qtd_nomes_unicos} nome(s) único(s) informados na lista.")
        if qtd_duplicados_lista > 0:
            avisos.append(f"Há {qtd_duplicados_lista} repetição(ões) na lista; revise e corrija os nomes repetidos antes do sorteio.")
    else:
        if not base_pronta:
            pendencias.append("nenhuma base foi carregada ou construída")
        elif not lista_revisada_atual:
            pendencias.append("a lista ainda não foi revisada com a versão atual dos dados")

        if cadastro_guiado_ativo:
            pendencias.append("há um cadastro guiado em andamento")
        if lista_revisada_atual and faltantes > 0:
            pendencias.append(f"há {faltantes} nome(s) faltante(s) na base")
        if lista_revisada_atual and bloqueios_base > 0:
            pendencias.append(f"há {bloqueios_base} nome(s) com duplicidade ou inconsistência na base atual")
        if lista_revisada_atual and not lista_confirmada_atual:
            pendencias.append("a lista revisada ainda não foi confirmada")

    if qtd_duplicados_base > 0:
        avisos.append(f"Base atual com {qtd_duplicados_base} nome(s) duplicado(s).")
    if total_inconsistencias > 0:
        detalhe = f": {resumo_incons}" if resumo_incons else ""
        avisos.append(f"Base atual com {total_inconsistencias} inconsistência(s){detalhe}.")

    nomes_referencia_alerta = qtd_nomes_unicos if sorteio_aleatorio_lista else qtd_nomes_informados
    if nomes_referencia_alerta > 0 and nomes_referencia_alerta < n_times * 2:
        avisos.append("Há poucos nomes para a quantidade de times escolhida; o sorteio pode ficar menos equilibrado.")

    if sorteio_aleatorio_lista:
        base_status = "sem base carregada · modo aleatório pela lista"
        lista_status = f"{qtd_nomes_unicos} nome(s) único(s)"
        if qtd_duplicados_lista > 0:
            lista_status += f" · {qtd_duplicados_lista} repetição(ões) para revisar"
        criterios_status = "Ignorados · sorteio apenas pelos nomes da lista"
        prontidao_status = "Pronto para sortear · modo aleatório" if len(pendencias) == 0 else f"Bloqueado · {len(pendencias)} pendência(s)"
        modo_sorteio = "aleatorio_lista"
        modo_status = "Aleatório por lista"
    else:
        base_status = f"{len(df_base)} jogador(es)" if base_pronta else "sem base carregada"
        if qtd_duplicados_base > 0 or total_inconsistencias > 0:
            partes_base = [base_status]
            if qtd_duplicados_base > 0:
                partes_base.append(f"{qtd_duplicados_base} duplicidade(s)")
            if total_inconsistencias > 0:
                partes_base.append(f"{total_inconsistencias} inconsistência(s)")
            base_status = " · ".join(partes_base)

        lista_status = f"{qtd_nomes_informados} nome(s) lido(s)"
        if lista_revisada_atual:
            lista_status += " · revisada"
            if lista_confirmada_atual:
                lista_status += " · confirmada"
        elif qtd_nomes_informados > 0:
            lista_status += " · pendente de revisão"

        criterios_status = resumo_criterios_ativos()
        prontidao_status = "Pronto para sortear" if len(pendencias) == 0 else f"Bloqueado · {len(pendencias)} pendência(s)"
        modo_sorteio = "balanceado"
        modo_status = "Balanceado com base"

    return {
        "pronto_para_sortear": len(pendencias) == 0,
        "base_status": base_status,
        "lista_status": lista_status,
        "criterios_status": criterios_status,
        "prontidao_status": prontidao_status,
        "pendencias": pendencias,
        "avisos": avisos,
        "modo_sorteio": modo_sorteio,
        "modo_status": modo_status,
        "sorteio_aleatorio_lista": sorteio_aleatorio_lista,
        "qtd_nomes_unicos_lista": qtd_nomes_unicos,
        "qtd_duplicados_lista": qtd_duplicados_lista,
    }
