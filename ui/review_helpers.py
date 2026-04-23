"""Helpers puros da revisão da lista.

Este módulo concentra apenas utilitários sem efeitos colaterais de Streamlit
ou escrita em session_state. Ele existe para reduzir o tamanho de
``ui/review_view.py`` sem alterar a orquestração funcional da baseline v124/v126.
"""

from __future__ import annotations

import re

from core.validators import (
    normalizar_nome_comparacao,
    normalizar_nome_duplicado_lista,
)


def _normalizar_cabecalho_lista(linha: str) -> str:
    linha = normalizar_nome_comparacao(str(linha or "")).upper()
    return " ".join(linha.split())



def _eh_inicio_secao_excluida(linha: str) -> bool:
    cabecalho = _normalizar_cabecalho_lista(linha)
    return cabecalho.startswith("GOLEIROS") or cabecalho.startswith("LISTA DE ESPERA")



def _linhas_principais_da_lista(texto_lista: str) -> tuple[list[str], int]:
    linhas = str(texto_lista or "").splitlines()
    indice_corte = len(linhas)

    for idx, linha in enumerate(linhas):
        if _eh_inicio_secao_excluida(linha):
            indice_corte = idx
            break

    return linhas, indice_corte



def _extrair_nome_lista_preservando_qualificador(texto: str) -> str:
    return " ".join(str(texto or "").strip().split())



def _extrair_nome_comparavel_da_linha(linha: str) -> str:
    linha_original = str(linha or "").strip()
    if not linha_original:
        return ""

    match = re.search(r"^\s*\d+[.\-)]?\s*(.+)", linha_original)
    nome_extraido = match.group(1) if match else linha_original
    nome_limpo = _extrair_nome_lista_preservando_qualificador(nome_extraido)
    return nome_limpo



def _atualizar_texto_lista_revisao(
    texto_lista: str,
    nome_alvo: str,
    *,
    novo_nome: str | None = None,
    remover: bool = False,
    manter_primeira_ocorrencia: bool = False,
) -> tuple[str, bool]:
    linhas, indice_corte = _linhas_principais_da_lista(texto_lista)
    alvo_normalizado = normalizar_nome_comparacao(nome_alvo)
    novas_linhas = []
    encontrou = False

    for linha_idx, linha_original in enumerate(linhas):
        linha = str(linha_original).strip()

        if linha_idx >= indice_corte or not linha:
            novas_linhas.append(linha_original)
            continue

        linha_comparavel = _extrair_nome_comparavel_da_linha(linha)
        linha_normalizada = normalizar_nome_comparacao(linha_comparavel)
        mesma_pessoa = bool(alvo_normalizado) and linha_normalizada == alvo_normalizado

        if not mesma_pessoa:
            novas_linhas.append(linha_original)
            continue

        if remover:
            if manter_primeira_ocorrencia and not encontrou:
                novas_linhas.append(linha_original)
            encontrou = True
            continue

        nome_destino = str(novo_nome or "").strip()
        if nome_destino:
            novas_linhas.append(nome_destino)
            encontrou = True
        else:
            novas_linhas.append(linha_original)

    return "\n".join(novas_linhas), encontrou



def _detalhe_do_nome_duplicado(diagnostico: dict, nome_duplicado: str) -> dict:
    alvo_normalizado = normalizar_nome_duplicado_lista(nome_duplicado)
    for detalhe in diagnostico.get("duplicados_detalhados", []) or []:
        if normalizar_nome_duplicado_lista(detalhe.get("nome", "")) == alvo_normalizado:
            return detalhe
    return {}



def _ocorrencias_numeradas_da_lista_principal(texto_lista: str) -> list[dict]:
    linhas, indice_corte = _linhas_principais_da_lista(texto_lista)
    pattern = r"^\s*(\d+)\s*[.\-\)]?\s*(.+)"
    ignorar = {".", "-", "...", "Lista", "Times"}
    ocorrencias = []

    for linha_idx, linha_original in enumerate(linhas[:indice_corte]):
        linha = str(linha_original).strip()
        if not linha:
            continue

        match = re.search(pattern, linha)
        if not match:
            continue

        numero, conteudo = match.groups()
        nome_limpo = _extrair_nome_lista_preservando_qualificador(conteudo)
        nome_formatado = " ".join(nome_limpo.split())
        if len(nome_formatado) <= 1 or nome_formatado in ignorar:
            continue

        ocorrencias.append(
            {
                "linha_idx": linha_idx,
                "valor": linha,
                "comparavel": _extrair_nome_comparavel_da_linha(linha),
                "numero": numero,
            }
        )

    return ocorrencias



def _ocorrencias_do_nome_duplicado_na_lista(
    texto_lista: str,
    diagnostico: dict,
    nome_duplicado: str,
    *,
    revisao_aleatoria: bool,
) -> list[dict]:
    detalhe_duplicado = _detalhe_do_nome_duplicado(diagnostico, nome_duplicado)

    if revisao_aleatoria:
        alvos_normalizados = {normalizar_nome_duplicado_lista(nome_duplicado)}
    else:
        ocorrencias_referencia = (
            detalhe_duplicado.get("ocorrencias_exibicao")
            or detalhe_duplicado.get("ocorrencias_corrigidas")
            or [nome_duplicado]
        )
        alvos_normalizados = {
            normalizar_nome_duplicado_lista(nome)
            for nome in ocorrencias_referencia
            if str(nome).strip()
        }

    ocorrencias = []
    for ocorrencia in _ocorrencias_numeradas_da_lista_principal(texto_lista):
        linha_normalizada = normalizar_nome_duplicado_lista(ocorrencia["comparavel"])
        if linha_normalizada and linha_normalizada in alvos_normalizados:
            ocorrencias.append(ocorrencia)

    return ocorrencias



def _aplicar_edicoes_em_ocorrencias_da_lista(
    texto_lista: str,
    edicoes_por_linha: dict[int, str],
    *,
    remover_linhas: set[int] | None = None,
) -> tuple[str, bool]:
    linhas, indice_corte = _linhas_principais_da_lista(texto_lista)
    remover_linhas = remover_linhas or set()

    novas_linhas = []
    alterou = False

    for linha_idx, linha_original in enumerate(linhas):
        linha = str(linha_original).strip()

        if linha_idx >= indice_corte or not linha:
            novas_linhas.append(linha_original)
            continue

        if linha_idx in remover_linhas:
            alterou = True
            continue

        if linha_idx in edicoes_por_linha:
            novo_valor = str(edicoes_por_linha[linha_idx]).strip()
            if novo_valor != linha:
                alterou = True
            if novo_valor:
                novas_linhas.append(novo_valor)
            else:
                alterou = True
            continue

        novas_linhas.append(linha_original)

    return "\n".join(novas_linhas), alterou



def _get_pendencia_meta(tipo: str) -> dict[str, str]:
    mapa = {
        "bloqueio_base": {
            "grupo": "Bloqueios da base",
            "gravidade": "Bloqueia sorteio",
            "tone": "error",
            "acao_principal": "Editar registro da base",
            "apoio": "Corrija ou remova o registro aqui mesmo para liberar a revisão.",
        },
        "fora_base": {
            "grupo": "Fora da base",
            "gravidade": "Corrigir antes de seguir",
            "tone": "warning",
            "acao_principal": "Corrigir nome na lista",
            "apoio": "Você pode corrigir o nome, cadastrar na base ou remover o item.",
        },
        "duplicado_lista": {
            "grupo": "Duplicados na lista",
            "gravidade": "Revisar ocorrências",
            "tone": "warning",
            "acao_principal": "Revisar duplicidade",
            "apoio": "Edite as ocorrências ou remova a entrada indevida.",
        },
    }
    return mapa.get(
        tipo,
        {
            "grupo": "Pendência",
            "gravidade": "Revisar",
            "tone": "info",
            "acao_principal": "Revisar item",
            "apoio": "Confira os detalhes deste item antes de seguir.",
        },
    )



def _expandir_bloqueio_base_padrao(
    *,
    idx: int,
    nome: str,
    qtd_bloqueios_base: int,
    expandir_primeiro_bloqueio: bool,
    nome_foco: str | None,
) -> bool:
    return (
        qtd_bloqueios_base == 1
        or (expandir_primeiro_bloqueio and idx == 0)
        or nome_foco == nome
    )



def _build_resumo_revisao_topo(diagnostico: dict) -> dict[str, object]:
    qtd_bloqueios = len(diagnostico.get("nomes_bloqueados_base", []) or [])
    qtd_fora_base = len(diagnostico.get("nao_encontrados", []) or [])
    qtd_duplicados = len(diagnostico.get("duplicados", []) or [])
    qtd_aptos = int(diagnostico.get("total_validos") or 0)

    tem_pendencias = (qtd_bloqueios + qtd_fora_base + qtd_duplicados) > 0
    status_pronto = not tem_pendencias

    return {
        "qtd_bloqueios": qtd_bloqueios,
        "qtd_fora_base": qtd_fora_base,
        "qtd_duplicados": qtd_duplicados,
        "qtd_aptos": qtd_aptos,
        "tem_pendencias": tem_pendencias,
        "status_pronto": status_pronto,
        "status_label": "Lista pronta para seguir" if status_pronto else "Corrigir antes de continuar",
        "acao_contextual": "Revisão concluída" if status_pronto else "Revise as pendências abaixo",
    }



def _faltantes_unicos_do_diagnostico(diagnostico: dict | None) -> list[str]:
    faltantes: list[str] = []
    if not diagnostico:
        return faltantes

    for nome in diagnostico.get("nao_encontrados", []) or []:
        nome_limpo = str(nome).strip()
        if nome_limpo and nome_limpo not in faltantes:
            faltantes.append(nome_limpo)
    return faltantes
