"""Componentes visuais reutilizáveis da revisão da lista."""

from __future__ import annotations

import html

import streamlit as st

import state.keys as K
from ui.primitives import render_inline_status_note


def _render_resumo_revisao_visual(total_brutos: int, total_validos: int, qtd_correcoes: int, total_pendencias: int, *, compacto: bool = False) -> None:
    if compacto:
        st.markdown(
            f"""
            <div class="review-summary-inline">
                <span class="review-summary-inline__label">Resumo rápido</span>
                <span class="review-summary-inline__item">Lidos: <strong>{total_brutos}</strong></span>
                <span class="review-summary-inline__item">Prontos: <strong>{total_validos}</strong></span>
                <span class="review-summary-inline__item">Ajustes: <strong>{qtd_correcoes}</strong></span>
                <span class="review-summary-inline__item">Pendências: <strong>{total_pendencias}</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    cards = [
        ("Lidos", str(total_brutos)),
        ("Prontos", str(total_validos)),
        ("Ajustes", str(qtd_correcoes)),
        ("Pendências", str(total_pendencias)),
    ]
    cards_html = "".join(
        f'<div class="review-summary-card"><div class="review-summary-card__label">{html.escape(rotulo)}</div><div class="review-summary-card__value">{html.escape(valor)}</div></div>'
        for rotulo, valor in cards
    )
    st.markdown(
        f"""
        <div class="review-summary-grid">
            {cards_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_revisao_status_banner(mensagem: str, *, tone: str) -> None:
    titulo = {
        "success": "Revisão pronta",
        "warning": "Atenção na revisão",
        "error": "Correção necessária",
        "info": "Status da revisão",
    }.get(tone, "Status da revisão")
    render_inline_status_note(titulo, mensagem, tone=tone)


def _render_lista_final_preview(rotulo: str, itens: list[str]) -> None:
    linhas = []
    for item in itens:
        texto = str(item or "").strip()
        if not texto:
            continue
        linhas.append(f'<div class="review-list-preview__line">{html.escape(texto)}</div>')

    if not linhas:
        linhas.append('<div class="review-list-preview__line review-list-preview__line--muted">Nenhum nome pronto para exibir.</div>')

    conteudo = "".join(linhas)
    st.markdown(
        f"""
        <div class="review-list-preview">
            <div class="review-list-preview__label">{html.escape(rotulo)}</div>
            <div class="review-list-preview__body">{conteudo}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def _render_pendencia_item_intro(
    tipo: str,
    nome_item: str,
    *,
    detalhe: str | None = None,
    acao_principal: str | None = None,
) -> None:
    meta = _get_pendencia_meta(tipo)
    acao = str(acao_principal or meta["acao_principal"]).strip()

    render_inline_status_note(
        meta["gravidade"],
        f"{nome_item} — Próxima ação: {acao}",
        tone=meta["tone"],
    )

    if detalhe:
        st.caption(detalhe)
    else:
        st.caption(meta["apoio"])


def _expandir_bloqueio_base_padrao(
    *,
    idx: int,
    nome: str,
    qtd_bloqueios_base: int,
    expandir_primeiro_bloqueio: bool,
) -> bool:
    nome_foco = st.session_state.get(K.REVISAO_FOCO_BLOQUEIO_NOME)
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
