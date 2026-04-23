"""Componentes visuais passivos da revisão da lista.

Os helpers aqui renderizam apenas HTML/markdown informativo. Eles não executam
st.form, st.button, st.rerun nem escrevem em session_state.
"""

from __future__ import annotations

import html

import streamlit as st

from ui.primitives import render_inline_status_note
from ui.review_helpers import _get_pendencia_meta


def _render_resumo_revisao_visual(
    total_brutos: int,
    total_validos: int,
    qtd_correcoes: int,
    total_pendencias: int,
    *,
    compacto: bool = False,
) -> None:
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



def _render_resumo_pre_sorteio_panel(resumo_topo: dict[str, object]) -> None:
    metricas = [
        ("Bloqueios", str(resumo_topo["qtd_bloqueios"])),
        ("Fora da base", str(resumo_topo["qtd_fora_base"])),
        ("Duplicados", str(resumo_topo["qtd_duplicados"])),
        ("Aptos", str(resumo_topo["qtd_aptos"])),
    ]

    metricas_html = "".join(
        f'<div class="review-pending-panel__metric">'
        f'<div class="review-pending-panel__metric-label">{html.escape(rotulo)}</div>'
        f'<div class="review-pending-panel__metric-value">{html.escape(valor)}</div>'
        f'</div>'
        for rotulo, valor in metricas
    )

    st.markdown(
        f"""
        <div class="review-pending-panel">
            <div class="review-pending-panel__eyebrow">Resumo pré-sorteio</div>
            <div class="review-pending-panel__title">{html.escape(str(resumo_topo['status_label']))}</div>
            <div class="review-pending-panel__desc">{html.escape(str(resumo_topo['acao_contextual']))}</div>
            <div class="review-pending-panel__metrics">{metricas_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def _render_lista_faltantes_identificados(faltantes: list[str]) -> None:
    if not faltantes:
        return
    st.markdown("**Atletas faltantes identificados nesta revisão**")
    for nome_faltante in faltantes:
        st.markdown(f"- `{nome_faltante}`")
