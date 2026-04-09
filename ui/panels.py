"""Painéis visuais reutilizáveis do app."""

from __future__ import annotations

import html

import streamlit as st


def render_step_cta_panel(
    titulo: str,
    descricao: str,
    *,
    tone: str = "info",
    eyebrow: str = "Próximo passo",
):
    st.markdown(
        f"""
        <div class="step-cta-panel step-cta-panel--{tone}">
            <div class="step-cta-panel__eyebrow">{eyebrow}</div>
            <div class="step-cta-panel__title">{titulo}</div>
            <div class="step-cta-panel__desc">{descricao}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_session_status_panel(
    *,
    modo_atual: str,
    base_status: str,
    lista_status: str,
    fluxo_status: str,
    proxima_acao: str,
):
    dados = [
        ("Modo", modo_atual),
        ("Base", base_status),
        ("Lista", lista_status),
        ("Fluxo", fluxo_status),
    ]
    itens_html = "".join(
        f'<div class="session-status-panel__item">'
        f'<div class="session-status-panel__label">{html.escape(rotulo)}</div>'
        f'<div class="session-status-panel__value">{html.escape(valor)}</div>'
        f'</div>'
        for rotulo, valor in dados
    )

    panel_html = (
        '<div class="session-status-panel">'
        '<div class="session-status-panel__eyebrow">Resumo da sessão</div>'
        f'<div class="session-status-panel__grid">{itens_html}</div>'
        '<div class="session-status-panel__next">'
        '<span class="session-status-panel__next-label">Próximo passo:</span>'
        f'<span class="session-status-panel__next-value">{html.escape(proxima_acao)}</span>'
        '</div>'
        '</div>'
    )

    st.markdown(panel_html, unsafe_allow_html=True)
