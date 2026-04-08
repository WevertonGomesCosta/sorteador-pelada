"""Primitivas visuais reutilizáveis do app."""

from __future__ import annotations

import html

import streamlit as st


def render_section_header(titulo: str, subtitulo: str | None = None):
    st.markdown(f"<div class='section-title'>{titulo}</div>", unsafe_allow_html=True)
    if subtitulo:
        st.markdown(f"<div class='section-subtitle'>{subtitulo}</div>", unsafe_allow_html=True)


def render_inline_status_note(
    titulo: str,
    descricao: str,
    *,
    tone: str = "info",
):
    st.markdown(
        f"""
        <div class="inline-status-note inline-status-note--{html.escape(tone)}">
            <span class="inline-status-note__title">{html.escape(titulo)}</span>
            <span class="inline-status-note__desc">{html.escape(descricao)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _titulo_expander(rotulo: str, status: str) -> str:
    return f"{rotulo} · {status}"


def render_app_meta_footer(
    *,
    desenvolvedor: str = "Weverton Gomes",
    portfolio_url: str = "https://wevertongomescosta.github.io/",
    versao: str = "v40",
    data_atualizacao: str = "08/04/2026",
):
    portfolio_url_safe = html.escape(portfolio_url, quote=True)
    st.markdown(
        f'<div class="app-meta-footer"><div class="app-meta-footer__title">Sobre este app</div><div class="app-meta-footer__text">Sorteador Pelada PRO · Desenvolvedor: {html.escape(desenvolvedor)} · Portfólio: <a class="app-meta-footer__link" href="{portfolio_url_safe}" target="_blank" rel="noopener noreferrer">{html.escape(portfolio_url)}</a> · Versão da base: {html.escape(versao)} · Última atualização: {html.escape(data_atualizacao)}</div></div>',
        unsafe_allow_html=True,
    )
