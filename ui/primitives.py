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
    desenvolvedor: str = "Weverton Gomes da Costa",
    portfolio_url: str = "https://wevertongomescosta.github.io/",
    versao: str = "v44",
    data_atualizacao: str = "20 de fevereiro de 2026",
):
    portfolio_url_safe = html.escape(portfolio_url, quote=True)
    desenvolvedor_safe = html.escape(desenvolvedor)
    versao_safe = html.escape(versao)
    data_safe = html.escape(data_atualizacao)
    st.markdown(
        (
            f'<div class="app-meta-footer">'
            f'<div class="app-meta-footer__title">Sobre este app</div>'
            f'<div class="app-meta-footer__text">Sorteador Pelada PRO · Desenvolvedor: {desenvolvedor_safe}</div>'
            f'<div class="app-meta-footer__text">Portfólio: <a class="app-meta-footer__link" href="{portfolio_url_safe}" target="_blank" rel="noopener noreferrer">{html.escape(portfolio_url)}</a> · Versão da base: {versao_safe}</div>'
            f'<div class="app-meta-footer__meta">Política de Privacidade | Licença CC BY-SA 4.0</div>'
            f'<div class="app-meta-footer__meta">© 2026 {desenvolvedor_safe}</div>'
            f'<div class="app-meta-footer__meta">Última atualização: {data_safe}</div>'
            f'</div>'
        ),
        unsafe_allow_html=True,
    )
