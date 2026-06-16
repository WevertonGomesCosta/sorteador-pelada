"""Primitivas visuais reutilizáveis do app."""

from __future__ import annotations

import html

import streamlit as st

APP_VERSION = "v129"
APP_LAST_UPDATED = "16 de junho de 2026"


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


def _data_ultima_atualizacao_projeto() -> str:
    """Retorna a data controlada da última release do código.

    A data do rodapé não deve depender do horário local de execução, deploy ou
    uso do app. Em ambientes hospedados, o `mtime` dos arquivos pode ser
    regravado no deploy e produzir uma data operacional falsa. Por isso, o
    metadado é fixado na release e atualizado junto com a versão.
    """
    return APP_LAST_UPDATED


def render_app_meta_footer(
    *,
    desenvolvedor: str = "Weverton Gomes da Costa",
    portfolio_url: str = "https://wevertongomescosta.github.io/",
    versao: str = "v129",
    data_atualizacao: str | None = None,
):
    portfolio_url_safe = html.escape(portfolio_url, quote=True)
    desenvolvedor_safe = html.escape(desenvolvedor)
    versao_safe = html.escape(versao)
    data_final = data_atualizacao or _data_ultima_atualizacao_projeto()
    data_safe = html.escape(data_final)
    st.markdown(
        (
            f'<div class="app-meta-footer">'
            f'<div class="app-meta-footer__title">Sobre este app</div>'
            f'<div class="app-meta-footer__subtitle">Sorteador Pelada PRO</div>'
            f'<div class="app-meta-footer__grid">'
            f'<div class="app-meta-footer__row"><span class="app-meta-footer__label">Aplicativo</span><span class="app-meta-footer__value">Sorteador Pelada PRO</span></div>'
            f'<div class="app-meta-footer__row"><span class="app-meta-footer__label">Desenvolvedor</span><span class="app-meta-footer__value">{desenvolvedor_safe}</span></div>'
            f'<div class="app-meta-footer__row"><span class="app-meta-footer__label">Portfólio</span><span class="app-meta-footer__value"><a class="app-meta-footer__link" href="{portfolio_url_safe}" target="_blank" rel="noopener noreferrer">{html.escape(portfolio_url)}</a></span></div>'
            f'<div class="app-meta-footer__row"><span class="app-meta-footer__label">Versão da base</span><span class="app-meta-footer__value">{versao_safe}</span></div>'
            f'</div>'
            f'<div class="app-meta-footer__legal">'
            f'<span class="app-meta-footer__legal-item">Política de privacidade</span>'
            f'<span class="app-meta-footer__legal-item">Licença CC BY-SA 4.0</span>'
            f'</div>'
            f'<div class="app-meta-footer__meta-grid">'
            f'<div class="app-meta-footer__meta">© 2026 {desenvolvedor_safe}</div>'
            f'<div class="app-meta-footer__meta">Última atualização: {data_safe}</div>'
            f'</div>'
            f'</div>'
        ),
        unsafe_allow_html=True,
    )
