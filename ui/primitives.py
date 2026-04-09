"""Primitivas visuais reutilizáveis do app."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

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


def _data_ultima_atualizacao_projeto() -> str:
    raiz = Path(__file__).resolve().parents[1]
    extensoes = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".css", ".Rproj"}
    meses = {
        1: "janeiro",
        2: "fevereiro",
        3: "março",
        4: "abril",
        5: "maio",
        6: "junho",
        7: "julho",
        8: "agosto",
        9: "setembro",
        10: "outubro",
        11: "novembro",
        12: "dezembro",
    }

    arquivos = [
        p
        for p in raiz.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix in extensoes
        and not any(parte.startswith(".") for parte in p.relative_to(raiz).parts)
    ]
    if not arquivos:
        return "8 de abril de 2026"

    mais_recente = max(arquivos, key=lambda p: p.stat().st_mtime)
    dt = datetime.fromtimestamp(mais_recente.stat().st_mtime)
    return f"{dt.day} de {meses[dt.month]} de {dt.year}"


def render_app_meta_footer(
    *,
    desenvolvedor: str = "Weverton Gomes da Costa",
    portfolio_url: str = "https://wevertongomescosta.github.io/",
    versao: str = "v56",
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
