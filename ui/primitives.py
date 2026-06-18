"""Primitivas visuais reutilizáveis do app."""

from __future__ import annotations

import html
import os
import subprocess
from datetime import date
from functools import lru_cache
from pathlib import Path

import streamlit as st

APP_VERSION_FALLBACK = "v129"
APP_LAST_UPDATED_FALLBACK = "16 de junho de 2026"
ROOT_DIR = Path(__file__).resolve().parents[1]

MESES_PT_BR = {
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


@lru_cache(maxsize=16)
def _executar_git(args: tuple[str, ...]) -> str | None:
    try:
        resultado = subprocess.run(
            ["git", *args],
            cwd=ROOT_DIR,
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None

    saida = resultado.stdout.strip()
    return saida or None


def _formatar_data_iso_pt_br(data_iso: str | None) -> str | None:
    if not data_iso:
        return None

    try:
        data = date.fromisoformat(data_iso[:10])
    except ValueError:
        return None

    mes = MESES_PT_BR.get(data.month)
    if not mes:
        return None

    return f"{data.day} de {mes} de {data.year}"


@lru_cache(maxsize=1)
def _versao_atual_projeto() -> str:
    """Retorna uma versão derivada do histórico Git do código.

    O número muda automaticamente a cada commit porque usa a contagem de commits
    do repositório. Em ambientes sem Git disponível, preserva um fallback estável.
    """
    versao_env = os.getenv("SORTEADOR_APP_VERSION", "").strip()
    if versao_env:
        return versao_env

    total_commits = _executar_git(("rev-list", "--count", "HEAD"))
    if total_commits and total_commits.isdigit():
        return f"v{total_commits}"

    short_sha = _executar_git(("rev-parse", "--short", "HEAD"))
    if short_sha:
        return f"git-{short_sha}"

    return APP_VERSION_FALLBACK


@lru_cache(maxsize=1)
def _data_ultima_atualizacao_projeto() -> str:
    """Retorna a data do último commit do código.

    A data exibida no rodapé passa a acompanhar alterações reais versionadas no
    repositório, em vez de depender de atualização manual ou do horário local de
    execução/deploy.
    """
    data_env = os.getenv("SORTEADOR_APP_LAST_UPDATED", "").strip()
    if data_env:
        return data_env

    data_commit = _executar_git(("log", "-1", "--format=%cs", "HEAD"))
    data_formatada = _formatar_data_iso_pt_br(data_commit)
    if data_formatada:
        return data_formatada

    return APP_LAST_UPDATED_FALLBACK


def render_app_meta_footer(
    *,
    desenvolvedor: str = "Weverton Gomes da Costa",
    portfolio_url: str = "https://wevertongomescosta.github.io/",
    versao: str | None = None,
    data_atualizacao: str | None = None,
):
    portfolio_url_safe = html.escape(portfolio_url, quote=True)
    desenvolvedor_safe = html.escape(desenvolvedor)
    versao_final = versao or _versao_atual_projeto()
    versao_safe = html.escape(versao_final)
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
            f'<div class="app-meta-footer__row"><span class="app-meta-footer__label">Versão do app</span><span class="app-meta-footer__value">{versao_safe}</span></div>'
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
