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
