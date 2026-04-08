"""Componentes de ação reutilizáveis do app."""

from __future__ import annotations

import streamlit as st


def render_action_button(
    label: str,
    *,
    key: str,
    role: str = "secondary",
    disabled: bool = False,
    use_primary_type: bool = False,
):
    with st.container(key=f"action-{role}-{key}"):
        button_type = "primary" if use_primary_type else "secondary"
        return st.button(
            label,
            key=key,
            disabled=disabled,
            type=button_type,
        )
