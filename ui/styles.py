"""Camada centralizada de estilos do app em base neutra.

Nesta etapa, removemos cores customizadas para deixar o Streamlit governar
o tema. Permanecem apenas regras de layout, espaçamento, tamanho e borda
baseadas em herança/currentColor.
"""

import streamlit as st

from ui.style_action_css import ACTION_BUTTON_CSS
from ui.style_base_css import APP_BASE_CSS


def apply_app_styles():
    st.markdown(
        f"""<style>{APP_BASE_CSS}
{ACTION_BUTTON_CSS}</style>""",
        unsafe_allow_html=True,
    )
