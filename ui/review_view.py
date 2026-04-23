"""Fluxos públicos de revisão da lista e correções inline do Sorteador Pelada PRO.

Este módulo agora atua como fachada estável da revisão. A implementação foi
quebrada em subblocos menores para reduzir risco de regressão:
- ui/review_text_ops.py
- ui/review_components.py
- ui/review_pending.py
- ui/review_cadastro.py
- ui/review_main.py
"""

from __future__ import annotations

from ui.review_main import render_revisao_lista_impl as _render_revisao_lista_impl
from ui.review_pending import (
    render_correcao_inline_bloqueios_base_impl as _render_correcao_inline_bloqueios_base_impl,
    render_correcao_inline_etapa2_impl as _render_correcao_inline_etapa2_impl,
    render_revisao_pendencias_panel_impl as _render_revisao_pendencias_panel_impl,
)


def render_revisao_pendencias_panel(*args, **kwargs):
    return _render_revisao_pendencias_panel_impl(*args, **kwargs)


def render_correcao_inline_bloqueios_base(*args, **kwargs):
    return _render_correcao_inline_bloqueios_base_impl(*args, **kwargs)


def render_correcao_inline_etapa2(*args, **kwargs):
    return _render_correcao_inline_etapa2_impl(*args, **kwargs)


def render_revisao_lista(*args, **kwargs):
    return _render_revisao_lista_impl(*args, **kwargs)
