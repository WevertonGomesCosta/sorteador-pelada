"""Compatibilidade temporária para a configuração inicial do app.

As funções desta etapa foram movidas para ui.group_config_view.
"""

from ui.group_config_view import (
    abrir_expander_grupo,
    ativar_fluxo_somente_lista,
    grupo_config_deve_abrir,
    render_group_config_expander,
)

__all__ = [
    "abrir_expander_grupo",
    "ativar_fluxo_somente_lista",
    "grupo_config_deve_abrir",
    "render_group_config_expander",
]
