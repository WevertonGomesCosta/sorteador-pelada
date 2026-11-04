"""Regressao da migracao para APIs atuais do Streamlit.

Os testes verificam chamadas estaticas e a construcao dos botoes. A rolagem
e os controles de navegador ainda devem ser validados na interface real.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._smoke_shared import *  # noqa: F401,F403
from ui import components as ui_components


ROOT = Path(__file__).resolve().parents[1]
ARQUIVOS_IFRAME = {
    "app.py": 4,
    "ui/group_config_view.py": 1,
    "ui/components.py": 2,
}


def _chamadas(tree: ast.AST, objeto: str, metodo: str) -> list[ast.Call]:
    return [
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and isinstance(item.func.value, ast.Name)
        and item.func.value.id == objeto
        and item.func.attr == metodo
    ]


class StreamlitCompatSmokeTestCase(unittest.TestCase):
    def test_nenhuma_chamada_da_api_obsoleta(self) -> None:
        for arquivo, esperado in ARQUIVOS_IFRAME.items():
            with self.subTest(arquivo=arquivo):
                codigo = (ROOT / arquivo).read_text(encoding="utf-8")
                tree = ast.parse(codigo, filename=arquivo)
                self.assertNotIn("streamlit.components.v1", codigo)
                self.assertEqual(_chamadas(tree, "components", "html"), [])
                self.assertEqual(len(_chamadas(tree, "st", "iframe")), esperado)

    def test_rolagem_e_botoes_conservam_dimensoes(self) -> None:
        for arquivo, esperado in ARQUIVOS_IFRAME.items():
            tree = ast.parse((ROOT / arquivo).read_text(encoding="utf-8"))
            for chamada in _chamadas(tree, "st", "iframe"):
                with self.subTest(arquivo=arquivo, linha=chamada.lineno):
                    kwargs = {item.arg: item.value for item in chamada.keywords}
                    self.assertIn("height", kwargs)
                    altura = kwargs["height"]
                    self.assertIsInstance(altura, ast.Constant)
                    self.assertEqual(
                        altura.value,
                        70 if arquivo == "ui/components.py" else 0,
                    )
                    self.assertIn("alt", kwargs)
                    if arquivo != "ui/components.py":
                        self.assertIn("tab_index", kwargs)
                        self.assertIsInstance(kwargs["tab_index"], ast.UnaryOp)

    def test_botoes_copiar_e_compartilhar_preservam_scripts(self) -> None:
        renderizacoes = []

        def registrar_arquivo(html, **kwargs):
            renderizacoes.append((html, kwargs))

        with patch.object(
            ui_components.st, "iframe", side_effect=registrar_arquivo, create=True
        ):
            ui_components.botao_copiar_js("Hiury\nOtávio")
            ui_components.botao_compartilhar_js("Time 1: Hiury")

        self.assertEqual(len(renderizacoes), 2)
        html_copia, opts_copia = renderizacoes[0]
        html_compartilhar, opts_compartilhar = renderizacoes[1]
        self.assertIn("function copiarTexto()", html_copia)
        self.assertIn("Hiury", html_copia)
        self.assertIn(json.dumps("Hiury\nOtávio"), html_copia)
        self.assertIn("function compartilharTexto()", html_compartilhar)
        self.assertIn("Time 1: Hiury", html_compartilhar)
        self.assertEqual(opts_copia["height"], 70)
        self.assertEqual(opts_compartilhar["height"], 70)

    def test_multiselect_usa_estado_sem_default_duplicado(self) -> None:
        codigo = (ROOT / "ui/review_view.py").read_text(encoding="utf-8")
        tree = ast.parse(codigo)
        chamadas = [
            c for c in _chamadas(tree, "st", "multiselect")
            if c.args
            and isinstance(c.args[0], ast.Constant)
            and c.args[0].value == "Remover nomes da lista revisada"
        ]
        self.assertEqual(len(chamadas), 1)
        kwargs = {kw.arg: kw.value for kw in chamadas[0].keywords}
        self.assertNotIn("default", kwargs)
        self.assertIsInstance(kwargs["key"], ast.Name)
        self.assertEqual(kwargs["key"].id, "remover_key")
        self.assertIn("st.session_state[remover_key] = []", codigo)

    def test_dependencia_declara_streamlit_compativel(self) -> None:
        requisitos = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("streamlit>=1.56,<2", requisitos)
