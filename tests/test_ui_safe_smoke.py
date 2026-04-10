from tests._smoke_shared import *


class UiSafeSmokeTestCase(unittest.TestCase):
    def test_summary_strings_refletem_estado_minimo(self) -> None:
        sessao_config = {
            "base_admin_carregada": True,
            "is_admin": True,
            "ultimo_arquivo": None,
            "grupo_origem_fluxo": "grupo",
            "grupo_busca_status": "idle",
        }
        with patch.object(summary_strings, "st", SimpleNamespace(session_state=sessao_config)):
            self.assertEqual(summary_strings.resumo_expander_configuracao("Pelada UFV"), "⚙️ Grupo e base · Base do grupo carregada")

        sessao_cadastro = {
            "cadastro_guiado_ativo": False,
            "revisao_pendente_pos_cadastro": True,
            "faltantes_cadastrados_na_rodada": ["Ana"],
            "qtd_jogadores_adicionados_manualmente": 1,
        }
        with patch.object(summary_strings, "st", SimpleNamespace(session_state=sessao_cadastro)):
            self.assertEqual(summary_strings.resumo_expander_cadastro_manual(), "📝 Cadastro manual · Faltantes cadastrados")

        with patch.object(summary_strings, "st", SimpleNamespace(session_state={})):
            with patch.object(criteria_state, "st", SimpleNamespace(session_state={"criterio_nota": False})):
                self.assertEqual(summary_strings.resumo_expander_criterios(), "⚙️ Critérios · Personalizado")

    def test_session_status_panel_escapa_conteudo_dinamico(self) -> None:
        chamadas = []

        def fake_markdown(html, unsafe_allow_html=False):
            chamadas.append((html, unsafe_allow_html))

        with patch.object(panels, "st", SimpleNamespace(markdown=fake_markdown)):
            panels.render_session_status_panel(
                modo_atual="<b>lista</b>",
                base_status="Base & teste",
                lista_status="3 nomes <script>",
                fluxo_status="Pronto > revisar",
                proxima_acao='Clicar em "🎲 SORTEAR TIMES"',
            )

        self.assertEqual(len(chamadas), 1)
        html_renderizado, unsafe = chamadas[0]
        self.assertTrue(unsafe)
        self.assertIn("&lt;b&gt;lista&lt;/b&gt;", html_renderizado)
        self.assertIn("Base &amp; teste", html_renderizado)
        self.assertIn("3 nomes &lt;script&gt;", html_renderizado)
        self.assertIn("Pronto &gt; revisar", html_renderizado)
        self.assertIn("Clicar em &quot;🎲 SORTEAR TIMES&quot;", html_renderizado)
