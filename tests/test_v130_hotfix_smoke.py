from tests._smoke_shared import *


def importar_result_view_com_componentes_mockados():
    import importlib
    import types

    componentes_mock = types.ModuleType("ui.components")
    componentes_mock.botao_compartilhar_js = lambda *args, **kwargs: None
    componentes_mock.botao_copiar_js = lambda *args, **kwargs: None

    sys.modules.pop("ui.result_view", None)
    with patch.dict(sys.modules, {"ui.components": componentes_mock}):
        return importlib.import_module("ui.result_view")


class V130HotfixSmokeTestCase(unittest.TestCase):
    def test_registro_valido_para_sorteio_aceita_posicao_g(self) -> None:
        row = pd.Series({
            "Nome": "Goleiro Um",
            "Nota": 8,
            "Posição": "G",
            "Velocidade": 3,
            "Movimentação": 3,
        })

        self.assertTrue(validators.registro_valido_para_sorteio(row))

    def test_preparar_df_sorteio_inclui_goleiro_valido(self) -> None:
        df_base = pd.DataFrame([
            {"Nome": "Ana", "Nota": 8, "Posição": "M", "Velocidade": 4, "Movimentação": 3},
            {"Nome": "Goleiro Um", "Nota": 7, "Posição": "G", "Velocidade": 3, "Movimentação": 2},
        ])

        df_jogar, bloqueios = validators.preparar_df_sorteio(df_base, ["Ana", "Goleiro Um"])

        self.assertEqual(bloqueios, [])
        self.assertEqual(set(df_jogar["Nome"].tolist()), {"Ana", "Goleiro Um"})
        self.assertEqual(df_jogar[df_jogar["Nome"] == "Goleiro Um"].iloc[0]["Posição"], "G")

    def test_alternar_sortear_goleiros_invalida_revisao_anterior(self) -> None:
        sessao = {
            K.DIAGNOSTICO_LISTA: {"sortear_goleiros": False},
            K.LISTA_REVISADA: ["Ana"],
            K.LISTA_REVISADA_CONFIRMADA: True,
            K.LISTA_TEXTO_REVISADO: "Ana",
            K.FALTANTES_REVISAO: [],
            K.CADASTRO_GUIADO_ATIVO: False,
            K.REVISAO_PENDENTE_POS_CADASTRO: False,
            K.RESULTADO: [[ ["Ana", 8, "M", 3, 3] ]],
            K.RESULTADO_CONTEXTO: {"modo_sorteio": "balanceado"},
            K.RESULTADO_ASSINATURA: "assinatura_antiga",
            K.RESULTADO_INVALIDADO_MSG: False,
            K.SCROLL_PARA_RESULTADO: True,
            K.DF_BASE: pd.DataFrame([
                {"Nome": "Ana", "Nota": 8, "Posição": "M", "Velocidade": 3, "Movimentação": 3},
            ]),
            K.NOVOS_JOGADORES: [],
            "sortear_goleiros": True,
        }

        with patch.object(flow_guard, "st", SimpleNamespace(session_state=sessao)):
            with patch.object(criteria_state, "st", SimpleNamespace(session_state=sessao)):
                flow_guard.invalidar_resultado_se_entrada_mudou("Ana", 2)

        self.assertIsNone(sessao[K.DIAGNOSTICO_LISTA])
        self.assertIsNone(sessao[K.LISTA_REVISADA])
        self.assertFalse(sessao[K.LISTA_REVISADA_CONFIRMADA])
        self.assertNotIn(K.RESULTADO, sessao)
        self.assertNotIn(K.RESULTADO_CONTEXTO, sessao)
        self.assertTrue(sessao[K.RESULTADO_INVALIDADO_MSG])

    def test_render_result_summary_panel_mostra_capitao_ativo_por_contexto(self) -> None:
        result_view = importar_result_view_com_componentes_mockados()

        chamadas = []

        def fake_markdown(html, unsafe_allow_html=False):
            chamadas.append((html, unsafe_allow_html))

        sessao = {
            "resultado_contexto": {"sortear_capitao": True},
            "diagnostico_lista": {},
        }

        with patch.object(result_view, "st", SimpleNamespace(session_state=sessao, markdown=fake_markdown)):
            result_view.render_result_summary_panel(
                qtd_jogadores_resultado=6,
                qtd_times_resultado=2,
                modo_criterios="Padrão",
                criterios_ativos_texto="Todos os 4 critérios",
                modo_sorteio="balanceado",
            )

        self.assertEqual(len(chamadas), 1)
        self.assertIn("Capitão:", chamadas[0][0])
        self.assertIn("Ativo", chamadas[0][0])

    def test_render_result_summary_panel_mostra_status_goleiros(self) -> None:
        result_view = importar_result_view_com_componentes_mockados()

        chamadas = []

        def fake_markdown(html, unsafe_allow_html=False):
            chamadas.append((html, unsafe_allow_html))

        sessao = {
            "resultado_contexto": {
                "sortear_goleiros": True,
                "goleiros_incluidos": True,
                "qtd_goleiros_lidos": 3,
            },
            "diagnostico_lista": {},
        }

        with patch.object(result_view, "st", SimpleNamespace(session_state=sessao, markdown=fake_markdown)):
            result_view.render_result_summary_panel(
                qtd_jogadores_resultado=15,
                qtd_times_resultado=3,
                modo_criterios="Padrão",
                criterios_ativos_texto="Todos os 4 critérios",
                modo_sorteio="balanceado",
            )

        html_renderizado = chamadas[0][0]
        self.assertIn("Goleiros:", html_renderizado)
        self.assertIn("Ativo · 3 lido(s) · incluídos", html_renderizado)
