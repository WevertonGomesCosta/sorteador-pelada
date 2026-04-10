from tests._smoke_shared import *


class CoreSmokeTestCase(unittest.TestCase):
    def test_imports_centrais_sem_erro(self) -> None:
        import core.base_summary  # noqa: F401
        import core.flow_guard  # noqa: F401
        import state.criteria_state  # noqa: F401
        import state.view_models  # noqa: F401

    def test_base_summary_computa_total_e_resumo(self) -> None:
        inconsistencias = {
            "nomes_vazios": 1,
            "notas_invalidas": 2,
            "movimentacoes_invalidas": 0,
        }
        self.assertEqual(total_inconsistencias_base(inconsistencias), 3)
        self.assertEqual(
            resumo_inconsistencias_base(inconsistencias),
            "1 nome(s) vazio(s); 2 nota(s) fora da faixa 1–10",
        )

    def test_flow_guard_assinatura_muda_quando_criterio_muda(self) -> None:
        df_base = pd.DataFrame([
            {"Nome": "Ana", "Nota": 8, "Posição": "M", "Velocidade": 4, "Movimentação": 3},
            {"Nome": "Bruno", "Nota": 7, "Posição": "D", "Velocidade": 3, "Movimentação": 2},
        ])
        sessao_fluxo = {K.DF_BASE: df_base, K.NOVOS_JOGADORES: []}

        with patch.object(flow_guard, "st", SimpleNamespace(session_state=sessao_fluxo)):
            with patch.object(criteria_state, "st", SimpleNamespace(session_state={})):
                assinatura_a = construir_assinatura_entrada_sorteio("Ana\nBruno", 2)
            with patch.object(criteria_state, "st", SimpleNamespace(session_state={"criterio_movimentacao": False})):
                assinatura_b = construir_assinatura_entrada_sorteio("Ana\nBruno", 2)

        self.assertNotEqual(assinatura_a, assinatura_b)

    def test_contar_duplicados_base_atual_normaliza_nomes(self) -> None:
        df_base = pd.DataFrame([
            {"Nome": "José"},
            {"Nome": "Jose"},
            {"Nome": "MARIA"},
        ])
        self.assertEqual(contar_duplicados_base_atual(df_base), 1)

    def test_invalidar_resultado_limpa_estado_quando_entrada_muda(self) -> None:
        df_base = pd.DataFrame([
            {"Nome": "Ana", "Nota": 8, "Posição": "M", "Velocidade": 4, "Movimentação": 3}
        ])
        sessao = {
            K.DF_BASE: df_base,
            K.NOVOS_JOGADORES: [],
            K.RESULTADO: [["Time 1"]],
            K.RESULTADO_CONTEXTO: {"modo": "teste"},
            K.SCROLL_PARA_RESULTADO: True,
        }

        with patch.object(flow_guard, "st", SimpleNamespace(session_state=sessao)):
            with patch.object(criteria_state, "st", SimpleNamespace(session_state={})):
                sessao[K.RESULTADO_ASSINATURA] = construir_assinatura_entrada_sorteio("Ana", 2)
                invalidar_resultado_se_entrada_mudou("Ana\nBruno", 2)

        self.assertNotIn(K.RESULTADO, sessao)
        self.assertNotIn(K.RESULTADO_CONTEXTO, sessao)
        self.assertIsNone(sessao[K.RESULTADO_ASSINATURA])
        self.assertFalse(sessao[K.SCROLL_PARA_RESULTADO])
        self.assertTrue(sessao[K.RESULTADO_INVALIDADO_MSG])

    def test_gate_pre_sorteio_no_modo_aleatorio_lista(self) -> None:
        sessao = {
            K.DF_BASE: pd.DataFrame(),
            K.DIAGNOSTICO_LISTA: {},
            K.LISTA_TEXTO_REVISADO: "",
            K.LISTA_REVISADA_CONFIRMADA: False,
            K.LISTA_REVISADA: False,
            K.CADASTRO_GUIADO_ATIVO: False,
            K.NOVOS_JOGADORES: [],
        }
        logic = DummyLogic(["Ana", "Bruno", "Carla"])

        with patch.object(flow_guard, "st", SimpleNamespace(session_state=sessao)):
            with patch.object(criteria_state, "st", SimpleNamespace(session_state={})):
                gate = construir_gate_pre_sorteio(
                    logic=logic,
                    lista_texto="Ana\nBruno\nCarla",
                    qtd_nomes_informados=3,
                    n_times=2,
                )

        self.assertTrue(gate["pronto_para_sortear"])
        self.assertEqual(gate["modo_sorteio"], "aleatorio_lista")

    def test_validators_validam_registro_e_slider_basico(self) -> None:
        registro_valido = pd.Series({"Nome": "Ana", "Posição": "M", "Nota": 8, "Velocidade": 4, "Movimentação": 3})
        registro_invalido = pd.Series({"Nome": "", "Posição": "X", "Nota": 11, "Velocidade": 0, "Movimentação": 7})

        self.assertTrue(validators.registro_valido_para_sorteio(registro_valido))
        self.assertFalse(validators.registro_valido_para_sorteio(registro_invalido))
        self.assertEqual(validators.valor_slider_corrigir(4.6, 1, 5, 3), 5)
        self.assertEqual(validators.valor_slider_corrigir("invalido", 1, 5, 3), 3)

    def test_validators_diagnosticam_bloqueios_reais(self) -> None:
        df_base = pd.DataFrame([
            {"Nome": "Ana", "Nota": 8, "Posição": "M", "Velocidade": 4, "Movimentação": 3},
            {"Nome": "Ana", "Nota": 0, "Posição": "M", "Velocidade": 4, "Movimentação": 3},
            {"Nome": "Carla", "Nota": 7, "Posição": "D", "Velocidade": 2, "Movimentação": 2},
        ])
        bloqueios = validators.diagnosticar_nomes_bloqueados_para_sorteio(df_base, ["Ana", "Bruno"])
        self.assertEqual(
            bloqueios,
            [
                {"nome": "Ana", "motivos": ["duplicado na base", "com inconsistência na base"]},
                {"nome": "Bruno", "motivos": ["sem registro na base atual"]},
            ],
        )
