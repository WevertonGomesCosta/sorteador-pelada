from tests._smoke_shared import *


class GoleirosSmokeTestCase(unittest.TestCase):
    def _logic(self):
        import importlib
        import types

        sys.modules.setdefault("pulp", types.SimpleNamespace())
        logic_module = importlib.import_module("core.logic")
        return logic_module.PeladaLogic()

    def test_processar_lista_mantem_goleiros_fora_por_padrao(self) -> None:
        texto_lista = """
1- Ana
2- Bruno
3- Carla
Goleiros:
1- Goleiro Um
2- Goleiro Dois
3- Goleiro Tres
Lista de espera:
1- Espera
"""
        processamento = self._logic().processar_lista(
            texto_lista,
            return_metadata=True,
            emit_warning=False,
            incluir_goleiros=False,
        )

        self.assertEqual(processamento["jogadores"], ["Ana", "Bruno", "Carla"])
        self.assertEqual(
            processamento["goleiros_lidos"],
            ["Goleiro Um", "Goleiro Dois", "Goleiro Tres"],
        )
        self.assertFalse(processamento["goleiros_incluidos"])

    def test_processar_lista_inclui_exatamente_tres_goleiros_quando_parametro_ativo(self) -> None:
        texto_lista = """
1- Ana
2- Bruno
3- Carla
Goleiros:
1- Goleiro Um
2- Goleiro Dois
3- Goleiro Tres
Lista de espera:
1- Espera
"""
        processamento = self._logic().processar_lista(
            texto_lista,
            return_metadata=True,
            emit_warning=False,
            incluir_goleiros=True,
        )

        self.assertEqual(
            processamento["jogadores"],
            ["Ana", "Bruno", "Carla", "Goleiro Um", "Goleiro Dois", "Goleiro Tres"],
        )
        self.assertTrue(processamento["goleiros_incluidos"])
        self.assertEqual(processamento["qtd_goleiros_lidos"], 3)

    def test_processar_lista_nao_inclui_goleiros_quando_quantidade_diferente_de_tres(self) -> None:
        texto_lista = """
1- Ana
2- Bruno
Goleiros:
1- Goleiro Um
2- Goleiro Dois
"""
        processamento = self._logic().processar_lista(
            texto_lista,
            return_metadata=True,
            emit_warning=False,
            incluir_goleiros=True,
        )

        self.assertEqual(processamento["jogadores"], ["Ana", "Bruno"])
        self.assertEqual(processamento["qtd_goleiros_lidos"], 2)
        self.assertFalse(processamento["goleiros_incluidos"])

    def test_diagnostico_inclui_goleiros_com_notas_da_base_quando_parametro_ativo(self) -> None:
        texto_lista = """
1- Ana
2- Bruno
3- Carla
Goleiros:
1- Goleiro Um
2- Goleiro Dois
3- Goleiro Tres
"""
        df_base = pd.DataFrame([
            {"Nome": "Ana", "Nota": 8, "Posição": "M", "Velocidade": 4, "Movimentação": 3},
            {"Nome": "Bruno", "Nota": 7, "Posição": "D", "Velocidade": 3, "Movimentação": 3},
            {"Nome": "Carla", "Nota": 9, "Posição": "A", "Velocidade": 5, "Movimentação": 4},
            {"Nome": "Goleiro Um", "Nota": 8, "Posição": "G", "Velocidade": 3, "Movimentação": 3},
            {"Nome": "Goleiro Dois", "Nota": 7, "Posição": "G", "Velocidade": 3, "Movimentação": 2},
            {"Nome": "Goleiro Tres", "Nota": 6, "Posição": "G", "Velocidade": 2, "Movimentação": 2},
        ])

        with patch.object(criteria_state, "st", SimpleNamespace(session_state={"sortear_goleiros": True})):
            diagnostico = self._logic().diagnosticar_lista_para_sorteio(texto_lista, df_base, [])

        self.assertTrue(diagnostico["goleiros_incluidos"])
        self.assertEqual(diagnostico["qtd_goleiros_lidos"], 3)
        self.assertEqual(diagnostico["nao_encontrados"], [])
        self.assertEqual(diagnostico["total_brutos"], 6)
        self.assertEqual(diagnostico["total_validos"], 6)

    def test_assinatura_muda_quando_sortear_goleiros_muda(self) -> None:
        df_base = pd.DataFrame([
            {"Nome": "Ana", "Nota": 8, "Posição": "M", "Velocidade": 4, "Movimentação": 3},
            {"Nome": "Goleiro Um", "Nota": 8, "Posição": "G", "Velocidade": 3, "Movimentação": 3},
        ])
        sessao_fluxo = {K.DF_BASE: df_base, K.NOVOS_JOGADORES: []}

        with patch.object(flow_guard, "st", SimpleNamespace(session_state=sessao_fluxo)):
            with patch.object(criteria_state, "st", SimpleNamespace(session_state={"sortear_goleiros": False})):
                assinatura_a = construir_assinatura_entrada_sorteio("Ana\nGoleiros:\n1- Goleiro Um", 2)
            with patch.object(criteria_state, "st", SimpleNamespace(session_state={"sortear_goleiros": True})):
                assinatura_b = construir_assinatura_entrada_sorteio("Ana\nGoleiros:\n1- Goleiro Um", 2)

        self.assertNotEqual(assinatura_a, assinatura_b)
