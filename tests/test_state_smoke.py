from tests._smoke_shared import *


class StateSmokeTestCase(unittest.TestCase):
    def test_criterios_ativos_padrao_e_personalizado(self) -> None:
        sessao_padrao = {}
        sessao_personalizada = {
            "criterio_posicao": False,
            "criterio_nota": False,
            "criterio_velocidade": True,
            "criterio_movimentacao": False,
        }

        with patch.object(criteria_state, "st", SimpleNamespace(session_state=sessao_padrao)):
            self.assertEqual(criteria_state.obter_criterios_ativos(), {"pos": True, "nota": True, "vel": True, "mov": True})
            self.assertEqual(criteria_state.resumo_criterios_ativos(), "Padrão · Posição, Nota, Velocidade e Movimentação")

        with patch.object(criteria_state, "st", SimpleNamespace(session_state=sessao_personalizada)):
            self.assertEqual(criteria_state.obter_criterios_ativos(), {"pos": False, "nota": False, "vel": True, "mov": False})
            self.assertEqual(criteria_state.resumo_criterios_ativos(), "Personalizado · Velocidade")

    def test_view_models_interpretam_estado_visual_minimo(self) -> None:
        self.assertTrue(
            view_models.determinar_visibilidade_revisao(
                diagnostico_disponivel=False,
                lista_confirmada=False,
                cadastro_guiado_ativo=True,
                revisao_pendente_pos_cadastro=False,
                faltantes_revisao_qtd=0,
                faltantes_cadastrados_qtd=0,
            )
        )
        etapa = view_models.determinar_etapa_visual_ativa(
            escolha_inicial_pendente=False,
            qtd_nomes=3,
            draft_lista="ana\nbruno\ncarla",
            lista_confirmada=False,
            resultado_disponivel=False,
            review_stage_visible=True,
            manual_section_visible=False,
            cadastro_guiado_ativo=False,
            revisao_pendente_pos_cadastro=False,
        )
        self.assertEqual(etapa, "revisao")
        status = view_models.construir_status_sessao_visual(
            origem_fluxo="lista",
            base_carregada_via_secao1=False,
            qtd_nomes=3,
            qtd_ignorados=1,
            diagnostico={"tem_nao_encontrados": False, "tem_duplicados": False, "tem_bloqueio_base": False},
            lista_revisada_ok=True,
            lista_confirmada_ok=False,
            base_pronta_ok=False,
            resultado_disponivel=False,
            cadastro_guiado_ativo=False,
            is_admin=False,
            ultimo_arquivo="",
            df_base_len=0,
            novos_jogadores_len=0,
        )
        self.assertEqual(status["modo_atual"], "Apenas sorteio com lista")
        self.assertEqual(status["fluxo_status"], "Confirmação pendente")
        self.assertIn("1 linha(s) ignorada(s)", status["lista_status"])
        blocos = view_models.construir_estado_blocos_visuais(
            etapa_visual_ativa="revisao",
            scroll_para_revisao=False,
            cadastro_guiado_ativo=False,
            revisao_pendente_pos_cadastro=False,
            cadastro_manual_nome_existente="",
        )
        self.assertTrue(blocos["review_stage_active_ui"])
        self.assertTrue(blocos["revisao_lista_expandida"])
