from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

sys.modules.setdefault("streamlit", SimpleNamespace(session_state={}))

from core.base_summary import resumo_inconsistencias_base, total_inconsistencias_base
from core import validators
from core.flow_guard import (
    construir_assinatura_entrada_sorteio,
    construir_gate_pre_sorteio,
    contar_duplicados_base_atual,
    invalidar_resultado_se_entrada_mudou,
)
from state import criteria_state, view_models
from ui import panels, summary_strings
import core.flow_guard as flow_guard
import state.keys as K


class DummyLogic:
    def __init__(self, jogadores: list[str], inconsistencias: dict | None = None) -> None:
        self._jogadores = jogadores
        self._inconsistencias = inconsistencias or {}

    def processar_lista(self, lista_texto: str, return_metadata: bool = True, emit_warning: bool = False) -> dict:
        return {"jogadores": list(self._jogadores)}

    def diagnosticar_inconsistencias_base(self, df_base: pd.DataFrame) -> dict:
        return dict(self._inconsistencias)


class SmokeBaseTestCase(unittest.TestCase):
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

    def test_criterios_ativos_padrao_e_personalizado(self) -> None:
        sessao_padrao = {}
        sessao_personalizada = {
            "criterio_posicao": False,
            "criterio_nota": False,
            "criterio_velocidade": True,
            "criterio_movimentacao": False,
        }

        with patch.object(criteria_state, "st", SimpleNamespace(session_state=sessao_padrao)):
            self.assertEqual(
                criteria_state.obter_criterios_ativos(),
                {"pos": True, "nota": True, "vel": True, "mov": True},
            )
            self.assertEqual(
                criteria_state.resumo_criterios_ativos(),
                "Padrão · Posição, Nota, Velocidade e Movimentação",
            )

        with patch.object(criteria_state, "st", SimpleNamespace(session_state=sessao_personalizada)):
            self.assertEqual(
                criteria_state.obter_criterios_ativos(),
                {"pos": False, "nota": False, "vel": True, "mov": False},
            )
            self.assertEqual(
                criteria_state.resumo_criterios_ativos(),
                "Personalizado · Velocidade",
            )

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
            diagnostico={
                "tem_nao_encontrados": False,
                "tem_duplicados": False,
                "tem_bloqueio_base": False,
            },
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

    def test_flow_guard_assinatura_muda_quando_criterio_muda(self) -> None:
        df_base = pd.DataFrame(
            [
                {"Nome": "Ana", "Nota": 8, "Posição": "M", "Velocidade": 4, "Movimentação": 3},
                {"Nome": "Bruno", "Nota": 7, "Posição": "D", "Velocidade": 3, "Movimentação": 2},
            ]
        )
        sessao_fluxo = {K.DF_BASE: df_base, K.NOVOS_JOGADORES: []}

        with patch.object(flow_guard, "st", SimpleNamespace(session_state=sessao_fluxo)):
            with patch.object(criteria_state, "st", SimpleNamespace(session_state={})):
                assinatura_a = construir_assinatura_entrada_sorteio("Ana\nBruno", 2)
            with patch.object(
                criteria_state,
                "st",
                SimpleNamespace(session_state={"criterio_movimentacao": False}),
            ):
                assinatura_b = construir_assinatura_entrada_sorteio("Ana\nBruno", 2)

        self.assertNotEqual(assinatura_a, assinatura_b)

    def test_contar_duplicados_base_atual_normaliza_nomes(self) -> None:
        df_base = pd.DataFrame(
            [
                {"Nome": "José"},
                {"Nome": "Jose"},
                {"Nome": "MARIA"},
            ]
        )
        self.assertEqual(contar_duplicados_base_atual(df_base), 1)

    def test_invalidar_resultado_limpa_estado_quando_entrada_muda(self) -> None:
        df_base = pd.DataFrame(
            [{"Nome": "Ana", "Nota": 8, "Posição": "M", "Velocidade": 4, "Movimentação": 3}]
        )
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


    def test_validators_validam_registro_e_slider_basico(self) -> None:
        registro_valido = pd.Series({
            "Nome": "Ana",
            "Posição": "M",
            "Nota": 8,
            "Velocidade": 4,
            "Movimentação": 3,
        })
        registro_invalido = pd.Series({
            "Nome": "",
            "Posição": "X",
            "Nota": 11,
            "Velocidade": 0,
            "Movimentação": 7,
        })

        self.assertTrue(validators.registro_valido_para_sorteio(registro_valido))
        self.assertFalse(validators.registro_valido_para_sorteio(registro_invalido))
        self.assertEqual(validators.valor_slider_corrigir(4.6, 1, 5, 3), 5)
        self.assertEqual(validators.valor_slider_corrigir("invalido", 1, 5, 3), 3)

    def test_validators_diagnosticam_bloqueios_reais(self) -> None:
        df_base = pd.DataFrame(
            [
                {"Nome": "Ana", "Nota": 8, "Posição": "M", "Velocidade": 4, "Movimentação": 3},
                {"Nome": "Ana", "Nota": 0, "Posição": "M", "Velocidade": 4, "Movimentação": 3},
                {"Nome": "Carla", "Nota": 7, "Posição": "D", "Velocidade": 2, "Movimentação": 2},
            ]
        )

        bloqueios = validators.diagnosticar_nomes_bloqueados_para_sorteio(df_base, ["Ana", "Bruno"])

        self.assertEqual(
            bloqueios,
            [
                {"nome": "Ana", "motivos": ["duplicado na base", "com inconsistência na base"]},
                {"nome": "Bruno", "motivos": ["sem registro na base atual"]},
            ],
        )

    def test_summary_strings_refletem_estado_minimo(self) -> None:
        sessao_config = {
            "base_admin_carregada": True,
            "is_admin": True,
            "ultimo_arquivo": None,
            "grupo_origem_fluxo": "grupo",
            "grupo_busca_status": "idle",
        }
        with patch.object(summary_strings, "st", SimpleNamespace(session_state=sessao_config)):
            self.assertEqual(
                summary_strings.resumo_expander_configuracao("Pelada UFV"),
                "⚙️ Grupo e base · Base do grupo carregada",
            )

        sessao_cadastro = {
            "cadastro_guiado_ativo": False,
            "revisao_pendente_pos_cadastro": True,
            "faltantes_cadastrados_na_rodada": ["Ana"],
            "qtd_jogadores_adicionados_manualmente": 1,
        }
        with patch.object(summary_strings, "st", SimpleNamespace(session_state=sessao_cadastro)):
            self.assertEqual(
                summary_strings.resumo_expander_cadastro_manual(),
                "📝 Cadastro manual · Faltantes cadastrados",
            )

        with patch.object(summary_strings, "st", SimpleNamespace(session_state={})):
            with patch.object(criteria_state, "st", SimpleNamespace(session_state={"criterio_nota": False})):
                self.assertEqual(
                    summary_strings.resumo_expander_criterios(),
                    "⚙️ Critérios · Personalizado",
                )

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
        self.assertEqual(gate["qtd_nomes_unicos_lista"], 3)
        self.assertIn("Modo aleatório por lista ativo", gate["avisos"][0])

    def test_gate_pre_sorteio_bloqueia_quando_lista_nao_foi_revisada(self) -> None:
        df_base = pd.DataFrame(
            [{"Nome": "Ana", "Nota": 8, "Posição": "M", "Velocidade": 4, "Movimentação": 3}]
        )
        sessao = {
            K.DF_BASE: df_base,
            K.DIAGNOSTICO_LISTA: {},
            K.LISTA_TEXTO_REVISADO: "",
            K.LISTA_REVISADA_CONFIRMADA: False,
            K.LISTA_REVISADA: False,
            K.CADASTRO_GUIADO_ATIVO: False,
            K.NOVOS_JOGADORES: [],
        }
        logic = DummyLogic(["Ana"])

        with patch.object(flow_guard, "st", SimpleNamespace(session_state=sessao)):
            with patch.object(criteria_state, "st", SimpleNamespace(session_state={})):
                gate = construir_gate_pre_sorteio(
                    logic=logic,
                    lista_texto="Ana",
                    qtd_nomes_informados=1,
                    n_times=1,
                )

        self.assertFalse(gate["pronto_para_sortear"])
        self.assertEqual(gate["modo_sorteio"], "balanceado")
        self.assertIn(
            "a lista ainda não foi revisada com a versão atual dos dados",
            gate["pendencias"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
