"""Regressao da leitura de nomes copiados do WhatsApp com Unicode invisivel."""

from tests._smoke_shared import *

from core.validators import (
    normalizar_nome_comparacao,
    normalizar_nome_duplicado_lista,
    remover_caracteres_invisiveis,
)
from ui.review_helpers import (
    _atualizar_texto_lista_revisao,
    _ocorrencias_do_nome_duplicado_na_lista,
)


# Sequencia literal de caracteres do nome enviado no WhatsApp (04/11/2026).
HIURY_WHATSAPP = "\u200b\u2060\u2068\u2060H\u2060i\u2060u\u2060r\u2060y\u2060\u2069"

LISTA_WHATSAPP = """\
📢 O PAGAMENTO DEVE SER FEITO ANTES DA PELADA.
Pix: joesiosouza@gmail.com

Pelada QUARTA 04/11:
1- Hugo
2- Douglas
3- Garrafa
4-Clauzio
5- Weverton
6- Junim Rocha
7- Vitor
8- Gustavo
9- Cleiton
10- Nico
11- {hiury}
12- Otávio
13- Werikson
14- Wesley
15- Bruno

Goleiros:
1- Baju
2-
3-

Lista de espera:
1- Pelota
2- Túlio da
""".format(hiury=HIURY_WHATSAPP)

JOGADORES_ESPERADOS = [
    "Hugo", "Douglas", "Garrafa", "Clauzio", "Weverton",
    "Junim Rocha", "Vitor", "Gustavo", "Cleiton", "Nico",
    "Hiury", "Otávio", "Werikson", "Wesley", "Bruno",
]


class UnicodeNamesTestCase(unittest.TestCase):
    @staticmethod
    def _logic():
        import importlib
        import types

        sys.modules.setdefault("pulp", types.SimpleNamespace())
        return importlib.import_module("core.logic").PeladaLogic()

    def test_limpeza_unicode_exacta_e_idempotente(self) -> None:
        self.assertEqual(remover_caracteres_invisiveis(HIURY_WHATSAPP), "Hiury")
        self.assertEqual(remover_caracteres_invisiveis("Otávio"), "Otávio")
        self.assertEqual(remover_caracteres_invisiveis("Túlio da"), "Túlio da")
        texto = "Junim Rocha (Convidado)"
        self.assertEqual(remover_caracteres_invisiveis(texto), texto)
        self.assertEqual(
            remover_caracteres_invisiveis(remover_caracteres_invisiveis(HIURY_WHATSAPP)),
            remover_caracteres_invisiveis(HIURY_WHATSAPP),
        )

    def test_comparacao_nome_duplicado_e_chave_base(self) -> None:
        logic = self._logic()
        self.assertEqual(logic.normalizar_chave(HIURY_WHATSAPP), "hiury")
        self.assertEqual(normalizar_nome_comparacao(HIURY_WHATSAPP), "HIURY")
        self.assertEqual(normalizar_nome_duplicado_lista(HIURY_WHATSAPP), "HIURY")
        self.assertEqual(logic.formatar_nome_visual(HIURY_WHATSAPP), "Hiury")
        self.assertNotEqual(
            normalizar_nome_duplicado_lista("Douglas"),
            normalizar_nome_duplicado_lista("Douglas (Pimpim)"),
        )

    def test_processar_lista_completa_sem_incluir_espera_ou_goleiro(self) -> None:
        resultado = self._logic().processar_lista(
            LISTA_WHATSAPP,
            return_metadata=True,
            emit_warning=False,
            incluir_goleiros=False,
        )
        self.assertEqual(resultado["jogadores"], JOGADORES_ESPERADOS)
        self.assertEqual(resultado["goleiros_lidos"], ["Baju"])
        self.assertFalse(resultado["goleiros_incluidos"])

    def test_correcao_automaticamente_encontra_hiury_na_base(self) -> None:
        df_base = pd.DataFrame({"Nome": JOGADORES_ESPERADOS})
        diagnostico = self._logic().diagnosticar_lista_para_sorteio(
            LISTA_WHATSAPP, df_base, []
        )
        self.assertEqual(diagnostico["total_brutos"], 15)
        self.assertEqual(diagnostico["total_validos"], 15)
        self.assertEqual(diagnostico["nomes_corrigidos"], JOGADORES_ESPERADOS)
        self.assertEqual(diagnostico["nao_encontrados"], [])
        self.assertEqual(diagnostico["duplicados"], [])

    def test_lista_aleatoria_detecta_hiury_repetido_invisivel(self) -> None:
        lista = f"1- Hiury\n2- {HIURY_WHATSAPP}\n3- Nico\nGoleiros:\n1- Baju\nLista de espera:\n1- Pelota"
        nomes, duplicados = flow_guard.extrair_nomes_unicos_da_lista(self._logic(), lista)
        self.assertEqual(nomes, ["Hiury", "Nico"])
        self.assertEqual(duplicados, 1)

    def test_revisao_encontra_ocorrencia_com_marcadores_invisiveis(self) -> None:
        lista = f"1- Hiury\n2- {HIURY_WHATSAPP}\nGoleiros:\n1- Baju\nLista de espera:\n1- Hiury"
        ocorrencias = _ocorrencias_do_nome_duplicado_na_lista(
            lista, {}, "Hiury", revisao_aleatoria=True
        )
        self.assertEqual(len(ocorrencias), 2)
        self.assertEqual([item["numero"] for item in ocorrencias], ["1", "2"])
        texto_corrigido, alterou = _atualizar_texto_lista_revisao(
            lista, "Hiury", novo_nome="Hiury Gomes"
        )
        self.assertTrue(alterou)
        self.assertEqual(texto_corrigido.count("Hiury Gomes"), 2)
        self.assertIn("Lista de espera:\n1- Hiury", texto_corrigido)

    def test_marcadores_em_numeracao_e_cabecalhos(self) -> None:
        lista = (
            "\u20601- \u200bHiury\n"
            "2- Otávio\n"
            "Goleiros\u2060:\n"
            "1- Baju\n"
            "Lista de espera\u200b:\n"
            "1- Pelota"
        )
        resultado = self._logic().processar_lista(
            lista, return_metadata=True, emit_warning=False,
            incluir_goleiros=True, qtd_goleiros_esperada=1,
        )
        self.assertEqual(resultado["jogadores"], ["Hiury", "Otávio", "Baju"])
        self.assertEqual(resultado["goleiros_lidos"], ["Baju"])
