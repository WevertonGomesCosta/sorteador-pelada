from tests._smoke_shared import *


class ResultViewSmokeTestCase(unittest.TestCase):
    def test_compartilhamento_coloca_goleiro_primeiro_com_marcador(self) -> None:
        from ui.result_view import construir_texto_compartilhamento_resultado

        times = [
            [
                ["Ana", 8, "M", 4, 3],
                ["Goleiro Um", 7, "G", 3, 2],
                ["Bruno", 6, "D", 2, 3],
            ]
        ]

        texto = construir_texto_compartilhamento_resultado(times=times)

        self.assertEqual(
            texto,
            "*Time 1:*\n(G) Goleiro Um\nAna\nBruno\n",
        )

    def test_compartilhamento_preserva_capitao_e_marca_goleiro(self) -> None:
        from ui.result_view import construir_texto_compartilhamento_resultado

        times = [
            [
                ["Ana", 8, "M", 4, 3, True],
                ["Goleiro Um", 7, "G", 3, 2, False],
            ]
        ]

        texto = construir_texto_compartilhamento_resultado(times=times)

        self.assertEqual(
            texto,
            "*Time 1:*\n(G) Goleiro Um\nAna (C)\n",
        )
