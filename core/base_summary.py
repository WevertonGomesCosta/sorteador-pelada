"""Resumos neutros de inconsistências da base.

Este módulo existe para evitar acoplamento entre a camada core e a camada de UI.
"""

from __future__ import annotations


def total_inconsistencias_base(inconsistencias: dict) -> int:
    return sum(int(v) for v in inconsistencias.values() if isinstance(v, int) and v > 0)



def resumo_inconsistencias_base(inconsistencias: dict) -> str:
    mensagens = []

    if inconsistencias.get("nomes_vazios", 0) > 0:
        mensagens.append(f'{inconsistencias["nomes_vazios"]} nome(s) vazio(s)')
    if inconsistencias.get("nomes_duplicados", 0) > 0:
        mensagens.append(f'{inconsistencias["nomes_duplicados"]} nome(s) duplicado(s)')
    if inconsistencias.get("posicoes_invalidas", 0) > 0:
        mensagens.append(f'{inconsistencias["posicoes_invalidas"]} posição(ões) inválida(s)')
    if inconsistencias.get("notas_invalidas", 0) > 0:
        mensagens.append(f'{inconsistencias["notas_invalidas"]} nota(s) fora da faixa 1–10')
    if inconsistencias.get("velocidades_invalidas", 0) > 0:
        mensagens.append(f'{inconsistencias["velocidades_invalidas"]} velocidade(s) fora da faixa 1–5')
    if inconsistencias.get("movimentacoes_invalidas", 0) > 0:
        mensagens.append(f'{inconsistencias["movimentacoes_invalidas"]} movimentação(ões) fora da faixa 1–5')

    return "; ".join(mensagens)
