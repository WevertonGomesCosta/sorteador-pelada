"""Motor de otimização dos times.

Esta camada concentra cálculo de odds e otimização, preservando a API pública
exposta por `PeladaLogic`.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pulp
import streamlit as st


def calcular_odds(times: list[list[list]]) -> list[float]:
    odd = []
    for time in times:
        if not time:
            odd.append(1.0)
            continue
        notas = [p[1] for p in time]
        vels = [p[3] for p in time]
        movs = [p[4] for p in time]
        forca = (np.mean(notas) * 1.0) + (np.mean(vels) * 0.8) + (np.mean(movs) * 0.6)
        odd.append(100 / (forca ** 1.5) if forca > 0 else 0)

    media = sum(odd) / len(odd) if odd else 1
    fator = 3.0 / media if media > 0 else 1
    return [o * fator for o in odd]


def otimizar(df: pd.DataFrame, n_times: int, params: dict[str, bool]) -> list[list[list]]:
    dados = []
    for j in df.to_dict("records"):
        dados.append(
            {
                "Nome": j["Nome"],
                "Nota": max(1, min(10, j["Nota"] + random.uniform(-0.7, 0.7))),
                "Posição": j["Posição"],
                "Velocidade": max(1, min(5, j["Velocidade"] + random.uniform(-0.4, 0.4))),
                "Movimentação": max(1, min(5, j["Movimentação"] + random.uniform(-0.4, 0.4))),
            }
        )

    n_jog = len(dados)
    if n_jog < n_times:
        st.error("Jogadores insuficientes.")
        st.stop()

    ids_j, ids_t = range(n_jog), range(n_times)
    t_vals = {
        "Nota": sum(d["Nota"] for d in dados),
        "Vel": sum(d["Velocidade"] for d in dados),
        "Mov": sum(d["Movimentação"] for d in dados),
    }
    medias = {k: v / n_times for k, v in t_vals.items()}

    prob = pulp.LpProblem("Pelada", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", ((i, j) for i in ids_j for j in ids_t), cat="Binary")

    for i in ids_j:
        prob += pulp.lpSum(x[i, j] for j in ids_t) == 1

    min_p = n_jog // n_times
    for j in ids_t:
        prob += pulp.lpSum(x[i, j] for i in ids_j) >= min_p
        prob += pulp.lpSum(x[i, j] for i in ids_j) <= min_p + 1

    if params["pos"]:
        for pos in ["D", "M", "A"]:
            idxs = [i for i, p in enumerate(dados) if p["Posição"] == pos]
            if idxs:
                mp = len(idxs) // n_times
                for j in ids_t:
                    prob += pulp.lpSum(x[i, j] for i in idxs) >= mp

    devs = {k: pulp.LpVariable.dicts(f"d_{k}", ids_t, lowBound=0) for k in ["Nota", "Vel", "Mov"]}
    k_map = {"Nota": "Nota", "Vel": "Velocidade", "Mov": "Movimentação"}

    for j in ids_t:
        for k_abv, k_full in k_map.items():
            soma = pulp.lpSum(x[i, j] * dados[i][k_full] for i in ids_j)
            prob += soma - medias[k_abv] <= devs[k_abv][j]
            prob += medias[k_abv] - soma <= devs[k_abv][j]

    obj = pulp.lpSum(0.1 * devs["Nota"][j] for j in ids_t)
    if params["nota"]:
        obj += pulp.lpSum(10 * devs["Nota"][j] for j in ids_t)
    if params["vel"]:
        obj += pulp.lpSum(4 * devs["Vel"][j] for j in ids_t)
    if params["mov"]:
        obj += pulp.lpSum(3 * devs["Mov"][j] for j in ids_t)

    prob += obj
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

    times = [[] for _ in range(n_times)]
    for i in ids_j:
        for j in ids_t:
            if pulp.value(x[i, j]) == 1:
                times[j].append(
                    [
                        dados[i]["Nome"],
                        dados[i]["Nota"],
                        dados[i]["Posição"],
                        dados[i]["Velocidade"],
                        dados[i]["Movimentação"],
                    ]
                )
                break

    return times
