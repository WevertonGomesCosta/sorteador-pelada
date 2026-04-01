import io
import random
import re
import unicodedata

import numpy as np
import pandas as pd
import pulp
import streamlit as st


class PeladaLogic:
    def __init__(self):
        self.url_padrao = "https://docs.google.com/spreadsheets/d/1gCQFG_mYX5DXjh1LRI_UdgrPtkYbkBVLoI3LeOjk5ak/export?format=xlsx"

    def normalizar_chave(self, texto):
        if not isinstance(texto, str):
            return ""
        nfkd = unicodedata.normalize("NFKD", texto.lower())
        return "".join([c for c in nfkd if not unicodedata.combining(c)]).strip()

    def formatar_nome_visual(self, texto):
        if not isinstance(texto, str):
            return ""
        return texto.strip().title()

    def criar_base_vazia(self):
        return pd.DataFrame(columns=["Nome", "Nota", "Posição", "Velocidade", "Movimentação"])

    def criar_exemplo(self):
        dados_exemplo = [
            {"Nome": "Exemplo Atacante", "Nota": 8.5, "Posição": "A", "Velocidade": 5, "Movimentação": 4},
            {"Nome": "Exemplo Meio", "Nota": 6.0, "Posição": "M", "Velocidade": 3, "Movimentação": 3},
            {"Nome": "Exemplo Zagueiro", "Nota": 7.0, "Posição": "D", "Velocidade": 2, "Movimentação": 2},
        ]
        return pd.DataFrame(dados_exemplo)

    def converter_df_para_excel(self, df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Notas pelada")
        return output.getvalue()

    def carregar_dados_originais(self):
        try:
            df = pd.read_excel(self.url_padrao, sheet_name="Notas pelada")
            return self.limpar_df(df)
        except Exception as e:
            st.error(f"Erro ao conectar com Google Sheets: {e}")
            return self.criar_base_vazia()

    def processar_upload(self, arquivo_upload):
        try:
            df = pd.read_excel(arquivo_upload)
            df = self.limpar_df(df)
            return df
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            return None

    def diagnosticar_inconsistencias_base(self, df):
        cols = ["Nome", "Nota", "Posição", "Velocidade", "Movimentação"]
        if df is None or df.empty:
            return {}

        df_diag = df.copy()
        for col in cols:
            if col not in df_diag.columns:
                df_diag[col] = 0 if col not in ["Nome", "Posição"] else ""

        df_diag = df_diag[cols].copy()

        nomes = df_diag["Nome"].fillna("").astype(str).str.strip()
        posicoes = df_diag["Posição"].fillna("").astype(str).str.strip().str.upper()

        nota = pd.to_numeric(df_diag["Nota"], errors="coerce")
        velocidade = pd.to_numeric(df_diag["Velocidade"], errors="coerce")
        movimentacao = pd.to_numeric(df_diag["Movimentação"], errors="coerce")

        inconsistencias = {
            "nomes_vazios": int(nomes.eq("").sum()),
            "posicoes_invalidas": int((~posicoes.isin(["D", "M", "A", "G"])).sum()),
            "notas_invalidas": int((nota.isna() | (nota < 1) | (nota > 10)).sum()),
            "velocidades_invalidas": int((velocidade.isna() | (velocidade < 1) | (velocidade > 5)).sum()),
            "movimentacoes_invalidas": int((movimentacao.isna() | (movimentacao < 1) | (movimentacao > 5)).sum()),
        }

        return inconsistencias

    def emitir_alerta_inconsistencias_base(self, inconsistencias):
        if not inconsistencias:
            return

        mensagens = []
        if inconsistencias.get("nomes_vazios", 0) > 0:
            mensagens.append(f'{inconsistencias["nomes_vazios"]} nome(s) vazio(s)')
        if inconsistencias.get("posicoes_invalidas", 0) > 0:
            mensagens.append(f'{inconsistencias["posicoes_invalidas"]} posição(ões) inválida(s)')
        if inconsistencias.get("notas_invalidas", 0) > 0:
            mensagens.append(f'{inconsistencias["notas_invalidas"]} nota(s) fora da faixa 1–10')
        if inconsistencias.get("velocidades_invalidas", 0) > 0:
            mensagens.append(f'{inconsistencias["velocidades_invalidas"]} velocidade(s) fora da faixa 1–5')
        if inconsistencias.get("movimentacoes_invalidas", 0) > 0:
            mensagens.append(f'{inconsistencias["movimentacoes_invalidas"]} movimentação(ões) fora da faixa 1–5')

        if mensagens:
            st.warning(
                "Integridade da base: foram detectadas inconsistências no carregamento — "
                + "; ".join(mensagens)
                + ". A base foi carregada, mas esses registros merecem revisão."
            )

    def limpar_df(self, df):
        cols = ["Nome", "Nota", "Posição", "Velocidade", "Movimentação"]
        if df is None or df.empty:
            return self.criar_base_vazia()

        inconsistencias = self.diagnosticar_inconsistencias_base(df)
        self.emitir_alerta_inconsistencias_base(inconsistencias)

        for col in cols:
            if col not in df.columns:
                df[col] = 0 if col != "Nome" and col != "Posição" else ""

        df = df[cols]
        df["Posição"] = df["Posição"].fillna("").astype(str)
        df = df[df["Posição"].str.upper() != "G"].reset_index(drop=True)
        df = df.dropna(subset=["Nota"])

        df["Nome"] = df["Nome"].apply(self.formatar_nome_visual)

        # Duplicidade na base não deve bloquear o carregamento.
        # A interface do app fará apenas o diagnóstico visual desses casos.
        return df.reset_index(drop=True)

    def processar_lista(self, texto, return_metadata=False, emit_warning=True):
        jogadores = []
        ignorados = []

        texto = texto or ""
        texto_lower = texto.lower()

        for kw in ["goleiros", "lista de espera"]:
            idx = texto_lower.find(kw)
            if idx != -1:
                texto = texto[:idx]
                break

        linhas = texto.split("\n")
        pattern = r"^\s*\d+[.\-\)]?\s*(.+)"
        tem_numero = any(re.search(pattern, linha) for linha in linhas)

        for linha in linhas:
            linha_original = linha.strip()
            if not linha_original:
                continue

            nome_extraido = ""
            if tem_numero:
                match = re.search(pattern, linha)
                if match:
                    nome_extraido = match.group(1)
            else:
                if len(linha_original) > 2:
                    nome_extraido = linha

            if nome_extraido:
                nome_limpo = nome_extraido.split("(")[0].strip()
                nome_formatado = self.formatar_nome_visual(nome_limpo)

                ignorar = [".", "-", "...", "Lista", "Times"]
                if len(nome_formatado) > 1 and nome_formatado not in ignorar:
                    jogadores.append(nome_formatado)
                else:
                    ignorados.append(linha_original)
            else:
                ignorados.append(linha_original)

        if emit_warning and len(jogadores) != len(set(jogadores)):
            st.warning("⚠️ Atenção: Existem nomes duplicados na lista digitada.")

        if return_metadata:
            return {
                "jogadores": jogadores,
                "ignorados": ignorados,
            }

        return jogadores

    def corrigir_nomes_pela_base(self, lista_nomes, df_base):
        """
        Compara a lista digitada com a base de dados ignorando acentos e case.
        Se digitar 'joao' e na base tiver 'João', corrige para 'João'.
        """
        if df_base.empty:
            return lista_nomes

        mapa_nomes = {self.normalizar_chave(nome): nome for nome in df_base["Nome"]}

        lista_corrigida = []
        for nome_input in lista_nomes:
            chave_input = self.normalizar_chave(nome_input)
            if chave_input in mapa_nomes:
                lista_corrigida.append(mapa_nomes[chave_input])
            else:
                lista_corrigida.append(nome_input)

        return lista_corrigida

    def diagnosticar_lista_para_sorteio(self, texto, df_base, novos_jogadores):
        processamento = self.processar_lista(
            texto,
            return_metadata=True,
            emit_warning=False,
        )

        nomes_brutos = list(processamento["jogadores"])
        ignorados = list(processamento["ignorados"])

        nomes_corrigidos = self.corrigir_nomes_pela_base(nomes_brutos, df_base)

        correcoes_aplicadas = []
        for original, corrigido in zip(nomes_brutos, nomes_corrigidos):
            if original != corrigido:
                correcoes_aplicadas.append(
                    {"original": original, "corrigido": corrigido}
                )

        duplicados = []
        nomes_unicos = []
        vistos = set()
        for nome in nomes_corrigidos:
            if nome in vistos:
                if nome not in duplicados:
                    duplicados.append(nome)
            else:
                vistos.add(nome)
                nomes_unicos.append(nome)

        nomes_conhecidos = set()
        if df_base is not None and not df_base.empty:
            nomes_conhecidos.update(
                [nome for nome in df_base["Nome"].dropna().tolist() if isinstance(nome, str)]
            )

        if novos_jogadores:
            for jogador in novos_jogadores:
                if isinstance(jogador, dict):
                    nome = jogador.get("Nome")
                    if isinstance(nome, str) and nome.strip():
                        nomes_conhecidos.add(nome)

        nao_encontrados = []
        for nome in nomes_unicos:
            if nome not in nomes_conhecidos:
                nao_encontrados.append(nome)

        lista_final_sugerida = [
            nome for nome in nomes_unicos if nome not in nao_encontrados
        ]

        return {
            "nomes_brutos": nomes_brutos,
            "ignorados": ignorados,
            "nomes_corrigidos": nomes_corrigidos,
            "correcoes_aplicadas": correcoes_aplicadas,
            "duplicados": duplicados,
            "nao_encontrados": nao_encontrados,
            "lista_final_sugerida": lista_final_sugerida,
            "total_brutos": len(nomes_brutos),
            "total_validos": len(lista_final_sugerida),
            "tem_nao_encontrados": len(nao_encontrados) > 0,
            "tem_duplicados": len(duplicados) > 0,
            "tem_correcoes": len(correcoes_aplicadas) > 0,
            "tem_problemas": any(
                [
                    len(ignorados) > 0,
                    len(correcoes_aplicadas) > 0,
                    len(duplicados) > 0,
                    len(nao_encontrados) > 0,
                ]
            ),
        }

    def calcular_odds(self, times):
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

    def otimizar(self, df, n_times, params):
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
