import re
import unicodedata

import pandas as pd
import streamlit as st

from core.optimizer import calcular_odds as calcular_odds_times, otimizar as otimizar_times
from data.repository import (
    carregar_dados_originais as carregar_dados_originais_repo,
    converter_df_para_excel as converter_df_para_excel_repo,
    criar_base_vazia as criar_base_vazia_repo,
    criar_exemplo as criar_exemplo_repo,
    limpar_df as limpar_df_repo,
    processar_upload as processar_upload_repo,
)


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
        return criar_base_vazia_repo()

    def criar_exemplo(self):
        return criar_exemplo_repo()

    def converter_df_para_excel(self, df):
        return converter_df_para_excel_repo(df)

    def carregar_dados_originais(self):
        return carregar_dados_originais_repo(self.url_padrao, self.limpar_df)

    def processar_upload(self, arquivo_upload):
        return processar_upload_repo(arquivo_upload, self.limpar_df)

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

    def listar_registros_inconsistentes(self, df):
        cols = ["Nome", "Nota", "Posição", "Velocidade", "Movimentação"]
        if df is None or df.empty:
            return pd.DataFrame(columns=cols + ["Problemas"])

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

        problemas_por_linha = []
        for i in range(len(df_diag)):
            problemas = []

            if nomes.iloc[i] == "":
                problemas.append("nome vazio")
            if posicoes.iloc[i] not in ["D", "M", "A", "G"]:
                problemas.append("posição inválida")
            if pd.isna(nota.iloc[i]) or nota.iloc[i] < 1 or nota.iloc[i] > 10:
                problemas.append("nota fora da faixa 1–10")
            if pd.isna(velocidade.iloc[i]) or velocidade.iloc[i] < 1 or velocidade.iloc[i] > 5:
                problemas.append("velocidade fora da faixa 1–5")
            if pd.isna(movimentacao.iloc[i]) or movimentacao.iloc[i] < 1 or movimentacao.iloc[i] > 5:
                problemas.append("movimentação fora da faixa 1–5")

            problemas_por_linha.append("; ".join(problemas))

        mascara = pd.Series([bool(p) for p in problemas_por_linha], index=df_diag.index)
        if not mascara.any():
            return pd.DataFrame(columns=cols + ["Problemas"])

        df_inconsistentes = df_diag.loc[mascara].copy()
        df_inconsistentes["Problemas"] = [p for p, m in zip(problemas_por_linha, mascara) if m]
        return df_inconsistentes.reset_index(drop=True)

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
        return limpar_df_repo(
            df,
            diagnosticar_inconsistencias_base_fn=self.diagnosticar_inconsistencias_base,
            listar_registros_inconsistentes_fn=self.listar_registros_inconsistentes,
            formatar_nome_visual_fn=self.formatar_nome_visual,
        )

    def processar_lista(self, texto, return_metadata=False, emit_warning=True):
        jogadores = []
        ignorados = []

        texto = texto or ""

        def _normalizar_cabecalho_linha(linha: str) -> str:
            linha = unicodedata.normalize("NFKD", str(linha))
            linha = "".join(ch for ch in linha if not unicodedata.combining(ch))
            linha = re.sub(r"[^a-zA-Z0-9]+", " ", linha.lower()).strip()
            return " ".join(linha.split())

        linhas_originais = texto.split("\n")
        linhas = []
        for linha in linhas_originais:
            cabecalho = _normalizar_cabecalho_linha(linha)
            if cabecalho.startswith("goleiros") or cabecalho.startswith("lista de espera"):
                break
            linhas.append(linha)

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
        return calcular_odds_times(times)

    def otimizar(self, df, n_times, params):
        return otimizar_times(df, n_times, params)
