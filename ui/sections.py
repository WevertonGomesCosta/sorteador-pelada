"""Seções visuais simples do Sorteador Pelada PRO."""

from __future__ import annotations

import html
import re
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from state.session import registrar_base_carregada_no_estado
from ui.actions import render_action_button
from ui.panels import render_session_status_panel, render_step_cta_panel
from ui.primitives import _titulo_expander, render_inline_status_note, render_section_header
from ui.summary_strings import resumo_expander_configuracao

from core.validators import (
    listar_bloqueios_base_atual,
    normalizar_nome_comparacao,
    registro_valido_para_sorteio,
    valor_slider_corrigir,
)


def _normalizar_cabecalho_lista(linha: str) -> str:
    linha = normalizar_nome_comparacao(str(linha or "")).upper()
    return " ".join(linha.split())


def _eh_inicio_secao_excluida(linha: str) -> bool:
    cabecalho = _normalizar_cabecalho_lista(linha)
    return cabecalho.startswith("GOLEIROS") or cabecalho.startswith("LISTA DE ESPERA")


def _linhas_principais_da_lista(texto_lista: str) -> tuple[list[str], int]:
    linhas = str(texto_lista or "").splitlines()
    indice_corte = len(linhas)

    for idx, linha in enumerate(linhas):
        if _eh_inicio_secao_excluida(linha):
            indice_corte = idx
            break

    return linhas, indice_corte


def _atualizar_texto_lista_revisao(
    texto_lista: str,
    nome_alvo: str,
    *,
    novo_nome: str | None = None,
    remover: bool = False,
    manter_primeira_ocorrencia: bool = False,
) -> tuple[str, bool]:
    linhas, indice_corte = _linhas_principais_da_lista(texto_lista)
    alvo_normalizado = normalizar_nome_comparacao(nome_alvo)
    novas_linhas = []
    encontrou = False

    for linha_idx, linha_original in enumerate(linhas):
        linha = str(linha_original).strip()

        if linha_idx >= indice_corte or not linha:
            novas_linhas.append(linha_original)
            continue

        linha_comparavel = _extrair_nome_comparavel_da_linha(linha)
        linha_normalizada = normalizar_nome_comparacao(linha_comparavel)
        mesma_pessoa = bool(alvo_normalizado) and linha_normalizada == alvo_normalizado

        if not mesma_pessoa:
            novas_linhas.append(linha_original)
            continue

        if remover:
            if manter_primeira_ocorrencia and not encontrou:
                novas_linhas.append(linha_original)
            encontrou = True
            continue

        nome_destino = str(novo_nome or "").strip()
        if nome_destino:
            novas_linhas.append(nome_destino)
            encontrou = True
        else:
            novas_linhas.append(linha_original)

    return "\n".join(novas_linhas), encontrou


def _extrair_nome_comparavel_da_linha(linha: str) -> str:
    linha_original = str(linha or "").strip()
    if not linha_original:
        return ""

    match = re.search(r"^\s*\d+[.\-)]?\s*(.+)", linha_original)
    nome_extraido = match.group(1) if match else linha_original
    nome_limpo = nome_extraido.split("(")[0].strip()
    return nome_limpo


def _origens_do_nome_duplicado(diagnostico: dict, nome_duplicado: str) -> list[str]:
    alvo_normalizado = normalizar_nome_comparacao(nome_duplicado)
    origens = []

    for original, corrigido in zip(
        diagnostico.get("nomes_brutos", []),
        diagnostico.get("nomes_corrigidos", []),
    ):
        if normalizar_nome_comparacao(corrigido) != alvo_normalizado:
            continue
        nome_original = str(original).strip()
        if nome_original and nome_original not in origens:
            origens.append(nome_original)

    nome_visual = str(nome_duplicado).strip()
    if nome_visual and nome_visual not in origens:
        origens.append(nome_visual)

    return origens



def _ocorrencias_numeradas_da_lista_principal(texto_lista: str) -> list[dict]:
    linhas, indice_corte = _linhas_principais_da_lista(texto_lista)
    pattern = r"^\s*(\d+)\s*[.\-\)]?\s*(.+)"
    ignorar = {".", "-", "...", "Lista", "Times"}
    ocorrencias = []

    for linha_idx, linha_original in enumerate(linhas[:indice_corte]):
        linha = str(linha_original).strip()
        if not linha:
            continue

        match = re.search(pattern, linha)
        if not match:
            continue

        numero, conteudo = match.groups()
        nome_limpo = str(conteudo).split("(")[0].strip()
        nome_formatado = " ".join(nome_limpo.split())
        if len(nome_formatado) <= 1 or nome_formatado in ignorar:
            continue

        ocorrencias.append(
            {
                "linha_idx": linha_idx,
                "valor": linha,
                "comparavel": _extrair_nome_comparavel_da_linha(linha),
                "numero": numero,
            }
        )

    return ocorrencias


def _ocorrencias_do_nome_duplicado_na_lista(
    texto_lista: str,
    diagnostico: dict,
    nome_duplicado: str,
    *,
    revisao_aleatoria: bool,
) -> list[dict]:
    linhas, indice_corte = _linhas_principais_da_lista(texto_lista)

    if revisao_aleatoria:
        alvos_normalizados = {normalizar_nome_comparacao(nome_duplicado)}
    else:
        alvos_normalizados = {
            normalizar_nome_comparacao(nome)
            for nome in (_origens_do_nome_duplicado(diagnostico, nome_duplicado) + [nome_duplicado])
            if str(nome).strip()
        }

    ocorrencias = []
    for ocorrencia in _ocorrencias_numeradas_da_lista_principal(texto_lista):
        linha_normalizada = normalizar_nome_comparacao(ocorrencia["comparavel"])
        if linha_normalizada and linha_normalizada in alvos_normalizados:
            ocorrencias.append(ocorrencia)

    return ocorrencias



def _aplicar_edicoes_em_ocorrencias_da_lista(
    texto_lista: str,
    edicoes_por_linha: dict[int, str],
    *,
    remover_linhas: set[int] | None = None,
) -> tuple[str, bool]:
    linhas, indice_corte = _linhas_principais_da_lista(texto_lista)
    remover_linhas = remover_linhas or set()

    novas_linhas = []
    alterou = False

    for linha_idx, linha_original in enumerate(linhas):
        linha = str(linha_original).strip()

        if linha_idx >= indice_corte or not linha:
            novas_linhas.append(linha_original)
            continue

        if linha_idx in remover_linhas:
            alterou = True
            continue

        if linha_idx in edicoes_por_linha:
            novo_valor = str(edicoes_por_linha[linha_idx]).strip()
            if novo_valor != linha:
                alterou = True
            if novo_valor:
                novas_linhas.append(novo_valor)
            else:
                alterou = True
            continue

        novas_linhas.append(linha_original)

    return "\n".join(novas_linhas), alterou


def render_revisao_pendencias_panel(
    logic,
    lista_texto: str,
    diagnostico: dict,
    *,
    revisao_aleatoria: bool,
    lista_input_key: str,
    atualizar_integridade_base_no_estado,
    diagnosticar_lista_no_estado,
    render_action_button,
):
    qtd_nao_encontrados = len(diagnostico.get("nao_encontrados", []))
    qtd_duplicados = len(diagnostico.get("duplicados", []))
    qtd_bloqueios_base = len(diagnostico.get("nomes_bloqueados_base", []))
    total_pendencias = qtd_nao_encontrados + qtd_bloqueios_base + qtd_duplicados

    if total_pendencias == 0:
        return

    metricas = [
        ("Pendências", str(total_pendencias)),
        ("Faltantes", str(qtd_nao_encontrados)),
        ("Bloqueios da base", str(qtd_bloqueios_base)),
    ]
    metricas.append(("Duplicados na lista", str(qtd_duplicados)))

    metricas_html = "".join(
        f'<div class="review-pending-panel__metric">'
        f'<div class="review-pending-panel__metric-label">{html.escape(rotulo)}</div>'
        f'<div class="review-pending-panel__metric-value">{html.escape(valor)}</div>'
        f'</div>'
        for rotulo, valor in metricas
    )

    resumo_partes = []
    if qtd_nao_encontrados > 0:
        resumo_partes.append(f"{qtd_nao_encontrados} nome(s) não encontrado(s)")
    if qtd_bloqueios_base > 0:
        resumo_partes.append(f"{qtd_bloqueios_base} bloqueio(s) na base")
    if qtd_duplicados > 0:
        resumo_partes.append(f"{qtd_duplicados} duplicado(s) na lista")

    resumo_texto = ", ".join(resumo_partes)
    expandir_primeiro_faltante = qtd_nao_encontrados > 0
    expandir_primeiro_duplicado = (not expandir_primeiro_faltante) and qtd_duplicados > 0
    expandir_primeiro_bloqueio = (not expandir_primeiro_faltante) and (qtd_duplicados == 0) and qtd_bloqueios_base > 0

    st.markdown('<div id="revisao-pendencias-anchor"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="review-pending-panel">
            <div class="review-pending-panel__eyebrow">Pendências para corrigir</div>
            <div class="review-pending-panel__title">Corrija os itens abaixo no próprio painel da revisão</div>
            <div class="review-pending-panel__desc">{html.escape(resumo_texto)}.</div>
            <div class="review-pending-panel__metrics">{metricas_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if qtd_nao_encontrados > 0 and not revisao_aleatoria:
        st.markdown("**Nomes não encontrados na base**")
        for idx, nome in enumerate(diagnostico.get("nao_encontrados", [])):
            with st.expander(
                f"❓ Corrigir nome da lista: {nome}",
                expanded=(qtd_nao_encontrados == 1 or (expandir_primeiro_faltante and idx == 0)),
            ):
                st.caption("Ajuste o nome, cadastre este atleta na base atual ou remova o item da lista sem sair desta etapa.")
                with st.form(f"form_pendencia_nao_encontrado_{idx}"):
                    nome_corrigido = st.text_input(
                        "Nome correto na lista",
                        value=nome,
                        key=f"pendencia_nome_corrigido_{idx}",
                    )
                    col_nf1, col_nf2, col_nf3 = st.columns(3)
                    aplicar_nome = col_nf1.form_submit_button("✅ Corrigir na lista")
                    cadastrar_nome = col_nf2.form_submit_button("➕ Cadastrar na base")
                    remover_nome = col_nf3.form_submit_button("➖ Remover da lista")

                    if aplicar_nome:
                        nome_destino = str(nome_corrigido).strip()
                        if hasattr(logic, "formatar_nome_visual") and nome_destino:
                            nome_destino = logic.formatar_nome_visual(nome_destino)
                        novo_texto_lista, alterou = _atualizar_texto_lista_revisao(
                            lista_texto,
                            nome,
                            novo_nome=nome_destino,
                        )
                        if alterou and novo_texto_lista.strip():
                            st.session_state[f"{lista_input_key}__pending"] = novo_texto_lista
                            st.session_state[f"{lista_input_key}__revisar"] = True
                            st.rerun()
                        st.warning("Não foi possível localizar esse nome na lista atual para aplicar a correção.")

                    if cadastrar_nome:
                        faltantes_priorizados = diagnostico.get("nao_encontrados", []).copy()
                        if idx < len(faltantes_priorizados):
                            nome_escolhido = faltantes_priorizados.pop(idx)
                            faltantes_priorizados.insert(0, nome_escolhido)
                        st.session_state.faltantes_revisao = faltantes_priorizados
                        st.session_state.cadastro_guiado_ativo = True
                        st.session_state.faltantes_cadastrados_na_rodada = []
                        st.session_state.revisao_pendente_pos_cadastro = False
                        st.session_state.lista_revisada_confirmada = False
                        st.session_state.lista_revisada = None
                        st.session_state.revisao_lista_expandida = True
                        st.rerun()

                    if remover_nome:
                        novo_texto_lista, alterou = _atualizar_texto_lista_revisao(
                            lista_texto,
                            nome,
                            remover=True,
                        )
                        if alterou:
                            st.session_state[f"{lista_input_key}__pending"] = novo_texto_lista
                            st.session_state[f"{lista_input_key}__revisar"] = True
                            st.rerun()
                        st.warning("Não foi possível localizar esse nome na lista atual para removê-lo.")

    if qtd_duplicados > 0:
        st.markdown("**Nomes repetidos detectados na lista**")
        for idx, nome in enumerate(diagnostico.get("duplicados", [])):
            ocorrencias_duplicado = _ocorrencias_do_nome_duplicado_na_lista(
                lista_texto,
                diagnostico,
                nome,
                revisao_aleatoria=revisao_aleatoria,
            )
            with st.expander(
                f"🔁 Corrigir nomes repetidos: {nome}",
                expanded=(qtd_duplicados == 1 or (expandir_primeiro_duplicado and idx == 0)),
            ):
                if revisao_aleatoria:
                    st.caption("Nomes repetidos não serão unificados automaticamente. Corrija os nomes abaixo antes de confirmar a lista.")
                else:
                    st.caption("Edite as ocorrências abaixo para corrigir o nome repetido. Se necessário, remova ocorrências indevidas e reaplique a revisão.")

                if ocorrencias_duplicado:
                    st.markdown(
                        "**Ocorrências detectadas na lista:** "
                        + ", ".join(f"`{html.escape(item['valor'])}`" for item in ocorrencias_duplicado)
                    )
                else:
                    st.warning("Não foi possível localizar as ocorrências atuais desse nome na lista.")
                    continue

                with st.form(f"form_pendencia_duplicado_{idx}"):
                    edicoes_por_linha = {}
                    remover_linhas = set()

                    for ocorrencia_idx, ocorrencia in enumerate(ocorrencias_duplicado, start=1):
                        linha_idx = ocorrencia["linha_idx"]
                        valor_atual = ocorrencia["valor"]
                        campo_key = f"pendencia_duplicado_nome_{idx}_{linha_idx}"
                        nome_editado = st.text_input(
                            f"Nome correto para a ocorrência {ocorrencia_idx}",
                            value=valor_atual,
                            key=campo_key,
                        )
                        edicoes_por_linha[linha_idx] = nome_editado
                        if not revisao_aleatoria:
                            remover = st.checkbox(
                                f"Remover esta ocorrência ({ocorrencia_idx})",
                                value=False,
                                key=f"pendencia_duplicado_remover_{idx}_{linha_idx}",
                            )
                            if remover:
                                remover_linhas.add(linha_idx)

                    aplicar_label = "✅ Corrigir nomes na lista" if revisao_aleatoria else "✅ Aplicar alterações na lista"
                    aplicar_correcao = st.form_submit_button(aplicar_label)

                    if aplicar_correcao:
                        if revisao_aleatoria and any(not str(valor).strip() for valor in edicoes_por_linha.values()):
                            st.warning("No sorteio aleatório, corrija os nomes preenchendo cada ocorrência. Remoções não são aplicadas neste caso.")
                        elif (not revisao_aleatoria) and len(remover_linhas) == len(ocorrencias_duplicado):
                            st.warning("Mantenha pelo menos uma ocorrência ou corrija os nomes antes de aplicar.")
                        elif (not revisao_aleatoria) and any(
                            (linha_idx not in remover_linhas) and (not str(valor).strip())
                            for linha_idx, valor in edicoes_por_linha.items()
                        ):
                            st.warning("Para remover uma ocorrência, use a opção de remoção. Para manter, informe um nome válido.")
                        else:
                            novo_texto_lista, alterou = _aplicar_edicoes_em_ocorrencias_da_lista(
                                lista_texto,
                                edicoes_por_linha,
                                remover_linhas=remover_linhas,
                            )
                            if alterou and novo_texto_lista.strip():
                                st.session_state[f"{lista_input_key}__pending"] = novo_texto_lista
                                st.session_state[f"{lista_input_key}__revisar"] = True
                                st.rerun()
                            st.warning("Faça pelo menos uma correção de nome para reaplicar a revisão.")

    if qtd_bloqueios_base > 0:
        st.markdown("**Bloqueios da base que impedem a confirmação**")
        for idx, item in enumerate(diagnostico.get("nomes_bloqueados_base", [])):
            nome = item.get("nome", "")
            motivos = item.get("motivos", [])
            with st.expander(
                f"🛠️ Corrigir base para liberar: {nome}",
                expanded=(qtd_bloqueios_base == 1 or (expandir_primeiro_bloqueio and idx == 0)),
            ):
                st.markdown(f"**Motivos detectados:** {'; '.join(motivos)}")
                st.caption("Edite ou remova os registros deste nome aqui mesmo. Ao salvar, a revisão será atualizada automaticamente.")
                render_correcao_inline_bloqueios_base(
                    logic,
                    lista_texto,
                    [item],
                    atualizar_integridade_base_no_estado=atualizar_integridade_base_no_estado,
                    diagnosticar_lista_no_estado=diagnosticar_lista_no_estado,
                    render_action_button=render_action_button,
                    inline_flat=True,
                    show_intro=False,
                    manage_focus=False,
                )


def formatar_df_visual_numeros_inteiros(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df_fmt = df.copy()
    for col in ["Nota", "Velocidade", "Movimentação"]:
        if col in df_fmt.columns:
            def _to_int_visual(v):
                try:
                    if pd.isna(v):
                        return v
                except Exception:
                    pass
                try:
                    return int(round(float(v)))
                except Exception:
                    return v
            df_fmt[col] = df_fmt[col].apply(_to_int_visual)
    return df_fmt


def estilo_celulas_inconsistentes(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(index=getattr(df, "index", []), columns=getattr(df, "columns", []))

    estilos = pd.DataFrame("", index=df.index, columns=df.columns)
    destaque = "font-weight: 700;"

    if "Nome" in df.columns:
        nomes = df["Nome"].fillna("").astype(str).str.strip()
        estilos.loc[nomes.eq(""), "Nome"] = destaque

    if "Posição" in df.columns:
        posicoes = df["Posição"].fillna("").astype(str).str.strip().str.upper()
        estilos.loc[~posicoes.isin(["D", "M", "A", "G"]), "Posição"] = destaque

    if "Nota" in df.columns:
        nota = pd.to_numeric(df["Nota"], errors="coerce")
        estilos.loc[nota.isna() | (nota < 1) | (nota > 10), "Nota"] = destaque

    if "Velocidade" in df.columns:
        velocidade = pd.to_numeric(df["Velocidade"], errors="coerce")
        estilos.loc[velocidade.isna() | (velocidade < 1) | (velocidade > 5), "Velocidade"] = destaque

    if "Movimentação" in df.columns:
        movimentacao = pd.to_numeric(df["Movimentação"], errors="coerce")
        estilos.loc[movimentacao.isna() | (movimentacao < 1) | (movimentacao > 5), "Movimentação"] = destaque

    return estilos


def render_base_summary():
    df_base = st.session_state.df_base
    qtd_jogadores = len(df_base)

    if st.session_state.is_admin:
        origem = "Grupo"
    elif qtd_jogadores == 0:
        origem = "Vazia"
    else:
        origem = "Sua base"

    modo = "Base do grupo" if st.session_state.is_admin else "Público"

    if df_base.empty:
        posicoes = "—"
    else:
        cont_pos = df_base["Posição"].value_counts()
        posicoes = " / ".join(
            [
                f"D {cont_pos.get('D', 0)}",
                f"M {cont_pos.get('M', 0)}",
                f"A {cont_pos.get('A', 0)}",
            ]
        )

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-label">⚽ Modo</div>
                <div class="summary-value">{modo}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">👥 Jogadores</div>
                <div class="summary-value">{qtd_jogadores} jogadores</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">📋 Base</div>
                <div class="summary-value">{origem}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">🧩 D / M / A</div>
                <div class="summary-value">{posicoes}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_base_inconsistencias_expander(
    logic=None,
    *,
    atualizar_integridade_base_no_estado=None,
    diagnosticar_lista_no_estado=None,
    render_action_button=None,
):
    registros = st.session_state.get("base_registros_inconsistentes_carregamento", [])
    if not registros:
        return

    df_inconsistentes = pd.DataFrame(registros)
    if df_inconsistentes.empty:
        return

    with st.expander("⚠️ Registros com inconsistências", expanded=False):
        st.caption("Revise apenas os registros inconsistentes. Você pode remover a linha ou, se preferir, corrigi-la no próprio bloco.")
        df_inconsistentes_display = df_inconsistentes.copy()
        styler = df_inconsistentes_display.style.apply(estilo_celulas_inconsistentes, axis=None)
        st.dataframe(
            styler,
            width="stretch",
            hide_index=True,
        )

        if logic is None or atualizar_integridade_base_no_estado is None or render_action_button is None:
            return

        st.markdown("---")
        st.caption("A exclusão é a ação mais rápida quando a linha está errada. Se quiser manter o registro, use a opção de correção.")

        for _, row in df_inconsistentes.iterrows():
            nome = str(row.get("Nome", "")).strip()
            posicao = str(row.get("Posição", "")).strip()
            nota = row.get("Nota", "")
            velocidade = row.get("Velocidade", "")
            movimentacao = row.get("Movimentação", "")

            candidatos = st.session_state.df_base.copy()
            if nome:
                candidatos = candidatos[candidatos["Nome"].astype(str).str.strip() == nome]
            else:
                candidatos = candidatos[candidatos["Nome"].fillna("").astype(str).str.strip() == ""]

            candidatos = candidatos.reset_index().rename(columns={"index": "_orig_index"})

            idx_original = None
            for _, cand in candidatos.iterrows():
                if (
                    str(cand.get("Posição", "")) == posicao
                    and str(cand.get("Nota", "")) == str(nota)
                    and str(cand.get("Velocidade", "")) == str(velocidade)
                    and str(cand.get("Movimentação", "")) == str(movimentacao)
                ):
                    idx_original = int(cand["_orig_index"])
                    break
            if idx_original is None and not candidatos.empty:
                idx_original = int(candidatos.iloc[0]["_orig_index"])
            if idx_original is None:
                continue

            with st.expander(f"🧾 Registro inconsistente: {nome or '(sem nome)'}", expanded=False):
                col_info1, col_info2, col_info3, col_info4, col_info5 = st.columns(5)
                col_info1.markdown(f"**Nome**\n\n{nome}")
                col_info2.markdown(f"**Posição**\n\n{posicao}")
                col_info3.markdown(f"**Nota**\n\n{nota}")
                col_info4.markdown(f"**Velocidade**\n\n{velocidade}")
                col_info5.markdown(f"**Movimentação**\n\n{movimentacao}")

                if render_action_button(
                    "🗑️ Excluir esta linha",
                    key=f"base_inconsistente_excluir_{idx_original}",
                    role="secondary",
                ):
                    st.session_state.df_base = (
                        st.session_state.df_base.drop(index=idx_original).reset_index(drop=True)
                    )
                    st.session_state.revisao_foco_bloqueio_nome = nome
                    atualizar_integridade_base_no_estado(logic)
                    if diagnosticar_lista_no_estado is not None and st.session_state.get("lista_texto_revisado", ""):
                        diagnosticar_lista_no_estado(logic, st.session_state.get("lista_texto_revisado", ""))
                        st.session_state.revisao_lista_expandida = True
                    st.rerun()

                with st.expander("✏️ Corrigir este registro", expanded=False):
                    with st.form(f"base_inconsistente_corrigir_{idx_original}"):
                        nome_corr = st.text_input(
                            "Nome",
                            value=nome,
                            key=f"base_inconsistente_nome_{idx_original}",
                        )
                        posicao_atual = posicao.upper() if posicao.upper() in ["D", "M", "A"] else "M"
                        pos_corr = st.selectbox(
                            "Posição",
                            ["D", "M", "A"],
                            index=["D", "M", "A"].index(posicao_atual),
                            key=f"base_inconsistente_pos_{idx_original}",
                        )
                        nota_corr = st.slider(
                            "Nota", 1, 10, valor_slider_corrigir(nota, 1, 10, 6),
                            key=f"base_inconsistente_nota_{idx_original}",
                        )
                        vel_corr = st.slider(
                            "Velocidade", 1, 5, valor_slider_corrigir(velocidade, 1, 5, 3),
                            key=f"base_inconsistente_vel_{idx_original}",
                        )
                        mov_corr = st.slider(
                            "Movimentação", 1, 5, valor_slider_corrigir(movimentacao, 1, 5, 3),
                            key=f"base_inconsistente_mov_{idx_original}",
                        )
                        salvar = st.form_submit_button("💾 Salvar correção")

                        if salvar:
                            nome_corrigido = str(nome_corr).strip()
                            if hasattr(logic, "formatar_nome_visual") and nome_corrigido:
                                nome_corrigido = logic.formatar_nome_visual(nome_corrigido)
                            st.session_state.df_base.loc[idx_original, "Nome"] = nome_corrigido
                            st.session_state.df_base.loc[idx_original, "Posição"] = pos_corr
                            st.session_state.df_base.loc[idx_original, "Nota"] = nota_corr
                            st.session_state.df_base.loc[idx_original, "Velocidade"] = vel_corr
                            st.session_state.df_base.loc[idx_original, "Movimentação"] = mov_corr
                            atualizar_integridade_base_no_estado(logic)
                            if diagnosticar_lista_no_estado is not None and st.session_state.get("lista_texto_revisado", ""):
                                diagnosticar_lista_no_estado(logic, st.session_state.get("lista_texto_revisado", ""))
                                st.session_state.revisao_lista_expandida = True
                            st.rerun()


def total_inconsistencias_base(inconsistencias: dict) -> int:
    if not inconsistencias:
        return 0
    return int(sum(v for v in inconsistencias.values() if isinstance(v, (int, float))))


def resumo_inconsistencias_base(inconsistencias: dict) -> str:
    if not inconsistencias:
        return ""

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

    return "; ".join(mensagens)


def render_base_integrity_alert():
    df_base = st.session_state.df_base

    if df_base.empty:
        return

    inconsistencias = st.session_state.get("base_inconsistencias_carregamento", {})
    total_inconsistencias = total_inconsistencias_base(inconsistencias)
    resumo_inconsistencias = resumo_inconsistencias_base(inconsistencias)

    nomes_normalizados = df_base["Nome"].astype(str).apply(normalizar_nome_comparacao)
    duplicados = nomes_normalizados[nomes_normalizados.duplicated(keep=False)]

    if not duplicados.empty:
        qtd_nomes_duplicados = duplicados.nunique()
        mensagem = (
            f"Atenção: a base atual contém {qtd_nomes_duplicados} nome(s) duplicado(s). "
            "Use o filtro “Mostrar apenas duplicados” para revisar esses registros."
        )
        if total_inconsistencias > 0 and resumo_inconsistencias:
            mensagem += f" Também foram detectadas inconsistências no carregamento: {resumo_inconsistencias}."
        st.warning(mensagem)
        return

    if total_inconsistencias > 0:
        st.warning(
            "Atenção: a base atual foi carregada com inconsistências nos dados. "
            f"Foram detectados: {resumo_inconsistencias}."
        )
        return

    st.caption("Integridade da base: limpa.")


def render_base_preview():
    df_base = st.session_state.df_base

    if df_base.empty:
        return

    render_section_header(
        "Prévia da base atual",
        "Confira rapidamente os jogadores atualmente disponíveis para o sorteio."
    )

    busca_nome = st.text_input(
        "Buscar jogador na base",
        placeholder="Ex: Cleiton",
        key="preview_busca_nome",
    ).strip()

    mostrar_apenas_duplicados = st.checkbox(
        "Mostrar apenas duplicados",
        key="preview_mostrar_apenas_duplicados",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        ordenar_por = st.selectbox(
            "Ordenar por",
            ["Nome", "Posição", "Nota"],
            key="preview_ordenar_por"
        )
    with col2:
        opcoes_mostrar = ["Todos", 10, 20, 50, 100]
        max_linhas = st.selectbox(
            "Mostrar",
            opcoes_mostrar,
            index=0,
            key="preview_max_linhas"
        )

    ascending = True
    if ordenar_por == "Nota":
        ascending = False

    df_preview = df_base.copy()
    nomes_normalizados_base = df_base["Nome"].astype(str).apply(normalizar_nome_comparacao)
    nomes_duplicados_normalizados = set(
        nomes_normalizados_base[nomes_normalizados_base.duplicated(keep=False)].tolist()
    )

    if mostrar_apenas_duplicados:
        nomes_normalizados = df_preview["Nome"].astype(str).apply(normalizar_nome_comparacao)
        mascara_duplicados = nomes_normalizados.isin(nomes_duplicados_normalizados)
        df_preview = df_preview[mascara_duplicados]

    if busca_nome:
        df_preview = df_preview[
            df_preview["Nome"].astype(str).str.contains(busca_nome, case=False, na=False)
        ]

    if busca_nome and mostrar_apenas_duplicados:
        st.caption(f"{len(df_preview)} registro(s) encontrado(s) entre os nomes duplicados.")
    elif busca_nome:
        st.caption(f"{len(df_preview)} jogador(es) encontrado(s).")
    elif mostrar_apenas_duplicados:
        nomes_normalizados = df_preview["Nome"].astype(str).apply(normalizar_nome_comparacao)
        qtd_nomes_duplicados = nomes_normalizados.nunique()
        st.caption(f"{len(df_preview)} registro(s) exibido(s) · {qtd_nomes_duplicados} nome(s) duplicado(s).")

    df_preview["_registro_valido"] = df_preview.apply(registro_valido_para_sorteio, axis=1)
    if ordenar_por == "Nome":
        df_preview = df_preview.sort_values(
            by=["Nome", "_registro_valido"],
            ascending=[True, True]
        ).reset_index(drop=True)
    else:
        df_preview = df_preview.sort_values(
            by=[ordenar_por, "_registro_valido"],
            ascending=[ascending, True]
        ).reset_index(drop=True)

    df_preview = df_preview.drop(columns=["_registro_valido"], errors="ignore")

    if max_linhas != "Todos":
        df_preview = df_preview.head(int(max_linhas))

    def destacar_linha_duplicada(linha):
        chave = normalizar_nome_comparacao(linha["Nome"])
        if chave in nomes_duplicados_normalizados:
            return [""] * len(linha)
        return [""] * len(linha)

    df_preview_display = formatar_df_visual_numeros_inteiros(df_preview)

    st.dataframe(
        df_preview_display.style.apply(destacar_linha_duplicada, axis=1),
        width="stretch",
        hide_index=True
    )



def render_correcao_inline_bloqueios_base(
    logic,
    lista_texto: str,
    nomes_bloqueados_base: list[dict],
    *,
    atualizar_integridade_base_no_estado,
    diagnosticar_lista_no_estado,
    render_action_button,
    inline_flat: bool = False,
    show_intro: bool = True,
    manage_focus: bool = True,
):
    if not nomes_bloqueados_base:
        if manage_focus:
            st.session_state.pop("revisao_foco_bloqueio_nome", None)
        return

    nome_foco = st.session_state.get("revisao_foco_bloqueio_nome")
    nomes_bloqueados_ordenados = nomes_bloqueados_base.copy()
    if nome_foco:
        nomes_bloqueados_ordenados.sort(key=lambda item: 0 if item.get("nome") == nome_foco else 1)

    if show_intro:
        st.caption("Você pode corrigir esses registros sem sair da revisão.")

    for item in nomes_bloqueados_ordenados:
        nome = item["nome"]
        motivos = item.get("motivos", [])
        df_nome = st.session_state.df_base.copy()
        if df_nome.empty:
            continue

        df_nome = df_nome[df_nome["Nome"] == nome].copy()
        if df_nome.empty:
            continue

        df_nome = df_nome.reset_index().rename(columns={"index": "_orig_index"})
        df_nome["_registro_valido"] = df_nome.apply(registro_valido_para_sorteio, axis=1)
        df_nome = df_nome.sort_values(
            by=["_registro_valido", "_orig_index"],
            ascending=[True, True],
        ).reset_index(drop=True)

        expandir_nome = bool(nome_foco == nome or (not nome_foco and len(nomes_bloqueados_ordenados) == 1))
        container = st.container() if inline_flat else st.expander(f"🛠️ Corrigir agora: {nome}", expanded=expandir_nome)

        with container:
            if not inline_flat:
                st.markdown(f"**Motivos detectados:** {'; '.join(motivos)}")
                st.caption("Remova o registro duplicado indesejado ou corrija os dados do registro, incluindo o nome quando necessário.")

            for i, row in df_nome.iterrows():
                idx_original = int(row["_orig_index"])
                registro_valido = registro_valido_para_sorteio(row)
                status = "✅ Registro válido" if registro_valido else "⚠️ Registro inconsistente"

                st.markdown(f"**Registro {i + 1}** — {status}")
                col_info1, col_info2, col_info3, col_info4, col_info5 = st.columns(5)
                col_info1.markdown(f"**Nome**\n\n{row.get('Nome', '')}")
                col_info2.markdown(f"**Posição**\n\n{row.get('Posição', '')}")
                col_info3.markdown(f"**Nota**\n\n{row.get('Nota', '')}")
                col_info4.markdown(f"**Velocidade**\n\n{row.get('Velocidade', '')}")
                col_info5.markdown(f"**Movimentação**\n\n{row.get('Movimentação', '')}")

                if render_action_button(
                    "🗑️ Remover este registro da base",
                    key=f"remover_registro_bloqueado_{nome}_{idx_original}",
                    role="secondary",
                ):
                    st.session_state.df_base = st.session_state.df_base.drop(index=idx_original).reset_index(drop=True)
                    atualizar_integridade_base_no_estado(logic)
                    if lista_texto:
                        diagnosticar_lista_no_estado(logic, lista_texto)
                        st.session_state.revisao_lista_expandida = True
                    st.rerun()

                with st.form(f"form_corrigir_registro_{nome}_{idx_original}"):
                    nome_atual = str(row.get("Nome", "")).strip()
                    nome_corr = st.text_input(
                        "Nome",
                        value=nome_atual,
                        key=f"corrigir_nome_{nome}_{idx_original}",
                    )
                    posicao_atual = str(row.get("Posição", "")).strip().upper()
                    if posicao_atual not in ["D", "M", "A"]:
                        posicao_atual = "M"

                    pos_corr = st.selectbox(
                        "Posição",
                        ["D", "M", "A"],
                        index=["D", "M", "A"].index(posicao_atual),
                        key=f"corrigir_posicao_{nome}_{idx_original}",
                    )
                    nota_corr = st.slider(
                        "Nota",
                        1,
                        10,
                        valor_slider_corrigir(row.get("Nota"), 1, 10, 6),
                        key=f"corrigir_nota_{nome}_{idx_original}",
                    )
                    vel_corr = st.slider(
                        "Velocidade",
                        1,
                        5,
                        valor_slider_corrigir(row.get("Velocidade"), 1, 5, 3),
                        key=f"corrigir_velocidade_{nome}_{idx_original}",
                    )
                    mov_corr = st.slider(
                        "Movimentação",
                        1,
                        5,
                        valor_slider_corrigir(row.get("Movimentação"), 1, 5, 3),
                        key=f"corrigir_movimentacao_{nome}_{idx_original}",
                    )
                    salvar = st.form_submit_button("💾 Salvar correção neste registro")

                    if salvar:
                        nome_corrigido = str(nome_corr).strip()
                        if hasattr(logic, "formatar_nome_visual") and nome_corrigido:
                            nome_corrigido = logic.formatar_nome_visual(nome_corrigido)
                        st.session_state.df_base.loc[idx_original, "Nome"] = nome_corrigido
                        st.session_state.df_base.loc[idx_original, "Posição"] = pos_corr
                        st.session_state.df_base.loc[idx_original, "Nota"] = nota_corr
                        st.session_state.df_base.loc[idx_original, "Velocidade"] = vel_corr
                        st.session_state.df_base.loc[idx_original, "Movimentação"] = mov_corr
                        atualizar_integridade_base_no_estado(logic)
                        if lista_texto:
                            diagnosticar_lista_no_estado(logic, lista_texto)
                            st.session_state.revisao_lista_expandida = True
                        st.rerun()

                st.markdown("---")


def render_correcao_inline_etapa2(
    logic,
    *,
    render_correcao_inline_bloqueios_base,
    atualizar_integridade_base_no_estado,
    diagnosticar_lista_no_estado,
    render_action_button,
):
    bloqueios = listar_bloqueios_base_atual(st.session_state.df_base)
    if not bloqueios:
        return

    with st.expander("🛠️ Corrigir base agora", expanded=False):
        render_correcao_inline_bloqueios_base(
            logic,
            st.session_state.get("lista_texto_revisado", ""),
            bloqueios,
            atualizar_integridade_base_no_estado=atualizar_integridade_base_no_estado,
            diagnosticar_lista_no_estado=diagnosticar_lista_no_estado,
            render_action_button=render_action_button,
        )


def render_revisao_lista(
    logic,
    lista_texto: str,
    *,
    render_action_button,
    diagnosticar_lista_no_estado,
    atualizar_integridade_base_no_estado=None,
    render_correcao_inline_bloqueios_base=None,
    lista_input_key: str = "lista_texto_input",
    **_ignored_kwargs,
):
    diagnostico = st.session_state.diagnostico_lista
    pos_cadastro_pendente = (
        st.session_state.revisao_pendente_pos_cadastro
        and not st.session_state.cadastro_guiado_ativo
        and len(st.session_state.faltantes_revisao) == 0
        and len(st.session_state.faltantes_cadastrados_na_rodada) > 0
    )

    if not diagnostico and not pos_cadastro_pendente:
        return

    expanded = (
        st.session_state.revisao_lista_expandida
        or st.session_state.lista_revisada_confirmada
        or pos_cadastro_pendente
    )

    with st.expander("🔎 Revisão da lista", expanded=expanded):
        if pos_cadastro_pendente and not diagnostico:
            qtd_cadastrados = len(st.session_state.faltantes_cadastrados_na_rodada)
            if qtd_cadastrados == 1:
                st.success("O nome faltante desta revisão foi cadastrado com sucesso.")
            else:
                st.success(f"Os {qtd_cadastrados} nomes faltantes desta revisão foram cadastrados com sucesso.")
            st.caption("Clique em **🔎 Revisar lista novamente** para atualizar a lista final e liberar a confirmação.")

            if render_action_button(
                "🔎 Revisar lista novamente",
                key="revisar_lista_pos_cadastro",
                role="primary",
                use_primary_type=True,
            ):
                diagnostico = diagnosticar_lista_no_estado(logic, lista_texto)
                st.session_state.revisao_pendente_pos_cadastro = False
                st.session_state.faltantes_cadastrados_na_rodada = []
                st.session_state.faltantes_revisao = []
                if diagnostico is None:
                    st.warning("Cole uma lista de jogadores para revisar novamente.")
                st.rerun()
            return

        modo_revisao = diagnostico.get("modo_revisao", "balanceado")
        revisao_aleatoria = modo_revisao == "aleatorio_lista"
        qtd_duplicados = len(diagnostico.get("duplicados", []))
        qtd_correcoes = len(diagnostico.get("correcoes_aplicadas", []))
        qtd_nao_encontrados = len(diagnostico.get("nao_encontrados", []))
        qtd_bloqueios_base = len(diagnostico.get("nomes_bloqueados_base", []))

        tem_pendencia_revisao = (
            qtd_nao_encontrados > 0
            or qtd_duplicados > 0
            or qtd_correcoes > 0
            or diagnostico.get("tem_bloqueio_base", False)
            or st.session_state.cadastro_guiado_ativo
            or pos_cadastro_pendente
        )

        if revisao_aleatoria:
            if st.session_state.lista_revisada_confirmada:
                st.success("Lista confirmada para sorteio aleatório por lista.")
            elif qtd_duplicados > 0:
                st.warning("Modo aleatório por lista: há nomes repetidos. Corrija os nomes abaixo antes de confirmar a lista.")
            else:
                st.success("Lista válida para sorteio aleatório por lista.")
        elif diagnostico.get("tem_bloqueio_base", False):
            st.error("Há problemas na base atual que precisam ser corrigidos antes da confirmação.")
        elif st.session_state.lista_revisada_confirmada:
            st.success("Lista confirmada com sucesso. Agora você já pode sortear os times.")
        elif tem_pendencia_revisao:
            st.warning("A revisão encontrou pendências. Veja os detalhes abaixo antes de confirmar a lista.")
        else:
            st.success("A lista está pronta para confirmação.")

        render_revisao_pendencias_panel(
            logic,
            lista_texto,
            diagnostico,
            revisao_aleatoria=revisao_aleatoria,
            lista_input_key=lista_input_key,
            atualizar_integridade_base_no_estado=atualizar_integridade_base_no_estado,
            diagnosticar_lista_no_estado=diagnosticar_lista_no_estado,
            render_action_button=render_action_button,
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lidos", diagnostico["total_brutos"])
        col2.metric("Prontos", diagnostico["total_validos"])
        col3.metric("Ajustes", qtd_correcoes)
        col4.metric("Pendências", qtd_nao_encontrados + qtd_bloqueios_base + qtd_duplicados)

        lista_final_atual = diagnostico["lista_final_sugerida"]
        lista_final_texto = "\n".join(lista_final_atual)

        st.text_area(
            "Lista final sugerida" if not revisao_aleatoria else "Nomes únicos que entrarão no sorteio",
            value=lista_final_texto,
            height=140,
            disabled=True,
            key="lista_final_sugerida_preview",
        )

        pode_confirmar = (
            diagnostico["total_validos"] > 0
            and (revisao_aleatoria or not diagnostico["tem_nao_encontrados"])
            and qtd_duplicados == 0
            and not diagnostico.get("tem_bloqueio_base", False)
            and not st.session_state.cadastro_guiado_ativo
        )

        if st.session_state.cadastro_guiado_ativo and st.session_state.faltantes_revisao:
            qtd_restantes = len(st.session_state.faltantes_revisao)
            render_step_cta_panel(
                "Continue o cadastro guiado dos faltantes",
                f"Ainda há {qtd_restantes} nome(s) pendente(s) nesta revisão. Conclua esse cadastro para liberar a confirmação da lista.",
                tone="warning",
                eyebrow="Etapa atual",
            )
        elif qtd_nao_encontrados > 0 and not revisao_aleatoria:
            render_step_cta_panel(
                "Cadastre os nomes faltantes para seguir",
                f"A revisão encontrou {qtd_nao_encontrados} nome(s) fora da base atual. Cadastre agora e depois revise novamente a lista.",
                tone="warning",
                eyebrow="Etapa atual",
            )
            if render_action_button(
                "➕ Cadastrar faltantes agora",
                key="revisao_cadastrar_faltantes",
                role="primary",
                use_primary_type=True,
            ):
                st.session_state.faltantes_revisao = diagnostico["nao_encontrados"].copy()
                st.session_state.cadastro_guiado_ativo = True
                st.session_state.faltantes_cadastrados_na_rodada = []
                st.session_state.revisao_pendente_pos_cadastro = False
                st.session_state.lista_revisada_confirmada = False
                st.session_state.lista_revisada = None
                st.session_state.revisao_lista_expandida = True
                st.rerun()
        elif qtd_duplicados > 0:
            render_step_cta_panel(
                "Corrija os nomes repetidos para seguir",
                "A revisão encontrou nomes repetidos na lista. Corrija os nomes no painel de pendências abaixo antes de confirmar a lista final.",
                tone="warning",
                eyebrow="Etapa atual",
            )
        elif diagnostico.get("tem_bloqueio_base", False):
            render_step_cta_panel(
                "Corrija a base atual antes da confirmação",
                "Existem duplicidades ou inconsistências na base. Use os detalhes abaixo para ajustar os registros bloqueados e revise novamente.",
                tone="warning",
                eyebrow="Etapa atual",
            )
        elif pode_confirmar and not st.session_state.lista_revisada_confirmada:
            st.markdown('<div id="revisao-confirmar-anchor"></div>', unsafe_allow_html=True)
            render_step_cta_panel(
                "Confirmar lista final",
                "A lista já está pronta. Confirme agora para liberar os critérios e o botão de sorteio logo em seguida.",
                tone="success",
                eyebrow="Etapa atual",
            )
            if render_action_button(
                "✅ Confirmar lista final",
                key="confirmar_lista_revisada",
                role="primary",
                use_primary_type=True,
            ):
                st.session_state.lista_revisada = diagnostico["lista_final_sugerida"]
                st.session_state.lista_revisada_confirmada = True
                st.session_state.revisao_lista_expandida = False
                st.session_state.scroll_para_sorteio = True
                st.rerun()
        elif st.session_state.lista_revisada_confirmada:
            render_step_cta_panel(
                "Lista final confirmada",
                "A próxima etapa já está liberada. Agora você pode ajustar os critérios e seguir para o sorteio dos times.",
                tone="success",
                eyebrow="Etapa concluída",
            )

        if qtd_correcoes > 0:
            with st.expander(f"🔁 Ajustes automáticos aplicados ({qtd_correcoes})", expanded=False):
                st.caption("Os nomes abaixo foram ajustados automaticamente com base na sua base atual.")
                for item in diagnostico["correcoes_aplicadas"]:
                    st.markdown(f"- `{item['original']}` → `{item['corrigido']}`")

        if qtd_duplicados > 0 or qtd_nao_encontrados > 0 or qtd_bloqueios_base > 0:
            total_pendencias = qtd_nao_encontrados + qtd_bloqueios_base + qtd_duplicados
            with st.expander(f"ℹ️ Resumo complementar das pendências ({total_pendencias})", expanded=False):
                st.caption("Use o painel de pendências acima como ponto principal de correção. Este bloco fica apenas como resumo rápido.")
                if qtd_duplicados > 0:
                    st.markdown(f"- Duplicados na lista: **{qtd_duplicados}**")
                if qtd_nao_encontrados > 0 and not revisao_aleatoria:
                    st.markdown(f"- Nomes não encontrados: **{qtd_nao_encontrados}**")
                if qtd_bloqueios_base > 0:
                    st.markdown(f"- Bloqueios da base: **{qtd_bloqueios_base}**")

        if st.session_state.cadastro_guiado_ativo and st.session_state.faltantes_revisao:
            faltantes_restantes = st.session_state.faltantes_revisao
            faltantes_feitos = st.session_state.faltantes_cadastrados_na_rodada
            nome_atual = faltantes_restantes[0]
            total_rodada = len(faltantes_restantes) + len(faltantes_feitos)
            indice_atual = len(faltantes_feitos) + 1
            ultimo_da_fila = len(faltantes_restantes) == 1

            with st.expander("📝 Cadastro guiado de faltantes", expanded=True):
                st.info(
                    f"Cadastro guiado iniciado — jogador {indice_atual} de {total_rodada}: **{nome_atual}**"
                )
                st.markdown(f"**Cadastrando agora:** {nome_atual}")

                with st.form("form_add_manual_guiado_inline"):
                    p_m = st.selectbox("Posição", ["M", "A", "D"], key="guiado_inline_posicao")
                    n_m = st.slider("Nota", 1, 10, 6, key="guiado_inline_nota")
                    v_m = st.slider("Velocidade", 1, 5, 3, key="guiado_inline_velocidade")
                    mv_m = st.slider("Movimentação", 1, 5, 3, key="guiado_inline_movimentacao")
                    label_submit = "Salvar e concluir" if ultimo_da_fila else "Salvar e próximo faltante"

                    with st.container(key="action-primary-form-salvar-faltante"):
                        submit_guiado = st.form_submit_button(label_submit)

                    if submit_guiado:
                        novo_nome = logic.formatar_nome_visual(nome_atual)
                        novo = {
                            'Nome': novo_nome,
                            'Nota': n_m,
                            'Posição': p_m,
                            'Velocidade': v_m,
                            'Movimentação': mv_m,
                        }
                        st.session_state.df_base.loc[len(st.session_state.df_base)] = novo
                        st.session_state.faltantes_cadastrados_na_rodada.append(novo_nome)
                        st.session_state.faltantes_revisao.pop(0)
                        st.session_state.lista_revisada_confirmada = False
                        st.session_state.lista_revisada = None
                        st.session_state.diagnostico_lista = None
                        st.session_state.revisao_lista_expandida = True

                        if not st.session_state.faltantes_revisao:
                            st.session_state.cadastro_guiado_ativo = False
                            st.session_state.faltantes_revisao = []
                            st.session_state.revisao_pendente_pos_cadastro = True

                        st.rerun()

        edicao_key = "lista_revisao_edicao"
        edicao_origem_key = "lista_revisao_edicao_origem"
        remover_key = "lista_revisao_remover"

        if st.session_state.get(edicao_origem_key) != lista_final_texto:
            st.session_state[edicao_key] = lista_final_texto
            st.session_state[edicao_origem_key] = lista_final_texto
            st.session_state[remover_key] = []

        with st.expander("✏️ Edição rápida da lista", expanded=False):
            st.caption(
                "Use este bloco para remover nomes, ajustar a lista rapidamente e reaplicar a revisão sem precisar colar tudo novamente."
            )

            nomes_para_remover = st.multiselect(
                "Remover nomes da lista revisada",
                options=lista_final_atual,
                default=st.session_state.get(remover_key, []),
                key=remover_key,
            )

            if render_action_button(
                "➖ Remover nomes selecionados",
                key="acao_remover_nomes_lista_revisada",
                role="secondary",
            ):
                texto_atual = str(st.session_state.get(edicao_key, lista_final_texto))
                linhas_atuais = [linha.strip() for linha in texto_atual.splitlines() if linha.strip()]
                linhas_restantes = [linha for linha in linhas_atuais if linha not in set(nomes_para_remover)]
                st.session_state[edicao_key] = "\n".join(linhas_restantes)
                st.session_state[remover_key] = []
                st.rerun()

            st.text_area(
                "Editar lista revisada",
                key=edicao_key,
                height=180,
                help="Você pode apagar nomes, reorganizar linhas ou ajustar rapidamente a lista antes de reaplicar a revisão.",
            )

            col_ed1, col_ed2 = st.columns(2)
            aplicar_edicao = col_ed1.button(
                "✅ Aplicar alterações e revisar novamente",
                key="acao_aplicar_edicao_lista_revisada",
                type="primary",
                use_container_width=True,
            )
            restaurar_edicao = col_ed2.button(
                "↺ Restaurar lista revisada atual",
                key="acao_restaurar_edicao_lista_revisada",
                use_container_width=True,
            )

            if restaurar_edicao:
                st.session_state[edicao_key] = lista_final_texto
                st.session_state[remover_key] = []
                st.rerun()

            if aplicar_edicao:
                novo_texto_lista = str(st.session_state.get(edicao_key, "")).strip()
                st.session_state[f"{lista_input_key}__pending"] = novo_texto_lista
                st.session_state[f"{lista_input_key}__revisar"] = True
                st.rerun()

        if diagnostico["ignorados"]:
            with st.expander(f"ℹ️ Itens ignorados na leitura ({len(diagnostico['ignorados'])})", expanded=False):
                for item in diagnostico["ignorados"]:
                    st.markdown(f"- {item}")


def abrir_expander_grupo():
    st.session_state.grupo_config_expanded = True


def grupo_config_deve_abrir() -> bool:
    if "grupo_config_expanded" not in st.session_state:
        st.session_state.grupo_config_expanded = True
    return bool(
        st.session_state.get("grupo_config_expanded", True)
        or str(st.session_state.get("grupo_nome_pelada", "")).strip()
        or str(st.session_state.get("grupo_senha_admin", "")).strip()
        or st.session_state.get("senha_admin_confirmada", False)
    )


def ativar_fluxo_somente_lista(logic):
    st.session_state.df_base = logic.criar_base_vazia()
    st.session_state.novos_jogadores = []
    st.session_state.is_admin = False
    st.session_state.base_admin_carregada = False
    st.session_state.ultimo_arquivo = None
    st.session_state.qtd_jogadores_adicionados_manualmente = 0
    st.session_state.senha_admin_confirmada = False
    st.session_state.base_inconsistencias_carregamento = {}
    st.session_state.base_registros_inconsistentes_carregamento = []
    st.session_state.grupo_nome_pelada = ""
    st.session_state.grupo_senha_admin = ""
    st.session_state.grupo_origem_fluxo = "lista"
    st.session_state.grupo_config_expanded = False
    st.session_state.scroll_para_lista = True
    st.session_state.lista_revisada_confirmada = False
    st.session_state.lista_revisada = None
    st.session_state.diagnostico_lista = None
    st.rerun()


def render_group_config_expander(logic, nome_pelada_adm: str, senha_adm: str) -> str:
    st.session_state.setdefault("grupo_config_expanded", True)
    st.session_state.setdefault("grupo_origem_fluxo", None)
    st.session_state.setdefault("grupo_busca_status", "idle")
    st.session_state.setdefault("grupo_nome_ultima_busca", "")

    with st.expander(
        resumo_expander_configuracao(nome_pelada_adm),
        expanded=True,
    ):
        st.markdown("**Como deseja iniciar o sorteio?**")
        col_lista, col_admin, col_excel = st.columns(3)
        with col_lista:
            if st.button("🎲 Apenas sorteio com lista", key="grupo_escolher_lista"):
                ativar_fluxo_somente_lista(logic)
        with col_admin:
            if st.button("🗂️ Carregar base do grupo", key="grupo_escolher_admin"):
                st.session_state.grupo_origem_fluxo = "admin"
                st.session_state.grupo_config_expanded = True
                st.rerun()
        with col_excel:
            if st.button("📄 Usar Excel próprio", key="grupo_escolher_excel"):
                st.session_state.grupo_origem_fluxo = "excel"
                st.session_state.grupo_config_expanded = True
                st.rerun()

        if "grupo_nome_pelada__pending" in st.session_state:
            st.session_state["grupo_nome_pelada"] = st.session_state.pop("grupo_nome_pelada__pending")
        if "grupo_senha_admin__pending" in st.session_state:
            st.session_state["grupo_senha_admin"] = st.session_state.pop("grupo_senha_admin__pending")

        origem_fluxo = st.session_state.get("grupo_origem_fluxo")
        nome_pelada = str(st.session_state.get("grupo_nome_pelada", "")).strip()
        uploaded_file = None
        base_grupo_carregada = bool(st.session_state.base_admin_carregada and st.session_state.is_admin)

        if origem_fluxo == "admin":
            st.markdown("---")
            st.markdown("**🗂️ Carregar base do grupo**")
            nome_digitado = st.text_input(
                "Nome da pelada:",
                placeholder="Ex: Pelada de Domingo",
                key="grupo_nome_pelada",
            ).strip()
            nome_pelada = nome_digitado

            ultima_busca = str(st.session_state.get("grupo_nome_ultima_busca", "")).strip()
            busca_status = st.session_state.get("grupo_busca_status", "idle")
            if nome_digitado != ultima_busca and not base_grupo_carregada:
                busca_status = "idle"
                st.session_state.grupo_busca_status = "idle"
                st.session_state.senha_admin_confirmada = False

            if st.button("🔎 Buscar grupo", key="grupo_buscar_nome"):
                st.session_state.grupo_nome_ultima_busca = nome_digitado
                if nome_digitado and nome_digitado.upper() == str(nome_pelada_adm).upper():
                    st.session_state.grupo_busca_status = "found"
                    st.session_state.scroll_para_confirmar_senha = True
                else:
                    st.session_state.grupo_busca_status = "not_found" if nome_digitado else "idle"
                st.rerun()

            busca_status = st.session_state.get("grupo_busca_status", "idle")
            senha_atual = st.session_state.get("grupo_senha_admin", "")
            if st.session_state.ultima_senha_digitada != senha_atual:
                st.session_state.senha_admin_confirmada = False
                st.session_state.ultima_senha_digitada = senha_atual

            if base_grupo_carregada:
                st.success("Base do grupo carregada com sucesso.")
            elif busca_status == "found":
                st.success("Base encontrada para esse grupo.")
                st.caption("Informe a senha para carregar a base.")
            elif busca_status == "not_found":
                st.warning("Grupo não encontrado. Confira o nome informado ou escolha a opção de Excel próprio.")
            else:
                st.info("Informe o nome da pelada e clique em **Buscar grupo** para localizar a base.")

            if busca_status == "found" and not base_grupo_carregada:
                st.markdown('<div id="confirmar-senha-anchor"></div>', unsafe_allow_html=True)
                if st.session_state.get("scroll_para_confirmar_senha", False):
                    components.html(
                        """
                        <script>
                        const parentDoc = window.parent.document;
                        const anchor = parentDoc.getElementById("confirmar-senha-anchor");
                        if (anchor) {
                            anchor.scrollIntoView({ behavior: "smooth", block: "center" });
                        }
                        </script>
                        """,
                        height=0,
                    )
                    st.session_state.scroll_para_confirmar_senha = False
                senha = st.text_input(
                    "Senha:",
                    type="password",
                    key="grupo_senha_admin",
                )
                if st.button(
                    "📥 Carregar base de dados",
                    key="grupo_confirmar_senha",
                ):
                    if senha != str(senha_adm):
                        st.session_state.senha_admin_confirmada = False
                        st.session_state.ultima_senha_digitada = senha
                        st.session_state.is_admin = False
                        st.session_state.base_admin_carregada = False
                        st.error("Senha incorreta")
                    else:
                        st.session_state.senha_admin_confirmada = True
                        st.session_state.ultima_senha_digitada = senha
                        registrar_base_carregada_no_estado(
                            logic,
                            logic.carregar_dados_originais(),
                            is_admin=True,
                            ultimo_arquivo=None,
                        )
                        st.session_state.grupo_config_expanded = False
                        st.success(f"Base carregada: {len(st.session_state.df_base)} jogadores.")
                        st.rerun()

        elif origem_fluxo == "excel":
            st.markdown("---")
            st.markdown("**📄 Usar Excel próprio**")
            st.caption("Envie sua planilha e depois toque em **Carregar base de dados**.")
            uploaded_file = st.file_uploader(
                "Enviar planilha Excel",
                type=["xlsx"],
                label_visibility="collapsed",
                key="grupo_upload_planilha",
            )

            if st.button(
                "📥 Carregar base de dados",
                key="grupo_carregar_base",
            ):
                if uploaded_file is None:
                    st.info("Você ainda não selecionou uma planilha própria para carregar.")
                else:
                    df_novo = logic.processar_upload(uploaded_file)
                    if df_novo is not None:
                        registrar_base_carregada_no_estado(
                            logic,
                            df_novo,
                            is_admin=False,
                            ultimo_arquivo=uploaded_file.name,
                        )
                        st.session_state.senha_admin_confirmada = False
                        st.session_state.grupo_config_expanded = False
                        st.success("Arquivo carregado!")
                        st.rerun()
        elif origem_fluxo == "lista":
            st.markdown("---")
            st.markdown("**🎲 Apenas sorteio com lista**")
            st.info("Neste modo, você não precisa carregar base nem Excel. O app usará apenas os nomes informados na lista para um sorteio aleatório.")
            st.caption("Você será levado diretamente para a seção da lista. Se houver nomes repetidos, revise e corrija cada ocorrência antes do sorteio.")
        else:
            st.caption("Escolha uma opção para começar: usar a base do grupo, enviar um Excel próprio ou seguir direto para o sorteio apenas com lista.")

        if (
            not st.session_state.df_base.empty
            or st.session_state.novos_jogadores
            or st.session_state.is_admin
        ):
            with st.expander("Ações secundárias", expanded=False):
                st.caption("Use a limpeza apenas quando quiser reiniciar a base atual.")
                if st.button("🗑 Limpar base atual", key="grupo_limpar_base_atual"):
                    st.session_state.df_base = logic.criar_base_vazia()
                    st.session_state.novos_jogadores = []
                    st.session_state.is_admin = False
                    st.session_state.base_admin_carregada = False
                    st.session_state.ultimo_arquivo = None
                    st.session_state.qtd_jogadores_adicionados_manualmente = 0
                    st.session_state.senha_admin_confirmada = False
                    st.session_state.base_inconsistencias_carregamento = {}
                    st.session_state.base_registros_inconsistentes_carregamento = []
                    st.session_state.grupo_busca_status = "idle"
                    st.session_state.grupo_nome_ultima_busca = ""
                    st.session_state.grupo_origem_fluxo = None
                    st.session_state.grupo_config_expanded = True
                    st.session_state["grupo_nome_pelada__pending"] = ""
                    st.session_state["grupo_senha_admin__pending"] = ""
                    st.rerun()

    return nome_pelada

