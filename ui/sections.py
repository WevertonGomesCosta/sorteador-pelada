"""Seções visuais simples do Sorteador Pelada PRO."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from state.session import registrar_base_carregada_no_estado

from core.validators import (
    listar_bloqueios_base_atual,
    normalizar_nome_comparacao,
    registro_valido_para_sorteio,
    valor_slider_corrigir,
)


def render_section_header(titulo: str, subtitulo: str | None = None):
    st.markdown(f"<div class='section-title'>{titulo}</div>", unsafe_allow_html=True)
    if subtitulo:
        st.markdown(f"<div class='section-subtitle'>{subtitulo}</div>", unsafe_allow_html=True)


def _titulo_expander(rotulo: str, status: str) -> str:
    return f"{rotulo} · {status}"


def resumo_expander_configuracao(nome_pelada_adm: str) -> str:
    nome_pelada = str(st.session_state.get("grupo_nome_pelada", "")).strip()
    base_admin_carregada = bool(st.session_state.get("base_admin_carregada", False) and st.session_state.get("is_admin", False))
    base_upload_carregada = bool(st.session_state.get("ultimo_arquivo")) and not st.session_state.get("is_admin", False)
    fluxo_lista = st.session_state.get("grupo_origem_fluxo") == "lista"
    busca_status = st.session_state.get("grupo_busca_status", "idle")
    nome_nao_encontrado = bool(nome_pelada) and busca_status == "not_found" and not base_admin_carregada and not base_upload_carregada

    if base_admin_carregada:
        status = "Base do grupo carregada"
    elif base_upload_carregada:
        status = "Excel próprio carregado"
    elif fluxo_lista:
        status = "Somente lista"
    elif busca_status == "found":
        status = "Grupo encontrado"
    elif nome_nao_encontrado:
        status = "Nome não encontrado"
    else:
        status = "Sem base"

    return _titulo_expander("⚙️ Grupo e base", status)


def _qtd_adicoes_manuais() -> int:
    return int(st.session_state.get("qtd_jogadores_adicionados_manualmente", 0))


def resumo_expander_cadastro_manual() -> str:
    cadastro_guiado_ativo = bool(st.session_state.get("cadastro_guiado_ativo", False))
    cadastro_guiado_concluido = bool(
        st.session_state.get("revisao_pendente_pos_cadastro", False)
        and len(st.session_state.get("faltantes_cadastrados_na_rodada", [])) > 0
        and not cadastro_guiado_ativo
    )
    qtd_manual = _qtd_adicoes_manuais()

    if cadastro_guiado_ativo:
        status = "Cadastro guiado ativo"
    elif cadastro_guiado_concluido:
        status = "Faltantes cadastrados"
    elif qtd_manual > 0:
        status = f"{qtd_manual} adicionados"
    else:
        status = "Opcional"

    return _titulo_expander("📝 Cadastro manual", status)


def obter_criterios_ativos() -> dict:
    return {
        "pos": bool(st.session_state.get("criterio_posicao", True)),
        "nota": bool(st.session_state.get("criterio_nota", True)),
        "vel": bool(st.session_state.get("criterio_velocidade", True)),
        "mov": bool(st.session_state.get("criterio_movimentacao", True)),
    }


def _criterios_estao_no_padrao() -> bool:
    criterios = obter_criterios_ativos()
    return (
        criterios["pos"],
        criterios["nota"],
        criterios["vel"],
        criterios["mov"],
    ) == (True, True, True, True)


def resumo_criterios_ativos() -> str:
    criterios = obter_criterios_ativos()
    ativos = []

    if criterios["pos"]:
        ativos.append("Posição")
    if criterios["nota"]:
        ativos.append("Nota")
    if criterios["vel"]:
        ativos.append("Velocidade")
    if criterios["mov"]:
        ativos.append("Movimentação")

    if len(ativos) == 4:
        return "Padrão · Posição, Nota, Velocidade e Movimentação"
    if not ativos:
        return "Personalizado · Nenhum critério ativo"

    return "Personalizado · " + ", ".join(ativos)


def resumo_expander_criterios() -> str:
    status = "Padrão" if _criterios_estao_no_padrao() else "Personalizado"
    return _titulo_expander("⚙️ Critérios", status)


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


def render_base_inconsistencias_expander():
    registros = st.session_state.get("base_registros_inconsistentes_carregamento", [])
    if not registros:
        return

    df_inconsistentes = pd.DataFrame(registros)
    if df_inconsistentes.empty:
        return

    with st.expander("⚠️ Registros com inconsistências", expanded=False):
        st.caption("Os registros abaixo foram carregados, mas merecem revisão antes do uso.")
        df_inconsistentes_display = df_inconsistentes.copy()
        styler = df_inconsistentes_display.style.apply(estilo_celulas_inconsistentes, axis=None)
        st.dataframe(
            styler,
            width="stretch",
            hide_index=True,
        )


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
):
    if not nomes_bloqueados_base:
        return

    st.caption("Você pode corrigir esses registros sem sair da revisão.")
    for item in nomes_bloqueados_base:
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
            ascending=[True, True]
        ).reset_index(drop=True)

        with st.expander(f"🛠️ Corrigir agora: {nome}", expanded=False):
            st.markdown(f"**Motivos detectados:** {'; '.join(motivos)}")
            st.caption("Remova o registro duplicado indesejado ou corrija os critérios do registro inconsistente.")

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
                    st.session_state.df_base = (
                        st.session_state.df_base.drop(index=idx_original).reset_index(drop=True)
                    )
                    atualizar_integridade_base_no_estado(logic)
                    if lista_texto:
                        diagnosticar_lista_no_estado(logic, lista_texto)
                        st.session_state.revisao_lista_expandida = True
                    st.rerun()

                with st.form(f"form_corrigir_registro_{nome}_{idx_original}"):
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
                st.success("O faltante desta revisão foi cadastrado com sucesso.")
            else:
                st.success(f"Os {qtd_cadastrados} faltantes desta revisão foram cadastrados com sucesso.")
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

        tem_pendencia_revisao = (
            len(diagnostico.get("nao_encontrados", [])) > 0
            or len(diagnostico.get("duplicados", [])) > 0
            or len(diagnostico.get("correcoes_aplicadas", [])) > 0
            or diagnostico.get("tem_bloqueio_base", False)
            or st.session_state.cadastro_guiado_ativo
            or pos_cadastro_pendente
        )

        if revisao_aleatoria:
            if diagnostico.get("duplicados"):
                st.warning("Modo aleatório por lista: nomes repetidos serão unificados antes do sorteio.")
            else:
                st.success("Lista válida para sorteio aleatório por lista.")
        elif diagnostico.get("tem_bloqueio_base", False):
            st.error("A lista não pode ser confirmada porque há nomes com duplicidade ou inconsistência na base atual.")
        elif st.session_state.lista_revisada_confirmada:
            st.success("Lista confirmada com sucesso. Agora você já pode sortear os times.")
        elif tem_pendencia_revisao:
            st.warning("Há pendências na lista. Resolva os pontos acima para continuar.")
        elif revisao_aleatoria:
            st.info("Revisão concluída em modo aleatório por lista. O sorteio usará apenas os nomes únicos informados.")
        else:
            st.success("A lista está pronta para confirmação.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lidos", diagnostico["total_brutos"])
        col2.metric("Prontos", diagnostico["total_validos"])
        col3.metric("Correções", len(diagnostico["correcoes_aplicadas"]))
        col4.metric("Não encontrados", len(diagnostico["nao_encontrados"]))

        if revisao_aleatoria and diagnostico.get("duplicados"):
            st.info("Nomes repetidos na lista foram identificados e serão considerados uma única vez no sorteio aleatório.")
            for nome in diagnostico["duplicados"]:
                st.markdown(f"- {nome}")

        if diagnostico["correcoes_aplicadas"]:
            st.info("Alguns nomes foram ajustados com base na sua base atual.")
            for item in diagnostico["correcoes_aplicadas"]:
                st.markdown(f"- `{item['original']}` → `{item['corrigido']}`")

        if diagnostico["duplicados"] and not revisao_aleatoria:
            st.warning("Encontramos nomes repetidos na lista. Apenas a primeira ocorrência será mantida na sugestão final.")
            for nome in diagnostico["duplicados"]:
                st.markdown(f"- {nome}")

        if diagnostico.get("nomes_bloqueados_base"):
            st.error("Os nomes abaixo têm duplicidade ou inconsistência na base atual e precisam ser corrigidos antes da confirmação:")
            for item in diagnostico["nomes_bloqueados_base"]:
                st.markdown(f"- **{item['nome']}** — {'; '.join(item['motivos'])}")
            render_correcao_inline_bloqueios_base(
                logic,
                lista_texto,
                diagnostico["nomes_bloqueados_base"],
                atualizar_integridade_base_no_estado=atualizar_integridade_base_no_estado,
                diagnosticar_lista_no_estado=diagnosticar_lista_no_estado,
                render_action_button=render_action_button,
            )

        if diagnostico["nao_encontrados"] and not revisao_aleatoria:
            st.error("Alguns nomes não foram encontrados na base atual.")
            for nome in diagnostico["nao_encontrados"]:
                st.markdown(f"- {nome}")
            st.caption("Cadastre esses jogadores agora no formulário abaixo e depois revise a lista novamente.")
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

        if st.session_state.cadastro_guiado_ativo and st.session_state.faltantes_revisao:
            faltantes_restantes = st.session_state.faltantes_revisao
            faltantes_feitos = st.session_state.faltantes_cadastrados_na_rodada
            nome_atual = faltantes_restantes[0]
            total_rodada = len(faltantes_restantes) + len(faltantes_feitos)
            indice_atual = len(faltantes_feitos) + 1
            ultimo_da_fila = len(faltantes_restantes) == 1

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

        if diagnostico["ignorados"]:
            with st.expander("Itens ignorados na leitura", expanded=False):
                for item in diagnostico["ignorados"]:
                    st.markdown(f"- {item}")

        lista_final_atual = diagnostico["lista_final_sugerida"]
        lista_final_texto = "\n".join(lista_final_atual)

        st.text_area(
            "Lista final sugerida" if not revisao_aleatoria else "Nomes únicos que entrarão no sorteio",
            value=lista_final_texto,
            height=140,
            disabled=True,
            key="lista_final_sugerida_preview",
        )

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
                st.session_state[lista_input_key] = novo_texto_lista
                diagnostico_editado = diagnosticar_lista_no_estado(logic, novo_texto_lista)
                if diagnostico_editado is None:
                    st.warning("A lista editada ficou vazia. Ajuste os nomes e tente novamente.")
                st.rerun()

        pode_confirmar = (
            diagnostico["total_validos"] > 0
            and not diagnostico["tem_nao_encontrados"]
            and not diagnostico.get("tem_bloqueio_base", False)
            and not st.session_state.cadastro_guiado_ativo
            and not revisao_aleatoria
        )
        if pode_confirmar and not st.session_state.lista_revisada_confirmada:
            if render_action_button(
                "✅ Confirmar lista final",
                key="confirmar_lista_revisada",
                role="primary",
                use_primary_type=True,
            ):
                st.session_state.lista_revisada = diagnostico["lista_final_sugerida"]
                st.session_state.lista_revisada_confirmada = True
                st.session_state.revisao_lista_expandida = False
                st.rerun()

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
    st.session_state.grupo_busca_status = "idle"
    st.session_state.grupo_nome_ultima_busca = ""
    st.session_state.grupo_origem_fluxo = "lista"
    st.session_state.grupo_config_expanded = False
    st.session_state.scroll_para_lista = True
    st.session_state.lista_revisada_confirmada = False
    st.session_state.lista_revisada = None
    st.session_state.diagnostico_lista = None
    st.rerun()


def render_group_config_expander(logic, nome_pelada_adm: str, senha_adm: str) -> str:
    if "grupo_config_expanded" not in st.session_state:
        st.session_state.grupo_config_expanded = True
    if "grupo_origem_fluxo" not in st.session_state:
        st.session_state.grupo_origem_fluxo = None

    with st.expander(
        resumo_expander_configuracao(nome_pelada_adm),
        expanded=True,
    ):
        st.markdown("**Como deseja iniciar o sorteio?**")
        col_lista, col_admin, col_excel = st.columns([1, 1, 1])
        with col_lista:
            if st.button("🎲 Apenas sorteio com lista", key="grupo_escolher_lista", use_container_width=True):
                ativar_fluxo_somente_lista(logic)
        with col_admin:
            if st.button("🗂️ Carregar base do grupo", key="grupo_escolher_admin", use_container_width=True):
                st.session_state.grupo_origem_fluxo = "admin"
                st.session_state.grupo_config_expanded = True
                st.session_state.grupo_busca_status = "idle"
                st.session_state.grupo_nome_ultima_busca = ""
                st.rerun()
        with col_excel:
            if st.button("📄 Usar Excel próprio", key="grupo_escolher_excel", use_container_width=True):
                st.session_state.grupo_origem_fluxo = "excel"
                st.session_state.grupo_config_expanded = True
                st.session_state.grupo_busca_status = "idle"
                st.session_state.grupo_nome_ultima_busca = ""
                st.rerun()

        origem_fluxo = st.session_state.get("grupo_origem_fluxo")
        nome_pelada = str(st.session_state.get("grupo_nome_pelada", "")).strip()
        nome_informado = nome_pelada
        uploaded_file = None
        base_grupo_carregada = st.session_state.base_admin_carregada
        busca_status = st.session_state.get("grupo_busca_status", "idle")
        ultima_busca = str(st.session_state.get("grupo_nome_ultima_busca", "")).strip()

        if origem_fluxo == "admin":
            st.markdown("---")
            st.markdown("**🗂️ Carregar base do grupo**")
            nome_pelada = st.text_input(
                "Nome da pelada:",
                placeholder="Ex: Pelada de Domingo",
                key="grupo_nome_pelada",
            )
            nome_informado = nome_pelada.strip()
            if nome_informado != ultima_busca:
                busca_status = "idle"
                st.session_state.grupo_busca_status = "idle"
                st.session_state.senha_admin_confirmada = False

            if st.button("🔎 Buscar grupo", key="grupo_buscar_nome"):
                st.session_state.grupo_nome_ultima_busca = nome_informado
                if nome_informado and nome_informado.upper() == str(nome_pelada_adm).upper():
                    st.session_state.grupo_busca_status = "found"
                    busca_status = "found"
                elif nome_informado:
                    st.session_state.grupo_busca_status = "not_found"
                    busca_status = "not_found"
                else:
                    st.session_state.grupo_busca_status = "idle"
                    busca_status = "idle"
                st.session_state.senha_admin_confirmada = False
                st.rerun()

            grupo_encontrado = busca_status == "found"

            if base_grupo_carregada and st.session_state.is_admin:
                st.success("Base do grupo carregada com sucesso.")
            elif grupo_encontrado:
                st.success("Base encontrada para esse grupo.")
                st.caption("Informe a senha para carregar a base.")
            elif busca_status == "not_found":
                st.warning("Grupo não encontrado. Confira o nome informado ou escolha a opção de Excel próprio.")
            else:
                st.info("Informe o nome da pelada e toque em Buscar grupo.")

            senha_atual = st.session_state.get("grupo_senha_admin", "")
            if st.session_state.ultima_senha_digitada != senha_atual:
                st.session_state.senha_admin_confirmada = False
                st.session_state.ultima_senha_digitada = senha_atual

            if grupo_encontrado and not base_grupo_carregada:
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
            st.caption("Você será levado diretamente para a seção da lista. Nomes repetidos serão unificados antes do sorteio.")
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
                    st.session_state.grupo_origem_fluxo = None
                    st.session_state.grupo_config_expanded = True
                    st.rerun()

    return nome_pelada

