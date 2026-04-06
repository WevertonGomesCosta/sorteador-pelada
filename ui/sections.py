"""Seções visuais simples do Sorteador Pelada PRO."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.validators import normalizar_nome_comparacao, registro_valido_para_sorteio


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
    grupo_encontrado = bool(nome_pelada) and nome_pelada.upper() == str(nome_pelada_adm).upper()
    nome_nao_encontrado = bool(nome_pelada) and not grupo_encontrado and not base_admin_carregada and not base_upload_carregada

    if base_admin_carregada:
        status = "Base admin carregada"
    elif base_upload_carregada:
        status = "Planilha própria carregada"
    elif grupo_encontrado:
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
        origem = "Admin"
    elif qtd_jogadores == 0:
        origem = "Vazia"
    else:
        origem = "Sua base"

    modo = "ADMIN" if st.session_state.is_admin else "Público"

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
