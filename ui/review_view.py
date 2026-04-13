"""Fluxos de revisão da lista e correções inline do Sorteador Pelada PRO."""

from __future__ import annotations

import html
import re

import streamlit as st

import state.keys as K

from core.validators import (
    listar_bloqueios_base_atual,
    normalizar_nome_comparacao,
    registro_valido_para_sorteio,
    valor_slider_corrigir,
)
from ui.panels import render_step_cta_panel
from ui.primitives import render_inline_status_note

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




def _render_resumo_revisao_visual(total_brutos: int, total_validos: int, qtd_correcoes: int, total_pendencias: int, *, compacto: bool = False) -> None:
    if compacto:
        st.markdown(
            f"""
            <div class="review-summary-inline">
                <span class="review-summary-inline__label">Resumo rápido</span>
                <span class="review-summary-inline__item">Lidos: <strong>{total_brutos}</strong></span>
                <span class="review-summary-inline__item">Prontos: <strong>{total_validos}</strong></span>
                <span class="review-summary-inline__item">Ajustes: <strong>{qtd_correcoes}</strong></span>
                <span class="review-summary-inline__item">Pendências: <strong>{total_pendencias}</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    cards = [
        ("Lidos", str(total_brutos)),
        ("Prontos", str(total_validos)),
        ("Ajustes", str(qtd_correcoes)),
        ("Pendências", str(total_pendencias)),
    ]
    cards_html = "".join(
        f'<div class="review-summary-card"><div class="review-summary-card__label">{html.escape(rotulo)}</div><div class="review-summary-card__value">{html.escape(valor)}</div></div>'
        for rotulo, valor in cards
    )
    st.markdown(
        f"""
        <div class="review-summary-grid">
            {cards_html}
        </div>
        """,
        unsafe_allow_html=True,
    )



def _render_revisao_status_banner(mensagem: str, *, tone: str) -> None:
    titulo = {
        "success": "Revisão pronta",
        "warning": "Atenção na revisão",
        "error": "Correção necessária",
        "info": "Status da revisão",
    }.get(tone, "Status da revisão")
    render_inline_status_note(titulo, mensagem, tone=tone)


def _render_lista_final_preview(rotulo: str, itens: list[str]) -> None:
    linhas = []
    for item in itens:
        texto = str(item or "").strip()
        if not texto:
            continue
        linhas.append(f'<div class="review-list-preview__line">{html.escape(texto)}</div>')

    if not linhas:
        linhas.append('<div class="review-list-preview__line review-list-preview__line--muted">Nenhum nome pronto para exibir.</div>')

    conteudo = "".join(linhas)
    st.markdown(
        f"""
        <div class="review-list-preview">
            <div class="review-list-preview__label">{html.escape(rotulo)}</div>
            <div class="review-list-preview__body">{conteudo}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_pendencia_meta(tipo: str) -> dict[str, str]:
    mapa = {
        "bloqueio_base": {
            "grupo": "Bloqueios da base",
            "gravidade": "Bloqueia sorteio",
            "tone": "error",
            "acao_principal": "Editar registro da base",
            "apoio": "Corrija ou remova o registro aqui mesmo para liberar a revisão.",
        },
        "fora_base": {
            "grupo": "Fora da base",
            "gravidade": "Corrigir antes de seguir",
            "tone": "warning",
            "acao_principal": "Corrigir nome na lista",
            "apoio": "Você pode corrigir o nome, cadastrar na base ou remover o item.",
        },
        "duplicado_lista": {
            "grupo": "Duplicados na lista",
            "gravidade": "Revisar ocorrências",
            "tone": "warning",
            "acao_principal": "Revisar duplicidade",
            "apoio": "Edite as ocorrências ou remova a entrada indevida.",
        },
    }
    return mapa.get(
        tipo,
        {
            "grupo": "Pendência",
            "gravidade": "Revisar",
            "tone": "info",
            "acao_principal": "Revisar item",
            "apoio": "Confira os detalhes deste item antes de seguir.",
        },
    )



def _render_pendencia_item_intro(
    tipo: str,
    nome_item: str,
    *,
    detalhe: str | None = None,
    acao_principal: str | None = None,
) -> None:
    meta = _get_pendencia_meta(tipo)
    acao = str(acao_principal or meta["acao_principal"]).strip()

    render_inline_status_note(
        meta["gravidade"],
        f"{nome_item} — Próxima ação: {acao}",
        tone=meta["tone"],
    )

    if detalhe:
        st.caption(detalhe)
    else:
        st.caption(meta["apoio"])



def _expandir_bloqueio_base_padrao(
    *,
    idx: int,
    nome: str,
    qtd_bloqueios_base: int,
    expandir_primeiro_bloqueio: bool,
) -> bool:
    nome_foco = st.session_state.get(K.REVISAO_FOCO_BLOQUEIO_NOME)
    return (
        qtd_bloqueios_base == 1
        or (expandir_primeiro_bloqueio and idx == 0)
        or nome_foco == nome
    )



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
        return 0

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
            <div class="review-pending-panel__title">Corrija os itens abaixo</div>
            <div class="review-pending-panel__desc">{html.escape(resumo_texto)}.</div>
            <div class="review-pending-panel__metrics">{metricas_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if qtd_nao_encontrados > 0 and not revisao_aleatoria:
        st.markdown("**Fora da base**")
        for idx, nome in enumerate(diagnostico.get("nao_encontrados", [])):
            with st.expander(
                f"❓ Fora da base: {nome}",
                expanded=(qtd_nao_encontrados == 1 or (expandir_primeiro_faltante and idx == 0)),
            ):
                _render_pendencia_item_intro(
                    "fora_base",
                    nome,
                    detalhe="Ação principal: corrigir o nome na lista. Se o nome estiver certo, cadastre na base; se não entrar, remova da lista.",
                )
                with st.form(f"form_pendencia_nao_encontrado_{idx}"):
                    nome_corrigido = st.text_input(
                        "Nome corrigido",
                        value=nome,
                        key=f"pendencia_nome_corrigido_{idx}",
                    )
                    col_nf1, col_nf2, col_nf3 = st.columns(3)
                    aplicar_nome = col_nf1.form_submit_button("✅ Corrigir nome")
                    cadastrar_nome = col_nf2.form_submit_button("➕ Cadastrar")
                    remover_nome = col_nf3.form_submit_button("➖ Remover")

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
                        st.session_state[K.FALTANTES_REVISAO] = faltantes_priorizados
                        st.session_state[K.CADASTRO_GUIADO_ATIVO] = True
                        st.session_state[K.FALTANTES_CADASTRADOS_NA_RODADA] = []
                        st.session_state[K.REVISAO_PENDENTE_POS_CADASTRO] = False
                        st.session_state[K.LISTA_REVISADA_CONFIRMADA] = False
                        st.session_state[K.LISTA_REVISADA] = None
                        st.session_state[K.REVISAO_LISTA_EXPANDIDA] = True
                        st.session_state[K.SCROLL_PARA_REVISAO] = True
                        st.session_state[K.SCROLL_DESTINO_REVISAO] = "cadastro"
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
        st.markdown("**Duplicados na lista**")
        for idx, nome in enumerate(diagnostico.get("duplicados", [])):
            ocorrencias_duplicado = _ocorrencias_do_nome_duplicado_na_lista(
                lista_texto,
                diagnostico,
                nome,
                revisao_aleatoria=revisao_aleatoria,
            )
            with st.expander(
                f"🔁 Duplicado: {nome}",
                expanded=(qtd_duplicados == 1 or (expandir_primeiro_duplicado and idx == 0)),
            ):
                if revisao_aleatoria:
                    st.caption("Corrija cada ocorrência abaixo antes de confirmar a lista.")
                else:
                    st.caption("Edite as ocorrências abaixo. Se precisar, remova entradas indevidas.")

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
                            f"Ocorrência {ocorrencia_idx}",
                            value=valor_atual,
                            key=campo_key,
                        )
                        edicoes_por_linha[linha_idx] = nome_editado
                        if not revisao_aleatoria:
                            remover = st.checkbox(
                                f"Remover ocorrência {ocorrencia_idx}",
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
        st.markdown("**Bloqueios da base**")
        for idx, item in enumerate(diagnostico.get("nomes_bloqueados_base", [])):
            nome = item.get("nome", "")
            motivos = item.get("motivos", [])
            motivos_texto = "; ".join(str(m).strip() for m in motivos if str(m).strip()) or "Registro inconsistente na base."
            with st.expander(
                f"🛠️ Base bloqueada: {nome}",
                expanded=_expandir_bloqueio_base_padrao(
                    idx=idx,
                    nome=nome,
                    qtd_bloqueios_base=qtd_bloqueios_base,
                    expandir_primeiro_bloqueio=expandir_primeiro_bloqueio,
                ),
            ):
                _render_pendencia_item_intro(
                    "bloqueio_base",
                    nome,
                )
                if render_action_button(
                    "🛠️ Editar registro",
                    key=f"acao_principal_bloqueio_{idx}_{normalizar_nome_comparacao(nome)}",
                    role="primary",
                    use_primary_type=True,
                ):
                    st.session_state[K.REVISAO_FOCO_BLOQUEIO_NOME] = nome
                    st.session_state[K.REVISAO_LISTA_EXPANDIDA] = True
                    st.session_state[K.SCROLL_PARA_REVISAO] = True
                    st.session_state[K.SCROLL_DESTINO_REVISAO] = "pendencias"
                    st.rerun()

                st.markdown(f"**Motivos detectados:** {motivos_texto}")
                st.caption("Edite ou remova os registros aqui mesmo. Ao salvar, a revisão será atualizada.")
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


















    return total_pendencias


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
            st.session_state.pop(K.REVISAO_FOCO_BLOQUEIO_NOME, None)
        return

    nome_foco = st.session_state.get(K.REVISAO_FOCO_BLOQUEIO_NOME)
    nomes_bloqueados_ordenados = nomes_bloqueados_base.copy()
    if nome_foco:
        nomes_bloqueados_ordenados.sort(key=lambda item: 0 if item.get("nome") == nome_foco else 1)

    if show_intro:
        st.caption("Você pode corrigir esses registros sem sair da revisão.")

    for item in nomes_bloqueados_ordenados:
        nome = item["nome"]
        motivos = item.get("motivos", [])
        df_nome = st.session_state[K.DF_BASE].copy()
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
                    st.session_state[K.DF_BASE] = st.session_state[K.DF_BASE].drop(index=idx_original).reset_index(drop=True)
                    atualizar_integridade_base_no_estado(logic)
                    if lista_texto:
                        diagnosticar_lista_no_estado(logic, lista_texto)
                        st.session_state[K.REVISAO_LISTA_EXPANDIDA] = True
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
                        st.session_state[K.DF_BASE].loc[idx_original, "Nome"] = nome_corrigido
                        st.session_state[K.DF_BASE].loc[idx_original, "Posição"] = pos_corr
                        st.session_state[K.DF_BASE].loc[idx_original, "Nota"] = nota_corr
                        st.session_state[K.DF_BASE].loc[idx_original, "Velocidade"] = vel_corr
                        st.session_state[K.DF_BASE].loc[idx_original, "Movimentação"] = mov_corr
                        atualizar_integridade_base_no_estado(logic)
                        if lista_texto:
                            diagnosticar_lista_no_estado(logic, lista_texto)
                            st.session_state[K.REVISAO_LISTA_EXPANDIDA] = True
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
    bloqueios = listar_bloqueios_base_atual(st.session_state[K.DF_BASE])
    if not bloqueios:
        return

    with st.expander("🛠️ Corrigir base agora", expanded=False):
        render_correcao_inline_bloqueios_base(
            logic,
            st.session_state.get(K.LISTA_TEXTO_REVISADO, ""),
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
    diagnostico = st.session_state[K.DIAGNOSTICO_LISTA]
    pos_cadastro_pendente = (
        st.session_state[K.REVISAO_PENDENTE_POS_CADASTRO]
        and not st.session_state[K.CADASTRO_GUIADO_ATIVO]
        and len(st.session_state[K.FALTANTES_REVISAO]) == 0
        and len(st.session_state[K.FALTANTES_CADASTRADOS_NA_RODADA]) > 0
    )

    if not diagnostico and not pos_cadastro_pendente:
        return

    expanded = bool(
        st.session_state.get(K.REVIEW_STAGE_ACTIVE_UI, False)
        or st.session_state[K.REVISAO_LISTA_EXPANDIDA]
        or st.session_state.get(K.CADASTRO_GUIADO_ATIVO, False)
        or pos_cadastro_pendente
        or st.session_state.get(K.SCROLL_PARA_REVISAO, False)
    )

    with st.expander("🔎 Revisão da lista", expanded=expanded):
        if pos_cadastro_pendente and not diagnostico:
            qtd_cadastrados = len(st.session_state[K.FALTANTES_CADASTRADOS_NA_RODADA])
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
                st.session_state[K.REVISAO_PENDENTE_POS_CADASTRO] = False
                st.session_state[K.FALTANTES_CADASTRADOS_NA_RODADA] = []
                st.session_state[K.FALTANTES_REVISAO] = []
                st.session_state[K.SCROLL_PARA_REVISAO] = True
                st.session_state[K.SCROLL_DESTINO_REVISAO] = "confirmar"
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
            or st.session_state[K.CADASTRO_GUIADO_ATIVO]
            or pos_cadastro_pendente
        )

        if revisao_aleatoria:
            if st.session_state[K.LISTA_REVISADA_CONFIRMADA]:
                _render_revisao_status_banner("Lista confirmada para sorteio aleatório por lista.", tone="success")
            elif qtd_duplicados > 0:
                _render_revisao_status_banner("Há nomes repetidos. Corrija os itens abaixo antes de confirmar a lista.", tone="warning")
            else:
                _render_revisao_status_banner("Lista válida para sorteio aleatório por lista.", tone="success")
        elif diagnostico.get("tem_bloqueio_base", False):
            _render_revisao_status_banner("Há problemas na base atual que precisam ser corrigidos antes da confirmação.", tone="error")
        elif st.session_state[K.LISTA_REVISADA_CONFIRMADA]:
            _render_revisao_status_banner("Lista confirmada com sucesso. Agora você já pode sortear os times.", tone="success")
        elif tem_pendencia_revisao:
            _render_revisao_status_banner("A revisão encontrou pendências. Veja os detalhes abaixo antes de confirmar a lista.", tone="warning")
        else:
            _render_revisao_status_banner("A lista está pronta para confirmação.", tone="success")

        total_pendencias = render_revisao_pendencias_panel(
            logic,
            lista_texto,
            diagnostico,
            revisao_aleatoria=revisao_aleatoria,
            lista_input_key=lista_input_key,
            atualizar_integridade_base_no_estado=atualizar_integridade_base_no_estado,
            diagnosticar_lista_no_estado=diagnosticar_lista_no_estado,
            render_action_button=render_action_button,
        )

        if total_pendencias > 0:
            _render_resumo_revisao_visual(
                diagnostico["total_brutos"],
                diagnostico["total_validos"],
                qtd_correcoes,
                total_pendencias,
                compacto=True,
            )
        else:
            _render_resumo_revisao_visual(
                diagnostico["total_brutos"],
                diagnostico["total_validos"],
                qtd_correcoes,
                0,
                compacto=False,
            )

        lista_final_atual = diagnostico["lista_final_sugerida"]
        lista_final_texto = "\n".join([str(nome).strip() for nome in lista_final_atual if str(nome).strip()])
        _render_lista_final_preview(
            "Lista final sugerida" if not revisao_aleatoria else "Nomes únicos do sorteio",
            lista_final_atual,
        )

        pode_confirmar = (
            diagnostico["total_validos"] > 0
            and (revisao_aleatoria or not diagnostico["tem_nao_encontrados"])
            and qtd_duplicados == 0
            and not diagnostico.get("tem_bloqueio_base", False)
            and not st.session_state[K.CADASTRO_GUIADO_ATIVO]
        )

        if st.session_state[K.CADASTRO_GUIADO_ATIVO] and st.session_state[K.FALTANTES_REVISAO]:
            qtd_restantes = len(st.session_state[K.FALTANTES_REVISAO])
            render_step_cta_panel(
                "Continue o cadastro guiado dos faltantes",
                f"Ainda há {qtd_restantes} nome(s) pendente(s) nesta revisão. Conclua esse cadastro para liberar a confirmação da lista.",
                tone="warning",
                eyebrow="Etapa atual",
            )
        elif qtd_nao_encontrados > 0 and not revisao_aleatoria:
            render_step_cta_panel(
                "Cadastre os nomes faltantes para seguir",
                f"A revisão encontrou {qtd_nao_encontrados} nome(s) fora da base. Cadastre agora para seguir.",
                tone="warning",
                eyebrow="Etapa atual",
            )
            if render_action_button(
                "➕ Cadastrar faltantes agora",
                key="revisao_cadastrar_faltantes",
                role="primary",
                use_primary_type=True,
            ):
                st.session_state[K.FALTANTES_REVISAO] = diagnostico["nao_encontrados"].copy()
                st.session_state[K.CADASTRO_GUIADO_ATIVO] = True
                st.session_state[K.FALTANTES_CADASTRADOS_NA_RODADA] = []
                st.session_state[K.REVISAO_PENDENTE_POS_CADASTRO] = False
                st.session_state[K.LISTA_REVISADA_CONFIRMADA] = False
                st.session_state[K.LISTA_REVISADA] = None
                st.session_state[K.REVISAO_LISTA_EXPANDIDA] = True
                st.session_state[K.SCROLL_PARA_REVISAO] = True
                st.session_state[K.SCROLL_DESTINO_REVISAO] = "cadastro"
                st.rerun()
        elif qtd_duplicados > 0:
            render_step_cta_panel(
                "Corrija os nomes repetidos para seguir",
                "Corrija os nomes repetidos no painel de pendências antes de confirmar a lista.",
                tone="warning",
                eyebrow="Etapa atual",
            )
        elif diagnostico.get("tem_bloqueio_base", False):
            render_step_cta_panel(
                "Corrija a base atual antes da confirmação",
                "Existem inconsistências na base. Ajuste os registros bloqueados para seguir.",
                tone="warning",
                eyebrow="Etapa atual",
            )
        elif pode_confirmar and not st.session_state[K.LISTA_REVISADA_CONFIRMADA]:
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
                st.session_state[K.LISTA_REVISADA] = diagnostico["lista_final_sugerida"]
                st.session_state[K.LISTA_REVISADA_CONFIRMADA] = True
                st.session_state[K.REVISAO_LISTA_EXPANDIDA] = False
                st.session_state[K.SCROLL_PARA_SORTEIO] = True
                st.rerun()
        elif st.session_state[K.LISTA_REVISADA_CONFIRMADA]:
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

        if st.session_state[K.CADASTRO_GUIADO_ATIVO] and st.session_state[K.FALTANTES_REVISAO]:
            st.markdown('<div id="revisao-cadastro-anchor"></div>', unsafe_allow_html=True)
            faltantes_restantes = st.session_state[K.FALTANTES_REVISAO]
            faltantes_feitos = st.session_state[K.FALTANTES_CADASTRADOS_NA_RODADA]
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
                        st.session_state[K.DF_BASE].loc[len(st.session_state[K.DF_BASE])] = novo
                        st.session_state[K.FALTANTES_CADASTRADOS_NA_RODADA].append(novo_nome)
                        st.session_state[K.FALTANTES_REVISAO].pop(0)
                        st.session_state[K.LISTA_REVISADA_CONFIRMADA] = False
                        st.session_state[K.LISTA_REVISADA] = None
                        st.session_state[K.DIAGNOSTICO_LISTA] = None
                        st.session_state[K.REVISAO_LISTA_EXPANDIDA] = True

                        if not st.session_state[K.FALTANTES_REVISAO]:
                            st.session_state[K.CADASTRO_GUIADO_ATIVO] = False
                            st.session_state[K.FALTANTES_REVISAO] = []
                            st.session_state[K.REVISAO_PENDENTE_POS_CADASTRO] = True

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

