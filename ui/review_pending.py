"""Subblocos de pendências e correções inline da revisão."""

from __future__ import annotations

import html

import streamlit as st

import state.keys as K
from core.validators import normalizar_nome_comparacao, registro_valido_para_sorteio, valor_slider_corrigir
from ui.review_components import _build_resumo_revisao_topo, _expandir_bloqueio_base_padrao, _render_pendencia_item_intro
from ui.review_text_ops import _aplicar_edicoes_em_ocorrencias_da_lista, _ocorrencias_do_nome_duplicado_na_lista


def render_revisao_pendencias_panel_impl(
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
    resumo_topo = _build_resumo_revisao_topo(diagnostico)

    metricas = [
        ("Bloqueios", str(resumo_topo["qtd_bloqueios"])),
        ("Fora da base", str(resumo_topo["qtd_fora_base"])),
        ("Duplicados", str(resumo_topo["qtd_duplicados"])),
        ("Aptos", str(resumo_topo["qtd_aptos"])),
    ]
    metricas_html = "".join(
        f'<div class="review-pending-panel__metric">'
        f'<div class="review-pending-panel__metric-label">{html.escape(rotulo)}</div>'
        f'<div class="review-pending-panel__metric-value">{html.escape(valor)}</div>'
        f'</div>'
        for rotulo, valor in metricas
    )

    expandir_primeiro_bloqueio = qtd_bloqueios_base > 0
    expandir_primeiro_faltante = (not expandir_primeiro_bloqueio) and qtd_nao_encontrados > 0
    expandir_primeiro_duplicado = (not expandir_primeiro_bloqueio) and (not expandir_primeiro_faltante) and qtd_duplicados > 0

    st.markdown('<div id="revisao-pendencias-anchor"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="review-pending-panel">
            <div class="review-pending-panel__eyebrow">Resumo pré-sorteio</div>
            <div class="review-pending-panel__title">{html.escape(str(resumo_topo['status_label']))}</div>
            <div class="review-pending-panel__desc">{html.escape(str(resumo_topo['acao_contextual']))}</div>
            <div class="review-pending-panel__metrics">{metricas_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if total_pendencias == 0:
        return 0

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
                _render_pendencia_item_intro("bloqueio_base", nome)
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
                    st.session_state[K.SCROLL_ALVO_ID_REVISAO] = "revisao-pendencias-anchor"
                    st.rerun()

                st.markdown(f"**Motivos detectados:** {motivos_texto}")
                st.caption("Edite ou remova os registros aqui mesmo. Ao salvar, a revisão será atualizada.")
                render_correcao_inline_bloqueios_base_impl(
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

    if qtd_nao_encontrados > 0 and not revisao_aleatoria:
        st.caption("Os nomes fora da base são tratados no bloco Cadastro guiado dos faltantes logo abaixo deste resumo.")

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
                detalhe_duplicado = (
                    "Ação principal: corrigir os nomes das ocorrências. "
                    "No sorteio aleatório, preencha todas; fora dele, você também pode remover ocorrências indevidas."
                    if revisao_aleatoria
                    else
                    "Ação principal: revisar as ocorrências e corrigir os nomes. "
                    "Se necessário, remova apenas as entradas indevidas."
                )
                _render_pendencia_item_intro("duplicado_lista", nome, detalhe=detalhe_duplicado)

                if ocorrencias_duplicado:
                    st.markdown(
                        "**Ocorrências detectadas na lista:** "
                        + ", ".join(f"`{html.escape(item['valor'])}`" for item in ocorrencias_duplicado)
                    )
                else:
                    st.warning("Não foi possível localizar as ocorrências atuais desse nome na lista.")
                    continue

                with st.form(f"form_pendencia_duplicado_{idx}"):
                    edicoes_por_linha: dict[int, str] = {}
                    remover_linhas: set[int] = set()

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

                    aplicar_label = "✅ Corrigir ocorrências" if revisao_aleatoria else "✅ Aplicar correções"
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
                            else:
                                st.warning("Faça pelo menos uma correção de nome para reaplicar a revisão.")

    return total_pendencias


def render_correcao_inline_bloqueios_base_impl(
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
        df_nome = df_nome.sort_values(by=["_registro_valido", "_orig_index"], ascending=[True, True]).reset_index(drop=True)

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
                    nome_corr = st.text_input("Nome", value=nome_atual, key=f"corrigir_nome_{nome}_{idx_original}")
                    posicao_atual = str(row.get("Posição", "")).strip().upper()
                    if posicao_atual not in ["D", "M", "A"]:
                        posicao_atual = "M"

                    pos_corr = st.selectbox(
                        "Posição",
                        ["D", "M", "A"],
                        index=["D", "M", "A"].index(posicao_atual),
                        key=f"corrigir_posicao_{nome}_{idx_original}",
                    )
                    nota_corr = st.slider("Nota", 1, 10, valor_slider_corrigir(row.get("Nota"), 1, 10, 6), key=f"corrigir_nota_{nome}_{idx_original}")
                    vel_corr = st.slider("Velocidade", 1, 5, valor_slider_corrigir(row.get("Velocidade"), 1, 5, 3), key=f"corrigir_velocidade_{nome}_{idx_original}")
                    mov_corr = st.slider("Movimentação", 1, 5, valor_slider_corrigir(row.get("Movimentação"), 1, 5, 3), key=f"corrigir_movimentacao_{nome}_{idx_original}")
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


def render_correcao_inline_etapa2_impl(
    logic,
    *,
    render_correcao_inline_bloqueios_base,
    atualizar_integridade_base_no_estado,
    diagnosticar_lista_no_estado,
    render_action_button,
):
    from core.validators import listar_bloqueios_base_atual

    bloqueios = listar_bloqueios_base_atual(st.session_state[K.DF_BASE])
    if not bloqueios:
        return

    with st.expander("🛠️ Corrigir base agora", expanded=False):
        render_correcao_inline_bloqueios_base_impl(
            logic,
            st.session_state.get(K.LISTA_TEXTO_REVISADO, ""),
            bloqueios,
            atualizar_integridade_base_no_estado=atualizar_integridade_base_no_estado,
            diagnosticar_lista_no_estado=diagnosticar_lista_no_estado,
            render_action_button=render_action_button,
        )
