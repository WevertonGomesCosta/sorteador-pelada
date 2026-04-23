"""Fluxos de revisão da lista e correções inline do Sorteador Pelada PRO."""

from __future__ import annotations

import html

import streamlit as st

import state.keys as K

from core.validators import (
    listar_bloqueios_base_atual,
    normalizar_nome_comparacao,
    registro_valido_para_sorteio,
    valor_slider_corrigir,
)
from ui.panels import render_step_cta_panel
from ui.review_helpers import (
    _aplicar_edicoes_em_ocorrencias_da_lista,
    _atualizar_texto_lista_revisao,
    _build_resumo_revisao_topo,
    _expandir_bloqueio_base_padrao,
    _faltantes_unicos_do_diagnostico,
    _ocorrencias_do_nome_duplicado_na_lista,
)
from ui.review_passive_components import (
    _render_lista_faltantes_identificados,
    _render_lista_final_preview,
    _render_pendencia_item_intro,
    _render_resumo_pre_sorteio_panel,
    _render_resumo_revisao_visual,
    _render_revisao_status_banner,
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
    resumo_topo = _build_resumo_revisao_topo(diagnostico)
    nome_foco = st.session_state.get(K.REVISAO_FOCO_BLOQUEIO_NOME)

    expandir_primeiro_bloqueio = qtd_bloqueios_base > 0
    expandir_primeiro_faltante = (not expandir_primeiro_bloqueio) and qtd_nao_encontrados > 0
    expandir_primeiro_duplicado = (not expandir_primeiro_bloqueio) and (not expandir_primeiro_faltante) and qtd_duplicados > 0

    st.markdown('<div id="revisao-pendencias-anchor"></div>', unsafe_allow_html=True)
    _render_resumo_pre_sorteio_panel(resumo_topo)

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
                    nome_foco=nome_foco,
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
                    st.session_state[K.SCROLL_ALVO_ID_REVISAO] = "revisao-pendencias-anchor"
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
                _render_pendencia_item_intro(
                    "duplicado_lista",
                    nome,
                    detalhe=detalhe_duplicado,
                )

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



def _sincronizar_fila_faltantes_com_diagnostico(diagnostico: dict | None) -> list[str]:
    faltantes_diagnostico = _faltantes_unicos_do_diagnostico(diagnostico)
    fila_atual = [str(nome).strip() for nome in st.session_state.get(K.FALTANTES_REVISAO, []) or [] if str(nome).strip()]

    faltantes_norm = {normalizar_nome_comparacao(nome): nome for nome in faltantes_diagnostico}
    fila_ordenada: list[str] = []
    vistos: set[str] = set()

    for nome in fila_atual:
        nome_norm = normalizar_nome_comparacao(nome)
        if nome_norm and nome_norm in faltantes_norm and nome_norm not in vistos:
            fila_ordenada.append(faltantes_norm[nome_norm])
            vistos.add(nome_norm)

    for nome in faltantes_diagnostico:
        nome_norm = normalizar_nome_comparacao(nome)
        if nome_norm and nome_norm not in vistos:
            fila_ordenada.append(nome)
            vistos.add(nome_norm)

    st.session_state[K.FALTANTES_REVISAO] = fila_ordenada
    st.session_state[K.CADASTRO_GUIADO_ATIVO] = bool(fila_ordenada)
    return fila_ordenada


def render_cadastro_guiado_dos_faltantes(
    logic,
    lista_texto: str,
    diagnostico: dict,
    *,
    lista_input_key: str,
    atualizar_integridade_base_no_estado,
    diagnosticar_lista_no_estado,
) -> None:
    fila_faltantes = _sincronizar_fila_faltantes_com_diagnostico(diagnostico)
    if not fila_faltantes:
        return

    faltantes_feitos = st.session_state.get(K.FALTANTES_CADASTRADOS_NA_RODADA, []) or []
    nome_atual = str(fila_faltantes[0]).strip()
    total_rodada = len(fila_faltantes) + len(faltantes_feitos)
    indice_atual = len(faltantes_feitos) + 1
    ultimo_da_fila = len(fila_faltantes) == 1

    nome_seed = normalizar_nome_comparacao(nome_atual) or f"faltante_{indice_atual}"
    widget_contexto_key = "cadastro_guiado_contexto_atual"
    form_key = "form_cadastro_guiado_direto_atual"
    nome_key = "cadastro_guiado_nome_atual"
    posicao_key = "cadastro_guiado_posicao_atual"
    nota_key = "cadastro_guiado_nota_atual"
    velocidade_key = "cadastro_guiado_velocidade_atual"
    movimentacao_key = "cadastro_guiado_movimentacao_atual"

    if st.session_state.get(widget_contexto_key) != nome_seed:
        st.session_state[widget_contexto_key] = nome_seed
        st.session_state[nome_key] = nome_atual
        st.session_state[posicao_key] = "M"
        st.session_state[nota_key] = 6
        st.session_state[velocidade_key] = 3
        st.session_state[movimentacao_key] = 3

    st.markdown('<div id="revisao-primeiro-faltante-anchor"></div>', unsafe_allow_html=True)
    st.markdown('<div id="revisao-cadastro-guiado-anchor"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Cadastro guiado dos faltantes**")
        st.caption("Ajuste o nome do atleta, preencha os parâmetros e siga para o próximo.")
        st.info(f"Jogador {indice_atual} de {total_rodada}: **{nome_atual}**")

        with st.form(form_key):
            st.text_input(
                "Nome do atleta",
                key=nome_key,
                help="Se o nome estiver diferente na lista, ajuste aqui antes de salvar.",
            )
            p_m = st.selectbox("Posição", ["M", "A", "D"], key=posicao_key)
            n_m = st.slider("Nota", 1, 10, 6, key=nota_key)
            v_m = st.slider("Velocidade", 1, 5, 3, key=velocidade_key)
            mv_m = st.slider("Movimentação", 1, 5, 3, key=movimentacao_key)
            col_salvar, col_remover = st.columns(2)
            label_submit = "Salvar e concluir" if ultimo_da_fila else "Salvar e próximo faltante"
            submit_guiado = col_salvar.form_submit_button(label_submit, use_container_width=True)
            remover_atleta = col_remover.form_submit_button("Remover", use_container_width=True)

        if submit_guiado:
            nome_digitado = str(st.session_state.get(nome_key, nome_atual)).strip()
            if hasattr(logic, "formatar_nome_visual") and nome_digitado:
                nome_digitado = logic.formatar_nome_visual(nome_digitado)
            if not nome_digitado:
                st.warning("Informe um nome válido para continuar.")
            else:
                nome_normalizado = normalizar_nome_comparacao(nome_digitado)
                base_existente = st.session_state[K.DF_BASE].copy()
                existe_na_base = False
                if not base_existente.empty and "Nome" in base_existente.columns:
                    existe_na_base = any(
                        normalizar_nome_comparacao(valor) == nome_normalizado
                        for valor in base_existente["Nome"].tolist()
                    )

                novo_texto_lista = lista_texto
                if nome_digitado != nome_atual:
                    novo_texto_lista, _ = _atualizar_texto_lista_revisao(
                        lista_texto,
                        nome_atual,
                        novo_nome=nome_digitado,
                    )
                    st.session_state[f"{lista_input_key}__pending"] = novo_texto_lista

                if existe_na_base:
                    diagnosticar_lista_no_estado(logic, novo_texto_lista)
                else:
                    novo = {
                        'Nome': nome_digitado,
                        'Nota': n_m,
                        'Posição': p_m,
                        'Velocidade': v_m,
                        'Movimentação': mv_m,
                    }
                    st.session_state[K.DF_BASE].loc[len(st.session_state[K.DF_BASE])] = novo
                    st.session_state[K.FALTANTES_CADASTRADOS_NA_RODADA].append(nome_digitado)

                st.session_state[K.LISTA_REVISADA_CONFIRMADA] = False
                st.session_state[K.LISTA_REVISADA] = None
                st.session_state[K.DIAGNOSTICO_LISTA] = None
                _sync_fluxo_faltantes_pos_cadastro(
                    logic,
                    novo_texto_lista,
                    atualizar_integridade_base_no_estado=atualizar_integridade_base_no_estado,
                    diagnosticar_lista_no_estado=diagnosticar_lista_no_estado,
                )
                st.rerun()

        if remover_atleta:
            novo_texto_lista, alterou = _atualizar_texto_lista_revisao(
                lista_texto,
                nome_atual,
                remover=True,
            )
            if alterou:
                st.session_state[f"{lista_input_key}__pending"] = novo_texto_lista
                st.session_state[K.LISTA_REVISADA_CONFIRMADA] = False
                st.session_state[K.LISTA_REVISADA] = None
                st.session_state[K.DIAGNOSTICO_LISTA] = None
                _sync_fluxo_faltantes_pos_cadastro(
                    logic,
                    novo_texto_lista,
                    atualizar_integridade_base_no_estado=atualizar_integridade_base_no_estado,
                    diagnosticar_lista_no_estado=diagnosticar_lista_no_estado,
                )
                st.rerun()
            else:
                st.warning("Não foi possível localizar esse nome na lista atual para removê-lo.")


def _sync_fluxo_faltantes_pos_cadastro(
    logic,
    lista_texto: str,
    *,
    atualizar_integridade_base_no_estado,
    diagnosticar_lista_no_estado,
) -> dict | None:
    if atualizar_integridade_base_no_estado is not None:
        atualizar_integridade_base_no_estado(logic)

    diagnostico_atualizado = diagnosticar_lista_no_estado(logic, lista_texto)

    faltantes_restantes = _sincronizar_fila_faltantes_com_diagnostico(diagnostico_atualizado)
    st.session_state[K.REVISAO_PENDENTE_POS_CADASTRO] = False
    st.session_state[K.REVISAO_LISTA_EXPANDIDA] = True

    if faltantes_restantes:
        st.session_state[K.SCROLL_PARA_REVISAO] = False
        st.session_state[K.SCROLL_DESTINO_REVISAO] = "top"
        st.session_state[K.SCROLL_ALVO_ID_REVISAO] = ""
    else:
        st.session_state[K.SCROLL_PARA_REVISAO] = True
        st.session_state[K.SCROLL_DESTINO_REVISAO] = "confirmar"
        st.session_state[K.SCROLL_ALVO_ID_REVISAO] = "revisao-confirmar-anchor"

    return diagnostico_atualizado




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
                st.session_state[K.SCROLL_ALVO_ID_REVISAO] = "revisao-confirmar-anchor"
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

        if (
            not revisao_aleatoria
            and st.session_state.get(K.CADASTRO_GUIADO_ATIVO)
            and len(st.session_state.get(K.FALTANTES_REVISAO, [])) > 0
        ):
            render_cadastro_guiado_dos_faltantes(
                logic,
                lista_texto,
                diagnostico,
                lista_input_key=lista_input_key,
                atualizar_integridade_base_no_estado=atualizar_integridade_base_no_estado,
                diagnosticar_lista_no_estado=diagnosticar_lista_no_estado,
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
            faltantes_para_listar = _faltantes_unicos_do_diagnostico(diagnostico)
            _render_lista_faltantes_identificados(faltantes_para_listar)
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
                st.session_state[K.SCROLL_DESTINO_REVISAO] = "pendencias"
                st.session_state[K.SCROLL_ALVO_ID_REVISAO] = "revisao-cadastro-guiado-anchor"
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

        _render_lista_final_preview(
            "Lista final sugerida" if not revisao_aleatoria else "Nomes únicos do sorteio",
            lista_final_atual,
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

            aplicar_edicao = st.button(
                "✅ Aplicar alterações e revisar novamente",
                key="acao_aplicar_edicao_lista_revisada",
                type="primary",
                use_container_width=True,
            )

            if aplicar_edicao:
                novo_texto_lista = str(st.session_state.get(edicao_key, "")).strip()
                st.session_state[f"{lista_input_key}__pending"] = novo_texto_lista
                st.session_state[f"{lista_input_key}__revisar"] = True
                st.rerun()

        if diagnostico["ignorados"]:
            with st.expander(f"ℹ️ Itens ignorados na leitura ({len(diagnostico['ignorados'])})", expanded=False):
                for item in diagnostico["ignorados"]:
                    st.markdown(f"- {item}")

