"""Fluxos específicos do cadastro guiado de faltantes."""

from __future__ import annotations

import streamlit as st

import state.keys as K
from core.validators import normalizar_nome_comparacao
from ui.review_text_ops import _atualizar_texto_lista_revisao


def _faltantes_unicos_do_diagnostico(diagnostico: dict | None) -> list[str]:
    faltantes: list[str] = []
    if not diagnostico:
        return faltantes

    for nome in diagnostico.get("nao_encontrados", []) or []:
        nome_limpo = str(nome).strip()
        if nome_limpo and nome_limpo not in faltantes:
            faltantes.append(nome_limpo)
    return faltantes


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
