"""Orquestração principal da revisão da lista."""

from __future__ import annotations

import streamlit as st

import state.keys as K
from ui.panels import render_step_cta_panel
from ui.review_cadastro import (
    _faltantes_unicos_do_diagnostico,
    render_cadastro_guiado_dos_faltantes,
)
from ui.review_components import (
    _render_lista_final_preview,
    _render_resumo_revisao_visual,
    _render_revisao_status_banner,
)
from ui.review_pending import render_revisao_pendencias_panel_impl


def render_revisao_lista_impl(
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

        total_pendencias = render_revisao_pendencias_panel_impl(
            logic,
            lista_texto,
            diagnostico,
            revisao_aleatoria=revisao_aleatoria,
            lista_input_key=lista_input_key,
            atualizar_integridade_base_no_estado=atualizar_integridade_base_no_estado,
            diagnosticar_lista_no_estado=diagnosticar_lista_no_estado,
            render_action_button=render_action_button,
        )

        if qtd_nao_encontrados > 0 and not revisao_aleatoria:
            render_step_cta_panel(
                "Cadastre os nomes faltantes para seguir",
                f"A revisão encontrou {qtd_nao_encontrados} nome(s) fora da base. Cadastre agora para seguir.",
                tone="warning",
                eyebrow="Etapa atual",
            )
            faltantes_para_listar = _faltantes_unicos_do_diagnostico(diagnostico)
            if faltantes_para_listar:
                st.markdown("**Atletas faltantes identificados nesta revisão**")
                for nome_faltante in faltantes_para_listar:
                    st.markdown(f"- `{nome_faltante}`")
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

        lista_final_atual = diagnostico["lista_final_sugerida"]
        lista_final_texto = "\n".join([str(nome).strip() for nome in lista_final_atual if str(nome).strip()])

        pode_confirmar = (
            diagnostico["total_validos"] > 0
            and (revisao_aleatoria or not diagnostico["tem_nao_encontrados"])
            and qtd_duplicados == 0
            and not diagnostico.get("tem_bloqueio_base", False)
            and not st.session_state[K.CADASTRO_GUIADO_ATIVO]
        )

        _render_lista_final_preview("Lista final sugerida", lista_final_atual)

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

        st.markdown('<div id="revisao-confirmar-anchor"></div>', unsafe_allow_html=True)
        if pode_confirmar:
            if render_action_button("✅ Confirmar lista final", key="confirmar_lista", role="primary", use_primary_type=True):
                st.session_state[K.LISTA_REVISADA] = lista_final_texto
                st.session_state[K.LISTA_REVISADA_CONFIRMADA] = True
                st.session_state[K.REVISAO_LISTA_EXPANDIDA] = True
                st.success("Lista final confirmada. Agora você já pode sortear os times.")
                st.rerun()
        else:
            motivos = []
            if st.session_state[K.CADASTRO_GUIADO_ATIVO]:
                motivos.append("Conclua o cadastro guiado dos faltantes")
            if diagnostico.get("tem_bloqueio_base", False):
                motivos.append("Corrija os bloqueios da base")
            if qtd_nao_encontrados > 0 and not revisao_aleatoria:
                motivos.append("Cadastre os nomes que ainda estão fora da base")
            if qtd_duplicados > 0:
                motivos.append("Revise os nomes duplicados")
            if diagnostico["total_validos"] == 0:
                motivos.append("A revisão precisa encontrar pelo menos um nome válido")

            mensagem = "; ".join(motivos) if motivos else "Revise os itens pendentes antes de confirmar."
            st.caption(f"🔒 Confirmação indisponível: {mensagem}.")
