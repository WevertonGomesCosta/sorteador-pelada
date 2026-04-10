"""App principal do Sorteador Pelada PRO.

Arquivo organizado por blocos funcionais para facilitar manutenção, auditoria
e refatorações futuras da camada visual.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
from datetime import datetime

import state.keys as K

from core.logic import PeladaLogic
from core.flow_guard import (
    construir_assinatura_entrada_sorteio,
    construir_gate_pre_sorteio,
    contar_duplicados_base_atual,
    extrair_nomes_unicos_da_lista,
    invalidar_resultado_se_entrada_mudou,
    sortear_times_aleatorios_por_lista,
)
from state.session import (
    atualizar_integridade_base_no_estado,
    diagnosticar_lista_no_estado,
    init_session_state,
    limpar_estado_revisao_lista,
    registrar_base_carregada_no_estado,
)
from ui.components import botao_instalar_app
from ui.manual_card import render_manual_card
from ui.result_view import (
    construir_cabecalho_padronizado_sorteio,
    construir_texto_compartilhamento_resultado,
    render_acoes_resultado,
    render_result_summary_panel,
    render_sort_ready_panel,
    render_team_cards,
)
from ui.actions import render_action_button
from ui.panels import render_step_cta_panel
from ui.primitives import render_app_meta_footer, render_inline_status_note, render_section_header
from ui.styles import apply_app_styles
from core.validators import (
    diagnosticar_nomes_bloqueados_para_sorteio,
    listar_bloqueios_base_atual,
    normalizar_nome_comparacao,
    preparar_df_sorteio,
    registro_valido_para_sorteio,
    valor_slider_corrigir,
)
from ui.base_view import (
    formatar_df_visual_numeros_inteiros,
    render_base_inconsistencias_expander,
    render_base_integrity_alert,
    render_base_preview,
    render_base_summary,
)
from ui.review_view import (
    render_correcao_inline_bloqueios_base,
    render_revisao_lista,
)
from state.ui_state import abrir_expander_cadastro_manual, ensure_local_session_state
from state.view_models import (
    construir_estado_blocos_visuais,
    construir_status_sessao_visual,
    determinar_etapa_visual_ativa,
    determinar_visibilidade_revisao,
)
from state.criteria_state import obter_criterios_ativos, resumo_criterios_ativos
from ui.group_config_view import render_group_config_expander
from ui.summary_strings import (
    resumo_expander_cadastro_manual,
    resumo_expander_configuracao,
    resumo_expander_criterios,
)
from ui.panels import render_session_status_panel
from ui.pre_sort_view import render_resumo_operacional_pre_sorteio

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Sorteador Pelada PRO",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- SEGREDOS (VIA ST.SECRETS) ---
try:
    NOME_PELADA_ADM = st.secrets["nome_admin"]
    SENHA_ADM = st.secrets["senha_admin"]
except Exception:
    NOME_PELADA_ADM = "QUARTA 18:30"
    SENHA_ADM = "1234"

# --- CSS ---
apply_app_styles()


def main():
    logic = PeladaLogic()
    st.title("⚽ Sorteador Pelada PRO")
    botao_instalar_app()

    init_session_state(logic)
    ensure_local_session_state()

    if K.LISTA_TEXTO_INPUT not in st.session_state:
        st.session_state[K.LISTA_TEXTO_INPUT] = ""
    draft_lista_key = K.LISTA_TEXTO_INPUT_DRAFT
    if draft_lista_key not in st.session_state:
        st.session_state[draft_lista_key] = st.session_state[K.LISTA_TEXTO_INPUT]
    pending_lista_key = K.LISTA_TEXTO_INPUT_PENDING
    if pending_lista_key in st.session_state:
        novo_texto_lista = st.session_state.pop(pending_lista_key)
        st.session_state[K.LISTA_TEXTO_INPUT] = novo_texto_lista
        st.session_state[draft_lista_key] = novo_texto_lista

    origem_fluxo_status = st.session_state.get(K.GRUPO_ORIGEM_FLUXO)
    base_carregada_via_secao1 = bool(
        st.session_state.get(K.BASE_ADMIN_CARREGADA, False)
        or st.session_state.get(K.ULTIMO_ARQUIVO)
    )

    processamento_status = logic.processar_lista(
        st.session_state.get(K.LISTA_TEXTO_INPUT, ""),
        return_metadata=True,
        emit_warning=False,
    )
    qtd_nomes_status = len(processamento_status.get("jogadores", []))
    qtd_ignorados_status = len(processamento_status.get("ignorados", []))
    diagnostico_status = st.session_state.get(K.DIAGNOSTICO_LISTA) or {}
    lista_revisada_ok_status = bool(st.session_state.get(K.DIAGNOSTICO_LISTA) is not None)
    lista_confirmada_ok_status = bool(
        st.session_state.get(K.LISTA_REVISADA_CONFIRMADA)
        and st.session_state.get(K.LISTA_REVISADA)
    )
    base_pronta_ok_status = bool(
        not st.session_state[K.DF_BASE].empty or st.session_state[K.NOVOS_JOGADORES]
    )
    resultado_disponivel_status = bool(st.session_state.get(K.RESULTADO))

    status_sessao_visual = construir_status_sessao_visual(
        origem_fluxo=origem_fluxo_status,
        base_carregada_via_secao1=base_carregada_via_secao1,
        qtd_nomes=qtd_nomes_status,
        qtd_ignorados=qtd_ignorados_status,
        diagnostico=diagnostico_status,
        lista_revisada_ok=lista_revisada_ok_status,
        lista_confirmada_ok=lista_confirmada_ok_status,
        base_pronta_ok=base_pronta_ok_status,
        resultado_disponivel=resultado_disponivel_status,
        cadastro_guiado_ativo=bool(st.session_state.get(K.CADASTRO_GUIADO_ATIVO, False)),
        is_admin=bool(st.session_state.get(K.IS_ADMIN, False)),
        ultimo_arquivo=str(st.session_state.get(K.ULTIMO_ARQUIVO, "")),
        df_base_len=len(st.session_state[K.DF_BASE]),
        novos_jogadores_len=len(st.session_state[K.NOVOS_JOGADORES]),
    )
    fluxo_somente_lista = bool(status_sessao_visual["fluxo_somente_lista"])
    escolha_inicial_pendente_status = bool(status_sessao_visual["escolha_inicial_pendente"])

    render_session_status_panel(
        modo_atual=str(status_sessao_visual["modo_atual"]),
        base_status=str(status_sessao_visual["base_status"]),
        lista_status=str(status_sessao_visual["lista_status"]),
        fluxo_status=str(status_sessao_visual["fluxo_status"]),
        proxima_acao=str(status_sessao_visual["proxima_acao"]),
    )

    render_section_header(
        "1. Configuração do grupo e base de dados",
        "Escolha como deseja começar: sorteio apenas com lista, base do grupo ou Excel próprio."
    )
    nome_pelada = render_group_config_expander(logic, NOME_PELADA_ADM, SENHA_ADM)

    if base_carregada_via_secao1:
        render_section_header(
            "2. Base de jogadores",
            "Confira a base atual carregada a partir da etapa 1."
        )
        render_base_summary()
        render_base_integrity_alert()
        render_base_inconsistencias_expander(
            logic,
            atualizar_integridade_base_no_estado=atualizar_integridade_base_no_estado,
            diagnosticar_lista_no_estado=diagnosticar_lista_no_estado,
            render_action_button=render_action_button,
        )
        titulo_secao_manual = "3. Adicionar jogadores manualmente"
        titulo_secao_lista = "4. Lista da pelada"
        subtitulo_lista = "Cole aqui os nomes confirmados para o sorteio. Eles serão comparados com a base carregada e, se necessário, você poderá completar os jogadores manualmente."
    elif fluxo_somente_lista:
        titulo_secao_manual = None
        titulo_secao_lista = "2. Lista da pelada"
        subtitulo_lista = "Cole aqui os nomes confirmados para o sorteio. Neste modo, o app fará um sorteio aleatório apenas entre os nomes únicos da lista."
    else:
        titulo_secao_manual = "2. Adicionar jogadores manualmente"
        titulo_secao_lista = "3. Lista da pelada"
        subtitulo_lista = "Cole aqui os nomes confirmados para o sorteio. Sem base carregada, o app poderá sortear aleatoriamente entre os nomes únicos da lista ou você pode montar sua base manualmente."

    faltantes_identificados = bool(
        len(st.session_state.get(K.FALTANTES_REVISAO, [])) > 0
        or len((st.session_state.get(K.DIAGNOSTICO_LISTA) or {}).get("nao_encontrados", [])) > 0
    )
    manual_section_visible = bool(
        st.session_state.get(K.MANUAL_SECTION_VISIBLE, False)
        or st.session_state.get(K.CADASTRO_MANUAL_EXPANDED, False)
        or st.session_state.get(K.CADASTRO_GUIADO_ATIVO, False)
        or st.session_state.get(K.REVISAO_PENDENTE_POS_CADASTRO, False)
        or len(st.session_state.get(K.FALTANTES_REVISAO, [])) > 0
        or len(st.session_state.get(K.FALTANTES_CADASTRADOS_NA_RODADA, [])) > 0
        or faltantes_identificados
    )
    st.session_state[K.MANUAL_SECTION_VISIBLE] = manual_section_visible

    if titulo_secao_manual is not None:
        if not manual_section_visible:
            render_section_header(
                titulo_secao_manual,
                "Adicione jogadores manualmente apenas se precisar complementar a base ou montar sua base do zero."
            )
            st.caption("Esta etapa é opcional e só precisa ser usada quando você quiser complementar a base ou cadastrar faltantes.")
            if render_action_button(
                "➕ Adicionar jogadores manualmente",
                key="abrir_secao_cadastro_manual",
                role="secondary",
            ):
                st.session_state[K.MANUAL_SECTION_VISIBLE] = True
                st.session_state[K.CADASTRO_MANUAL_EXPANDED] = True
                st.rerun()
        else:
            render_section_header(
                titulo_secao_manual,
                "Use esta etapa para montar sua base do zero ou complementar a base atual com novos jogadores."
            )
            if faltantes_identificados and not st.session_state.get(K.CADASTRO_GUIADO_ATIVO, False):
                st.info("Há nomes faltantes identificados na revisão. Você pode cadastrá-los agora para continuar o fluxo com a base atual.")
            render_manual_card(
                logic,
                nome_pelada,
                on_open_expander=abrir_expander_cadastro_manual,
                render_inline_correction=lambda logic_ref, lista_texto, nomes_bloqueados_base: render_correcao_inline_bloqueios_base(
                    logic_ref,
                    lista_texto,
                    nomes_bloqueados_base,
                    atualizar_integridade_base_no_estado=atualizar_integridade_base_no_estado,
                    diagnosticar_lista_no_estado=diagnosticar_lista_no_estado,
                    render_action_button=render_action_button,
                ),
            )

            render_base_preview()

    st.markdown('<div id="lista-anchor"></div>', unsafe_allow_html=True)
    render_section_header(
        titulo_secao_lista,
        subtitulo_lista,
    )
    if st.session_state.get(K.SCROLL_PARA_LISTA, False):
        components.html(
            '''
            <script>
            const parentDoc = window.parent.document;
            const anchor = parentDoc.getElementById("lista-anchor");
            if (anchor) {
                anchor.scrollIntoView({ behavior: "smooth", block: "start" });
            }
            </script>
            ''',
            height=0,
        )
        st.session_state[K.SCROLL_PARA_LISTA] = False
    st.markdown(f"**Modo:** {'🗂️ Base do grupo' if st.session_state[K.IS_ADMIN] else '👤 Público (Base própria)'}")

    auto_revisar_lista = bool(st.session_state.pop(K.LISTA_TEXTO_INPUT_REVISAR, False))
    revisao_pendente_pos_cadastro = bool(st.session_state.get(K.REVISAO_PENDENTE_POS_CADASTRO, False))
    mostrar_botao_revisao_principal = bool(
        not st.session_state.get(K.LISTA_REVISADA_CONFIRMADA, False)
        and not st.session_state.get(K.CADASTRO_GUIADO_ATIVO, False)
        and not revisao_pendente_pos_cadastro
    )

    lista_rascunho = st.text_area(
        "Cole a lista (Numerada ou não):",
        height=120,
        placeholder="1. Jogador A\n2. Jogador B...",
        key=draft_lista_key,
    )
    lista_texto = st.session_state.get(K.LISTA_TEXTO_INPUT, "")
    if mostrar_botao_revisao_principal:
        revisar_lista = st.button(
            "🔎 Revisar lista",
        )
        if revisar_lista:
            st.session_state[K.LISTA_TEXTO_INPUT] = lista_rascunho
            lista_texto = lista_rascunho
    else:
        revisar_lista = False

    processamento_previa = logic.processar_lista(
        lista_texto,
        return_metadata=True,
        emit_warning=False,
    )
    qtd_nomes_informados = len(processamento_previa["jogadores"])
    qtd_itens_ignorados = len(processamento_previa.get("ignorados", []))
    lista_alterada_pos_revisao = (
        bool(st.session_state[K.LISTA_TEXTO_REVISADO])
        and lista_texto != st.session_state[K.LISTA_TEXTO_REVISADO]
    )
    precisa_revisar_lista = bool(
        qtd_nomes_informados > 0
        and (
            st.session_state[K.DIAGNOSTICO_LISTA] is None
            or lista_alterada_pos_revisao
        )
    )

    if qtd_nomes_informados > 0 or qtd_itens_ignorados > 0:
        if qtd_itens_ignorados == 0:
            st.caption(
                f"Leitura atual: {qtd_nomes_informados} nomes reconhecidos · nenhuma linha ignorada."
            )
        else:
            st.caption(
                f"Leitura atual: {qtd_nomes_informados} nomes reconhecidos · {qtd_itens_ignorados} linhas ignoradas."
            )

    if qtd_nomes_informados == 0:
        render_inline_status_note(
            "Lista vazia.",
            "Cole os nomes para começar.",
            tone="info",
        )
    elif lista_alterada_pos_revisao:
        render_inline_status_note(
            "Lista alterada após a revisão.",
            "A revisão anterior foi invalidada e precisa ser refeita.",
            tone="warning",
        )
    elif precisa_revisar_lista:
        render_inline_status_note(
            "Lista pronta.",
            "Os nomes já podem ser conferidos na revisão.",
            tone="info",
        )
    elif st.session_state.get(K.LISTA_REVISADA_CONFIRMADA) and st.session_state.get(K.LISTA_REVISADA):
        render_inline_status_note(
            "Lista final confirmada.",
            "A etapa da lista já está concluída.",
            tone="success",
        )
    elif st.session_state.get(K.DIAGNOSTICO_LISTA) is not None:
        render_inline_status_note(
            "Lista revisada.",
            "Só falta concluir a confirmação final.",
            tone="success",
        )

    col1, col2 = st.columns(2)
    n_times = col1.selectbox("Nº Times:", range(2, 11), index=1)

    invalidar_resultado_se_entrada_mudou(lista_texto, n_times)

    if qtd_nomes_informados > 0:
        base_por_time = qtd_nomes_informados // n_times
        resto_times = qtd_nomes_informados % n_times

        if resto_times == 0:
            st.caption(
                f"Distribuição estimada: {n_times} time(s) · {base_por_time} por time."
            )
        else:
            qtd_times_menores = n_times - resto_times
            qtd_times_maiores = resto_times
            st.caption(
                f"Prévia: {qtd_nomes_informados} nome(s) lido(s) da lista · {n_times} time(s) · "
                f"{qtd_times_menores} time(s) com {base_por_time} e {qtd_times_maiores} time(s) com {base_por_time + 1}."
            )

        if qtd_nomes_informados < n_times * 2:
            st.warning(
                "Atenção: há poucos nomes na lista para a quantidade de times escolhida. O sorteio pode ficar pouco equilibrado."
            )

    if lista_alterada_pos_revisao and (
        st.session_state[K.DIAGNOSTICO_LISTA] is not None
        or st.session_state[K.LISTA_REVISADA_CONFIRMADA]
    ):
        limpar_estado_revisao_lista()
        st.info("A lista foi alterada após a última revisão. Revise novamente antes de sortear.")

    base_pronta_ok = bool(
        not st.session_state[K.DF_BASE].empty or st.session_state[K.NOVOS_JOGADORES]
    )

    if revisar_lista or auto_revisar_lista:
        diagnostico = diagnosticar_lista_no_estado(logic, lista_texto)
        if diagnostico is None:
            st.session_state[K.SCROLL_PARA_REVISAO] = False
            st.session_state[K.SCROLL_DESTINO_REVISAO] = "top"
            st.warning("Cole uma lista de jogadores para revisar antes do sorteio.")
        else:
            modo_revisao = diagnostico.get("modo_revisao", "balanceado")
            revisao_aleatoria = modo_revisao == "aleatorio_lista"
            tem_pendencias_revisao = (
                len(diagnostico.get("nao_encontrados", [])) > 0
                or len(diagnostico.get("duplicados", [])) > 0
                or diagnostico.get("tem_bloqueio_base", False)
            )
            pode_ir_direto_para_confirmacao = (
                base_pronta_ok
                and diagnostico["total_validos"] > 0
                and (revisao_aleatoria or not diagnostico["tem_nao_encontrados"])
                and not diagnostico.get("tem_bloqueio_base", False)
                and not st.session_state.get(K.CADASTRO_GUIADO_ATIVO, False)
                and not tem_pendencias_revisao
            )
            st.session_state[K.SCROLL_PARA_REVISAO] = True
            if tem_pendencias_revisao:
                st.session_state[K.SCROLL_DESTINO_REVISAO] = "pendencias"
            else:
                st.session_state[K.SCROLL_DESTINO_REVISAO] = (
                    "confirmar" if pode_ir_direto_para_confirmacao else "top"
                )

    review_stage_visible = determinar_visibilidade_revisao(
        diagnostico_disponivel=st.session_state[K.DIAGNOSTICO_LISTA] is not None,
        lista_confirmada=bool(st.session_state[K.LISTA_REVISADA_CONFIRMADA]),
        cadastro_guiado_ativo=bool(st.session_state[K.CADASTRO_GUIADO_ATIVO]),
        revisao_pendente_pos_cadastro=bool(st.session_state[K.REVISAO_PENDENTE_POS_CADASTRO]),
        faltantes_revisao_qtd=len(st.session_state.get(K.FALTANTES_REVISAO, [])),
        faltantes_cadastrados_qtd=len(st.session_state.get(K.FALTANTES_CADASTRADOS_NA_RODADA, [])),
    )

    etapa_visual_ativa = determinar_etapa_visual_ativa(
        escolha_inicial_pendente=escolha_inicial_pendente_status,
        qtd_nomes=qtd_nomes_informados,
        draft_lista=st.session_state.get(draft_lista_key, ""),
        lista_confirmada=bool(st.session_state.get(K.LISTA_REVISADA_CONFIRMADA) and st.session_state.get(K.LISTA_REVISADA)),
        resultado_disponivel=bool(st.session_state.get(K.RESULTADO)),
        review_stage_visible=review_stage_visible,
        manual_section_visible=manual_section_visible,
        cadastro_guiado_ativo=bool(st.session_state.get(K.CADASTRO_GUIADO_ATIVO, False)),
        revisao_pendente_pos_cadastro=bool(st.session_state.get(K.REVISAO_PENDENTE_POS_CADASTRO, False)),
    )

    estado_blocos_visuais = construir_estado_blocos_visuais(
        etapa_visual_ativa=etapa_visual_ativa,
        scroll_para_revisao=bool(st.session_state.get(K.SCROLL_PARA_REVISAO, False)),
        cadastro_guiado_ativo=bool(st.session_state.get(K.CADASTRO_GUIADO_ATIVO, False)),
        revisao_pendente_pos_cadastro=bool(st.session_state.get(K.REVISAO_PENDENTE_POS_CADASTRO, False)),
        cadastro_manual_nome_existente=str(st.session_state.get(K.CADASTRO_MANUAL_NOME_EXISTENTE, "")),
    )

    st.session_state[K.GRUPO_CONFIG_EXPANDED] = bool(estado_blocos_visuais["grupo_config_expanded"])
    st.session_state[K.CADASTRO_MANUAL_EXPANDED] = bool(estado_blocos_visuais["cadastro_manual_expanded"])
    st.session_state[K.REVIEW_STAGE_ACTIVE_UI] = bool(estado_blocos_visuais["review_stage_active_ui"])
    if bool(estado_blocos_visuais["atualiza_revisao_lista_expandida"]):
        st.session_state[K.REVISAO_LISTA_EXPANDIDA] = bool(estado_blocos_visuais["revisao_lista_expandida"])

    if not review_stage_visible:
        st.caption("Depois de revisar a lista, as etapas de revisão, critérios e sorteio serão exibidas em sequência.")

    lista_revisada_ok = bool(st.session_state[K.DIAGNOSTICO_LISTA] is not None)
    lista_confirmada_ok = bool(
        st.session_state[K.LISTA_REVISADA_CONFIRMADA] and st.session_state[K.LISTA_REVISADA]
    )

    if review_stage_visible:
        review_section_num = int(titulo_secao_lista.split('.')[0]) + 1
        st.markdown('<div id="revisao-anchor"></div>', unsafe_allow_html=True)
        render_section_header(
            f"{review_section_num}. Revisão da lista",
            "Confira a lista completa e confirme quando estiver pronta para liberar o sorteio."
        )
        render_revisao_lista(
            logic,
            lista_texto,
            render_action_button=render_action_button,
            diagnosticar_lista_no_estado=diagnosticar_lista_no_estado,
            atualizar_integridade_base_no_estado=atualizar_integridade_base_no_estado,
            render_correcao_inline_bloqueios_base=render_correcao_inline_bloqueios_base,
            lista_input_key=K.LISTA_TEXTO_INPUT,
        )
        if st.session_state.get(K.SCROLL_PARA_REVISAO, False):
            destino_revisao = st.session_state.get(K.SCROLL_DESTINO_REVISAO, "top")
            components.html(
                f"""
                <script>
                const parentDoc = window.parent.document;
                const destino = {json.dumps(destino_revisao)};
                const maxTentativas = 40;
                let tentativas = 0;

                function rolarParaDestinoDaRevisao() {{
                    const topAnchor = parentDoc.getElementById("revisao-anchor");
                    const pendingAnchor = parentDoc.getElementById("revisao-pendencias-anchor");
                    const cadastroAnchor = parentDoc.getElementById("revisao-cadastro-anchor");
                    const confirmAnchor = parentDoc.getElementById("revisao-confirmar-anchor");
                    const alvoPreferencial = destino === "confirmar"
                        ? confirmAnchor
                        : (destino === "pendencias"
                            ? pendingAnchor
                            : (destino === "cadastro" ? cadastroAnchor : topAnchor));
                    const alvo = alvoPreferencial || topAnchor;

                    if (alvo) {{
                        window.parent.requestAnimationFrame(() => {{
                            alvo.scrollIntoView({{ behavior: "smooth", block: "start" }});
                        }});
                        return;
                    }}

                    if (tentativas < maxTentativas) {{
                        tentativas += 1;
                        window.parent.setTimeout(rolarParaDestinoDaRevisao, 120);
                    }}
                }}

                window.parent.setTimeout(rolarParaDestinoDaRevisao, 80);
                </script>
                """,
                height=0,
            )
            st.session_state[K.SCROLL_PARA_REVISAO] = False
            st.session_state[K.SCROLL_DESTINO_REVISAO] = "top"

        lista_confirmada_ok = bool(
            st.session_state[K.LISTA_REVISADA_CONFIRMADA] and st.session_state[K.LISTA_REVISADA]
        )

        if not lista_confirmada_ok:
            st.caption('Depois de confirmar a lista final, o app vai liberar os critérios e levar você direto ao botão de sortear.')

    if lista_confirmada_ok:
        criterios_section_num = int(titulo_secao_lista.split('.')[0]) + 2
        render_section_header(
            f"{criterios_section_num}. Critérios do sorteio",
            "Escolha quais características devem ser equilibradas entre os times."
        )
        with st.expander(resumo_expander_criterios(), expanded=False):
            st.checkbox("Equilibrar Posição", value=True, key="criterio_posicao")
            st.checkbox("Equilibrar Nota", value=True, key="criterio_nota")
            st.checkbox("Equilibrar Velocidade", value=True, key="criterio_velocidade")
            st.checkbox("Equilibrar Movimentação", value=True, key="criterio_movimentacao")

            criterios_ativos = obter_criterios_ativos()
            qtd_ativos = sum(criterios_ativos.values())

            st.caption(f"Configuração ativa: {resumo_criterios_ativos()}")

            if qtd_ativos == 4:
                st.caption(
                    "Modo padrão: o sorteio tentará equilibrar posição, nota, velocidade e movimentação."
                )
            elif qtd_ativos == 0:
                st.warning(
                    "Nenhum critério está ativo. O sorteio ficará mais próximo de uma divisão aleatória simples."
                )
            else:
                st.caption(
                    "Modo personalizado: o sorteio equilibrará apenas os critérios selecionados."
                )

        sorteio_section_num = criterios_section_num + 1
        render_section_header(
            f"{sorteio_section_num}. Sorteio",
            "Quando tudo estiver pronto, use o botão abaixo para sortear os times."
        )

        gate_pre_sorteio = construir_gate_pre_sorteio(logic, lista_texto, qtd_nomes_informados, n_times)
        with st.expander("📋 Ver resumo pré-sorteio", expanded=False):
            render_resumo_operacional_pre_sorteio(gate_pre_sorteio)
        render_sort_ready_panel(
            lista_revisada_ok,
            lista_confirmada_ok,
            base_pronta_ok,
            sorteio_aleatorio_lista=bool(gate_pre_sorteio.get("sorteio_aleatorio_lista", False)),
        )

        pode_sortear_agora = bool(gate_pre_sorteio["pronto_para_sortear"])
        diagnostico_atual = st.session_state[K.DIAGNOSTICO_LISTA] or {}

        if st.session_state.get(K.RESULTADO_INVALIDADO_MSG, False):
            st.info("O resultado anterior foi invalidado porque os dados de entrada mudaram. Faça um novo sorteio.")
            st.session_state[K.RESULTADO_INVALIDADO_MSG] = False

        if st.session_state.get(K.RESULTADO):
            render_step_cta_panel(
                "Sorteio concluído",
                "Use as ações do resultado para copiar, compartilhar ou ajustar e sortear novamente.",
                tone="success",
                eyebrow="Etapa concluída",
            )
        elif st.session_state[K.CADASTRO_GUIADO_ATIVO]:
            render_step_cta_panel(
                "Conclua o cadastro guiado para liberar o sorteio",
                "Finalize os jogadores faltantes e depois revise novamente a lista.",
                tone="warning",
                eyebrow="Próximo passo",
            )
        elif bool(gate_pre_sorteio.get("sorteio_aleatorio_lista", False)) and pode_sortear_agora:
            render_step_cta_panel(
                "Sortear times aleatoriamente",
                "Neste modo o app usa apenas os nomes únicos da lista.",
                tone="success",
                eyebrow="Próximo passo",
            )
        elif not lista_revisada_ok:
            render_step_cta_panel(
                "Revise a lista antes de sortear",
                "Confira os nomes e ajustes da revisão.",
                tone="warning",
                eyebrow="Próximo passo",
            )
        elif diagnostico_atual.get("tem_nao_encontrados", False):
            render_step_cta_panel(
                "Cadastre os faltantes para liberar o sorteio",
                "Inclua quem não foi encontrado e revise novamente a lista.",
                tone="warning",
                eyebrow="Próximo passo",
            )
        elif diagnostico_atual.get("tem_bloqueio_base", False):
            render_step_cta_panel(
                "Corrija a base atual antes de sortear",
                "Há registros inconsistentes que ainda bloqueiam o fluxo.",
                tone="warning",
                eyebrow="Próximo passo",
            )
        elif not lista_confirmada_ok:
            render_step_cta_panel(
                "Confirme a lista final para liberar o sorteio",
                "A revisão já foi concluída; falta apenas a confirmação final.",
                tone="warning",
                eyebrow="Próximo passo",
            )
        elif not base_pronta_ok:
            render_step_cta_panel(
                "Carregue ou complemente a base antes de sortear",
                "Volte à etapa 1 ou use o cadastro manual para completar os jogadores.",
                tone="warning",
                eyebrow="Próximo passo",
            )
        else:
            render_step_cta_panel(
                "Tudo pronto para sortear",
                "Quando quiser, execute o sorteio final dos times.",
                tone="success",
                eyebrow="Próximo passo",
            )

        st.markdown('<div id="sortear-anchor"></div>', unsafe_allow_html=True)
        if st.session_state.get(K.SCROLL_PARA_SORTEIO, False):
            components.html(
                """
                <script>
                const parentDoc = window.parent.document;
                const anchor = parentDoc.getElementById("sortear-anchor");
                if (anchor) {
                    anchor.scrollIntoView({ behavior: "smooth", block: "start" });
                }
                </script>
                """,
                height=0,
            )
            st.session_state[K.SCROLL_PARA_SORTEIO] = False
        sortear_times = render_action_button(
            "🎲 SORTEAR TIMES",
            key="acao_sortear_times",
            role="primary",
            disabled=not pode_sortear_agora,
            use_primary_type=True,
        )
    else:
        gate_pre_sorteio = construir_gate_pre_sorteio(logic, lista_texto, qtd_nomes_informados, n_times)
        pode_sortear_agora = False
        sortear_times = False

    if sortear_times:
        gate_pre_sorteio = construir_gate_pre_sorteio(logic, lista_texto, qtd_nomes_informados, n_times)
        revisao_atual_valida = (
            st.session_state[K.LISTA_REVISADA_CONFIRMADA]
            and st.session_state[K.LISTA_REVISADA] is not None
            and lista_texto == st.session_state[K.LISTA_TEXTO_REVISADO]
        )

        if not gate_pre_sorteio["pronto_para_sortear"]:
            render_resumo_operacional_pre_sorteio(gate_pre_sorteio)
            diagnostico = diagnosticar_lista_no_estado(logic, lista_texto)
            if diagnostico is None:
                st.warning("Cole uma lista de jogadores para revisar antes do sorteio.")
            elif diagnostico["tem_nao_encontrados"]:
                st.warning("Existem nomes não encontrados. Cadastre esses jogadores na etapa 3 e confirme a revisão antes de sortear.")
            elif diagnostico.get("tem_bloqueio_base", False):
                st.error("Existem nomes da lista com duplicidade ou inconsistência na base atual. Corrija a base e revise a lista novamente antes de sortear.")
            elif not revisao_atual_valida:
                st.info("Revise a lista e clique em **✅ Confirmar lista final** antes de sortear.")
            else:
                st.warning("O sorteio permanece bloqueado até a resolução das pendências operacionais acima.")
        else:
            if gate_pre_sorteio.get("sorteio_aleatorio_lista", False):
                try:
                    with st.spinner('Sorteando aleatoriamente...'):
                        nomes_sorteio, _ = extrair_nomes_unicos_da_lista(logic, lista_texto)
                        times = sortear_times_aleatorios_por_lista(nomes_sorteio, n_times)
                        st.session_state[K.RESULTADO] = times
                        st.session_state[K.RESULTADO_CONTEXTO] = {
                            'qtd_jogadores': len(nomes_sorteio),
                            'qtd_times': len([time for time in times if time]),
                            'criterios': {'pos': False, 'nota': False, 'vel': False, 'mov': False},
                            'modo_sorteio': 'aleatorio_lista',
                            'timestamp_sorteio_iso': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        st.session_state[K.RESULTADO_ASSINATURA] = construir_assinatura_entrada_sorteio(lista_texto, n_times)
                        st.session_state[K.RESULTADO_INVALIDADO_MSG] = False
                        st.session_state[K.SCROLL_PARA_RESULTADO] = True
                except Exception as e:
                    st.error(f"Erro: {e}")
            else:
                nomes_corrigidos = st.session_state[K.LISTA_REVISADA]

                if st.session_state[K.DF_BASE].empty:
                    st.session_state[K.AVISO_SEM_PLANILHA] = True
                    st.session_state[K.NOMES_PENDENTES] = nomes_corrigidos
                    st.rerun()

                conhecidos = st.session_state[K.DF_BASE]['Nome'].tolist()
                novos_nomes_temp = [x['Nome'] for x in st.session_state[K.NOVOS_JOGADORES]]
                faltantes = [n for n in nomes_corrigidos if n not in conhecidos and n not in novos_nomes_temp]

                if faltantes:
                    st.session_state[K.FALTANTES_TEMP] = faltantes
                    st.rerun()
                else:
                    df_final = st.session_state[K.DF_BASE].copy()
                    if st.session_state[K.NOVOS_JOGADORES]:
                        df_final = pd.concat([df_final, pd.DataFrame(st.session_state[K.NOVOS_JOGADORES])], ignore_index=True)

                    df_jogar, nomes_bloqueados_base = preparar_df_sorteio(df_final, nomes_corrigidos)

                    if nomes_bloqueados_base:
                        detalhes = " | ".join(
                            [f"{item['nome']}: {'; '.join(item['motivos'])}" for item in nomes_bloqueados_base]
                        )
                        st.error(
                            "Não é possível realizar o sorteio porque há nomes com duplicidade ou inconsistência na base atual. "
                            f"Corrija a base primeiro. Detalhes: {detalhes}"
                        )
                    else:
                        try:
                            with st.spinner('Sorteando...'):
                                criterios_ativos = obter_criterios_ativos()
                                times = logic.otimizar(df_jogar, n_times, criterios_ativos)
                                st.session_state[K.RESULTADO] = times
                                st.session_state[K.RESULTADO_CONTEXTO] = {
                                    'qtd_jogadores': len(nomes_corrigidos),
                                    'qtd_times': len([time for time in times if time]),
                                    'criterios': criterios_ativos.copy(),
                                    'modo_sorteio': 'balanceado',
                                    'timestamp_sorteio_iso': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                }
                                st.session_state[K.RESULTADO_ASSINATURA] = construir_assinatura_entrada_sorteio(lista_texto, n_times)
                                st.session_state[K.RESULTADO_INVALIDADO_MSG] = False
                                st.session_state[K.SCROLL_PARA_RESULTADO] = True
                        except Exception as e:
                            st.error(f"Erro: {e}")

    if st.session_state.get(K.AVISO_SEM_PLANILHA):
        st.warning("⚠️ NENHUMA BASE FOI CARREGADA!")
        st.markdown(f"""
        Você ainda não carregou uma base administrada nem enviou uma planilha própria.

        Você pode seguir para a **etapa 3** e adicionar manualmente os **{len(st.session_state[K.NOMES_PENDENTES])} jogadores** da lista,
        ou voltar à **etapa 1** para carregar uma base antes do sorteio.
        """)

        col_conf1, col_conf2 = st.columns(2)
        if col_conf1.button("✅ Seguir para cadastro manual"):
            st.session_state[K.FALTANTES_TEMP] = st.session_state[K.NOMES_PENDENTES]
            st.session_state[K.AVISO_SEM_PLANILHA] = False
            st.rerun()

        if col_conf2.button("📂 Voltar para carregar base"):
            st.session_state[K.AVISO_SEM_PLANILHA] = False
            st.rerun()

    if K.FALTANTES_TEMP in st.session_state and st.session_state[K.FALTANTES_TEMP]:
        nome_atual = st.session_state[K.FALTANTES_TEMP][0]
        atual_i = len(st.session_state[K.NOVOS_JOGADORES]) + 1

        st.info(f"🆕 Cadastrando novo jogador ({atual_i}): **{nome_atual}**")

        with st.form("form_cadastro_faltante"):
            n_val = st.slider("Nota", 1.0, 10.0, 6.0, 0.5)
            p_val = st.selectbox("Posição", ["M", "A", "D"])
            v_val = st.slider("Velocidade", 1, 5, 3)
            m_val = st.slider("Movimentação", 1, 5, 3)

            if st.form_submit_button("Salvar e Próximo"):
                novo = {'Nome': nome_atual, 'Nota': n_val, 'Posição': p_val, 'Velocidade': v_val, 'Movimentação': m_val}
                st.session_state[K.DF_BASE].loc[len(st.session_state[K.DF_BASE])] = novo
                st.session_state[K.FALTANTES_TEMP].pop(0)
                st.rerun()

    if K.RESULTADO in st.session_state and not st.session_state.get(K.AVISO_SEM_PLANILHA) and not st.session_state.get(K.FALTANTES_TEMP):
        st.markdown('<div id="resultado-anchor"></div>', unsafe_allow_html=True)
        render_section_header(
            "6. Resultado do sorteio",
            "Confira os times abaixo e, quando estiver tudo certo, copie ou compartilhe o resultado."
        )

        if st.session_state.get(K.SCROLL_PARA_RESULTADO, False):
            components.html(
                '''
                <script>
                const parentDoc = window.parent.document;
                const anchor = parentDoc.getElementById("resultado-anchor");
                if (anchor) {
                    anchor.scrollIntoView({ behavior: "smooth", block: "start" });
                }
                </script>
                ''',
                height=0,
            )
            st.session_state[K.SCROLL_PARA_RESULTADO] = False
        times = st.session_state[K.RESULTADO]
        contexto_resultado = st.session_state.get(K.RESULTADO_CONTEXTO, {})
        modo_sorteio_resultado = contexto_resultado.get('modo_sorteio', 'balanceado')
        if modo_sorteio_resultado == 'aleatorio_lista':
            odds = [None for _ in times]
        else:
            odds = logic.calcular_odds(times)

        criterios_resultado = contexto_resultado.get('criterios', obter_criterios_ativos())

        criterios_ativos_legiveis = []
        if criterios_resultado.get('pos', False):
            criterios_ativos_legiveis.append('Posição')
        if criterios_resultado.get('nota', False):
            criterios_ativos_legiveis.append('Nota')
        if criterios_resultado.get('vel', False):
            criterios_ativos_legiveis.append('Velocidade')
        if criterios_resultado.get('mov', False):
            criterios_ativos_legiveis.append('Movimentação')

        if modo_sorteio_resultado == 'aleatorio_lista':
            modo_criterios = 'Aleatório'
            criterios_ativos_texto = 'Somente nomes únicos da lista · sem métricas e sem odds'
            observacao_resultado = 'Sorteio aleatório pela lista, sem uso de critérios de equilíbrio.'
        else:
            modo_criterios = 'Padrão' if all(criterios_resultado.values()) else 'Personalizado'
            if modo_criterios == 'Padrão':
                criterios_ativos_texto = 'Todos os 4 critérios'
            else:
                criterios_ativos_texto = ', '.join(criterios_ativos_legiveis) if criterios_ativos_legiveis else 'Nenhum'
            observacao_resultado = ''
        qtd_jogadores_resultado = contexto_resultado.get('qtd_jogadores')
        if qtd_jogadores_resultado is None:
            lista_revisada_atual = st.session_state.get(K.LISTA_REVISADA) or []
            if lista_revisada_atual:
                qtd_jogadores_resultado = len(lista_revisada_atual)
            else:
                qtd_jogadores_resultado = sum(len(time) for time in times if time)
        qtd_times_resultado = contexto_resultado.get('qtd_times', len([time for time in times if time]))

        timestamp_sorteio_iso = contexto_resultado.get('timestamp_sorteio_iso', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        cabecalho_padronizado = construir_cabecalho_padronizado_sorteio(
            timestamp_iso=timestamp_sorteio_iso,
            modo_sorteio_resultado=modo_sorteio_resultado,
            qtd_jogadores_resultado=qtd_jogadores_resultado,
            qtd_times_resultado=qtd_times_resultado,
            modo_criterios=modo_criterios,
            criterios_ativos_texto=criterios_ativos_texto,
        )

        st.success("Times gerados com sucesso.")
        st.caption(
            f"{qtd_times_resultado} time(s) · {qtd_jogadores_resultado} jogador(es) · {cabecalho_padronizado['modo_legivel']}"
        )
        st.info("Revise os times abaixo. Depois use 📋 COPIAR ou 📤 COMPARTILHAR para enviar o resultado.")

        texto_copiar = construir_texto_compartilhamento_resultado(
            times=times,
        )
        render_acoes_resultado(
            texto_copiar=texto_copiar,
        )

        st.caption(cabecalho_padronizado['cabecalho_curto'])

        for i, time in enumerate(times):
            if not time:
                continue
            ordem = {'G': 0, 'D': 1, 'M': 2, 'A': 3}
            time.sort(key=lambda x: (ordem.get(x[2], 99), x[0]))

        render_team_cards(times, odds)

        with st.expander("📋 Ver detalhes do sorteio", expanded=False):
            render_result_summary_panel(
                qtd_jogadores_resultado=qtd_jogadores_resultado,
                qtd_times_resultado=qtd_times_resultado,
                modo_criterios=modo_criterios,
                criterios_ativos_texto=criterios_ativos_texto,
                modo_sorteio=modo_sorteio_resultado,
                observacao_resultado=observacao_resultado,
            )

    st.divider()
    render_app_meta_footer()

if __name__ == "__main__":
    main()
