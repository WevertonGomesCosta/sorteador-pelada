"""App principal do Sorteador Pelada PRO.

Arquivo organizado por blocos funcionais para facilitar manutenção, auditoria
e refatorações futuras da camada visual.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
from datetime import datetime

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
    resumo_inconsistencias_base,
    total_inconsistencias_base,
)
from ui.review_view import (
    render_correcao_inline_bloqueios_base,
    render_revisao_lista,
)
from state.ui_state import abrir_expander_cadastro_manual, ensure_local_session_state
from ui.group_config_view import render_group_config_expander
from ui.summary_strings import (
    obter_criterios_ativos,
    resumo_criterios_ativos,
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

    if "lista_texto_input" not in st.session_state:
        st.session_state.lista_texto_input = ""
    draft_lista_key = "lista_texto_input__draft"
    if draft_lista_key not in st.session_state:
        st.session_state[draft_lista_key] = st.session_state.lista_texto_input
    pending_lista_key = "lista_texto_input__pending"
    if pending_lista_key in st.session_state:
        novo_texto_lista = st.session_state.pop(pending_lista_key)
        st.session_state.lista_texto_input = novo_texto_lista
        st.session_state[draft_lista_key] = novo_texto_lista

    origem_fluxo_status = st.session_state.get("grupo_origem_fluxo")
    base_carregada_via_secao1 = bool(
        st.session_state.get("base_admin_carregada", False)
        or st.session_state.get("ultimo_arquivo")
    )
    fluxo_somente_lista = bool(
        origem_fluxo_status == "lista"
        and not base_carregada_via_secao1
    )

    processamento_status = logic.processar_lista(
        st.session_state.get("lista_texto_input", ""),
        return_metadata=True,
        emit_warning=False,
    )
    qtd_nomes_status = len(processamento_status.get("jogadores", []))
    qtd_ignorados_status = len(processamento_status.get("ignorados", []))
    diagnostico_status = st.session_state.get("diagnostico_lista") or {}
    lista_revisada_ok_status = bool(st.session_state.get("diagnostico_lista") is not None)
    lista_confirmada_ok_status = bool(
        st.session_state.get("lista_revisada_confirmada")
        and st.session_state.get("lista_revisada")
    )
    base_pronta_ok_status = bool(
        not st.session_state.df_base.empty or st.session_state.novos_jogadores
    )
    resultado_disponivel_status = bool(st.session_state.get("resultado"))
    escolha_inicial_pendente_status = bool(
        origem_fluxo_status is None
        and not base_carregada_via_secao1
        and qtd_nomes_status == 0
        and not st.session_state.get("cadastro_guiado_ativo", False)
    )

    if st.session_state.get("cadastro_guiado_ativo", False):
        fluxo_status = "Cadastro guiado em andamento"
    elif escolha_inicial_pendente_status:
        fluxo_status = "Escolha como iniciar"
    elif qtd_nomes_status == 0:
        fluxo_status = "Aguardando lista"
    elif not lista_revisada_ok_status:
        fluxo_status = "Revisão pendente"
    elif diagnostico_status.get("tem_nao_encontrados", False):
        fluxo_status = "Faltantes pendentes"
    elif diagnostico_status.get("tem_duplicados", False):
        fluxo_status = "Nomes repetidos na lista"
    elif diagnostico_status.get("tem_bloqueio_base", False):
        fluxo_status = "Base com bloqueios"
    elif not lista_confirmada_ok_status:
        fluxo_status = "Confirmação pendente"
    elif resultado_disponivel_status:
        fluxo_status = "Resultado disponível"
    else:
        fluxo_status = "Pronto para sortear"

    if st.session_state.get("is_admin", False):
        modo_atual_status = "Base do grupo"
    elif bool(st.session_state.get("ultimo_arquivo")):
        modo_atual_status = "Excel próprio"
    elif fluxo_somente_lista:
        modo_atual_status = "Apenas sorteio com lista"
    elif escolha_inicial_pendente_status:
        modo_atual_status = "Escolha inicial"
    else:
        modo_atual_status = "Público / base própria"

    if st.session_state.get("base_admin_carregada", False) and st.session_state.get("is_admin", False):
        base_status = f"Base do grupo carregada · {len(st.session_state.df_base)} jogador(es)"
    elif bool(st.session_state.get("ultimo_arquivo")):
        base_status = f"Excel próprio carregado · {len(st.session_state.df_base)} jogador(es)"
    elif base_pronta_ok_status:
        qtd_base_total = len(st.session_state.df_base) + len(st.session_state.novos_jogadores)
        base_status = f"Base disponível · {qtd_base_total} jogador(es)"
    else:
        base_status = "Nenhuma base carregada"

    if qtd_nomes_status == 0:
        lista_status = "Nenhuma lista informada"
    else:
        lista_status = f"{qtd_nomes_status} nome(s) reconhecido(s)"
        if qtd_ignorados_status > 0:
            lista_status += f" · {qtd_ignorados_status} linha(s) ignorada(s)"
        if lista_confirmada_ok_status:
            lista_status += " · confirmada"
        elif lista_revisada_ok_status:
            lista_status += " · revisada"

    if escolha_inicial_pendente_status:
        proxima_acao_status = "Escolha como iniciar: apenas lista, base do grupo ou Excel próprio"
    elif qtd_nomes_status == 0:
        proxima_acao_status = "Cole a lista de jogadores"
    elif st.session_state.get("cadastro_guiado_ativo", False):
        proxima_acao_status = "Concluir cadastro guiado dos faltantes"
    elif not lista_revisada_ok_status:
        proxima_acao_status = "Corrigir a lista na etapa de revisão"
    elif diagnostico_status.get("tem_nao_encontrados", False):
        proxima_acao_status = "Cadastrar faltantes na revisão"
    elif diagnostico_status.get("tem_duplicados", False):
        proxima_acao_status = "Corrigir nomes repetidos na revisão"
    elif diagnostico_status.get("tem_bloqueio_base", False):
        proxima_acao_status = "Corrigir inconsistências da base"
    elif not lista_confirmada_ok_status:
        proxima_acao_status = "Clicar em ✅ Confirmar lista final"
    elif resultado_disponivel_status:
        proxima_acao_status = "Copiar, compartilhar ou ajustar e sortear novamente"
    else:
        proxima_acao_status = "Clicar em 🎲 SORTEAR TIMES"

    render_session_status_panel(
        modo_atual=modo_atual_status,
        base_status=base_status,
        lista_status=lista_status,
        fluxo_status=fluxo_status,
        proxima_acao=proxima_acao_status,
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
        len(st.session_state.get("faltantes_revisao", [])) > 0
        or len((st.session_state.get("diagnostico_lista") or {}).get("nao_encontrados", [])) > 0
    )
    manual_section_visible = bool(
        st.session_state.get("manual_section_visible", False)
        or st.session_state.get("cadastro_manual_expanded", False)
        or st.session_state.get("cadastro_guiado_ativo", False)
        or st.session_state.get("revisao_pendente_pos_cadastro", False)
        or len(st.session_state.get("faltantes_revisao", [])) > 0
        or len(st.session_state.get("faltantes_cadastrados_na_rodada", [])) > 0
        or faltantes_identificados
    )
    st.session_state.manual_section_visible = manual_section_visible

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
                st.session_state.manual_section_visible = True
                st.session_state.cadastro_manual_expanded = True
                st.rerun()
        else:
            render_section_header(
                titulo_secao_manual,
                "Use esta etapa para montar sua base do zero ou complementar a base atual com novos jogadores."
            )
            if faltantes_identificados and not st.session_state.get("cadastro_guiado_ativo", False):
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
    if st.session_state.get("scroll_para_lista", False):
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
        st.session_state.scroll_para_lista = False
    st.markdown(f"**Modo:** {'🗂️ Base do grupo' if st.session_state.is_admin else '👤 Público (Base própria)'}")

    auto_revisar_lista = bool(st.session_state.pop("lista_texto_input__revisar", False))
    revisao_pendente_pos_cadastro = bool(st.session_state.get("revisao_pendente_pos_cadastro", False))
    mostrar_botao_revisao_principal = bool(
        not st.session_state.get("lista_revisada_confirmada", False)
        and not st.session_state.get("cadastro_guiado_ativo", False)
        and not revisao_pendente_pos_cadastro
    )

    lista_rascunho = st.text_area(
        "Cole a lista (Numerada ou não):",
        height=120,
        placeholder="1. Jogador A\n2. Jogador B...",
        key=draft_lista_key,
    )
    lista_texto = st.session_state.get("lista_texto_input", "")
    if mostrar_botao_revisao_principal:
        revisar_lista = st.button(
            "🔎 Revisar lista",
            type="primary",
            use_container_width=True,
        )
        if revisar_lista:
            st.session_state.lista_texto_input = lista_rascunho
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
        bool(st.session_state.lista_texto_revisado)
        and lista_texto != st.session_state.lista_texto_revisado
    )
    precisa_revisar_lista = bool(
        qtd_nomes_informados > 0
        and (
            st.session_state.diagnostico_lista is None
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
    elif st.session_state.get("lista_revisada_confirmada") and st.session_state.get("lista_revisada"):
        render_inline_status_note(
            "Lista final confirmada.",
            "A etapa da lista já está concluída.",
            tone="success",
        )
    elif st.session_state.get("diagnostico_lista") is not None:
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
        st.session_state.diagnostico_lista is not None
        or st.session_state.lista_revisada_confirmada
    ):
        limpar_estado_revisao_lista()
        st.info("A lista foi alterada após a última revisão. Revise novamente antes de sortear.")

    base_pronta_ok = bool(
        not st.session_state.df_base.empty or st.session_state.novos_jogadores
    )

    if revisar_lista or auto_revisar_lista:
        diagnostico = diagnosticar_lista_no_estado(logic, lista_texto)
        if diagnostico is None:
            st.session_state.scroll_para_revisao = False
            st.session_state.scroll_destino_revisao = "top"
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
                and not st.session_state.get("cadastro_guiado_ativo", False)
                and not tem_pendencias_revisao
            )
            st.session_state.scroll_para_revisao = True
            if tem_pendencias_revisao:
                st.session_state.scroll_destino_revisao = "pendencias"
            else:
                st.session_state.scroll_destino_revisao = (
                    "confirmar" if pode_ir_direto_para_confirmacao else "top"
                )

    review_stage_visible = bool(
        st.session_state.diagnostico_lista is not None
        or st.session_state.lista_revisada_confirmada
        or st.session_state.cadastro_guiado_ativo
        or st.session_state.revisao_pendente_pos_cadastro
        or len(st.session_state.get("faltantes_revisao", [])) > 0
        or len(st.session_state.get("faltantes_cadastrados_na_rodada", [])) > 0
    )

    if not review_stage_visible:
        st.caption("Depois de revisar a lista, as etapas de revisão, critérios e sorteio serão exibidas em sequência.")

    lista_revisada_ok = bool(st.session_state.diagnostico_lista is not None)
    lista_confirmada_ok = bool(
        st.session_state.lista_revisada_confirmada and st.session_state.lista_revisada
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
            lista_input_key="lista_texto_input",
        )
        if st.session_state.get("scroll_para_revisao", False):
            destino_revisao = st.session_state.get("scroll_destino_revisao", "top")
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
            st.session_state.scroll_para_revisao = False
            st.session_state.scroll_destino_revisao = "top"

        lista_confirmada_ok = bool(
            st.session_state.lista_revisada_confirmada and st.session_state.lista_revisada
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
        diagnostico_atual = st.session_state.diagnostico_lista or {}

        if st.session_state.get("resultado_invalidado_msg", False):
            st.info("O resultado anterior foi invalidado porque os dados de entrada mudaram. Faça um novo sorteio.")
            st.session_state.resultado_invalidado_msg = False

        if st.session_state.get("resultado"):
            render_step_cta_panel(
                "Sorteio concluído",
                "Use as ações do resultado para copiar, compartilhar ou ajustar e sortear novamente.",
                tone="success",
                eyebrow="Etapa concluída",
            )
        elif st.session_state.cadastro_guiado_ativo:
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
        if st.session_state.get("scroll_para_sorteio", False):
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
            st.session_state.scroll_para_sorteio = False
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
            st.session_state.lista_revisada_confirmada
            and st.session_state.lista_revisada is not None
            and lista_texto == st.session_state.lista_texto_revisado
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
                        st.session_state.resultado = times
                        st.session_state.resultado_contexto = {
                            'qtd_jogadores': len(nomes_sorteio),
                            'qtd_times': len([time for time in times if time]),
                            'criterios': {'pos': False, 'nota': False, 'vel': False, 'mov': False},
                            'modo_sorteio': 'aleatorio_lista',
                            'timestamp_sorteio_iso': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        st.session_state.resultado_assinatura = construir_assinatura_entrada_sorteio(lista_texto, n_times)
                        st.session_state.resultado_invalidado_msg = False
                        st.session_state.scroll_para_resultado = True
                except Exception as e:
                    st.error(f"Erro: {e}")
            else:
                nomes_corrigidos = st.session_state.lista_revisada

                if st.session_state.df_base.empty:
                    st.session_state.aviso_sem_planilha = True
                    st.session_state.nomes_pendentes = nomes_corrigidos
                    st.rerun()

                conhecidos = st.session_state.df_base['Nome'].tolist()
                novos_nomes_temp = [x['Nome'] for x in st.session_state.novos_jogadores]
                faltantes = [n for n in nomes_corrigidos if n not in conhecidos and n not in novos_nomes_temp]

                if faltantes:
                    st.session_state.faltantes_temp = faltantes
                    st.rerun()
                else:
                    df_final = st.session_state.df_base.copy()
                    if st.session_state.novos_jogadores:
                        df_final = pd.concat([df_final, pd.DataFrame(st.session_state.novos_jogadores)], ignore_index=True)

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
                                st.session_state.resultado = times
                                st.session_state.resultado_contexto = {
                                    'qtd_jogadores': len(nomes_corrigidos),
                                    'qtd_times': len([time for time in times if time]),
                                    'criterios': criterios_ativos.copy(),
                                    'modo_sorteio': 'balanceado',
                                    'timestamp_sorteio_iso': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                }
                                st.session_state.resultado_assinatura = construir_assinatura_entrada_sorteio(lista_texto, n_times)
                                st.session_state.resultado_invalidado_msg = False
                                st.session_state.scroll_para_resultado = True
                        except Exception as e:
                            st.error(f"Erro: {e}")

    if st.session_state.get('aviso_sem_planilha'):
        st.warning("⚠️ NENHUMA BASE FOI CARREGADA!")
        st.markdown(f"""
        Você ainda não carregou uma base administrada nem enviou uma planilha própria.

        Você pode seguir para a **etapa 3** e adicionar manualmente os **{len(st.session_state.nomes_pendentes)} jogadores** da lista,
        ou voltar à **etapa 1** para carregar uma base antes do sorteio.
        """)

        col_conf1, col_conf2 = st.columns(2)
        if col_conf1.button("✅ Seguir para cadastro manual"):
            st.session_state.faltantes_temp = st.session_state.nomes_pendentes
            st.session_state.aviso_sem_planilha = False
            st.rerun()

        if col_conf2.button("📂 Voltar para carregar base"):
            st.session_state.aviso_sem_planilha = False
            st.rerun()

    if 'faltantes_temp' in st.session_state and st.session_state.faltantes_temp:
        nome_atual = st.session_state.faltantes_temp[0]
        atual_i = len(st.session_state.novos_jogadores) + 1

        st.info(f"🆕 Cadastrando novo jogador ({atual_i}): **{nome_atual}**")

        with st.form("form_cadastro_faltante"):
            n_val = st.slider("Nota", 1.0, 10.0, 6.0, 0.5)
            p_val = st.selectbox("Posição", ["M", "A", "D"])
            v_val = st.slider("Velocidade", 1, 5, 3)
            m_val = st.slider("Movimentação", 1, 5, 3)

            if st.form_submit_button("Salvar e Próximo"):
                novo = {'Nome': nome_atual, 'Nota': n_val, 'Posição': p_val, 'Velocidade': v_val, 'Movimentação': m_val}
                st.session_state.df_base.loc[len(st.session_state.df_base)] = novo
                st.session_state.faltantes_temp.pop(0)
                st.rerun()

    if 'resultado' in st.session_state and not st.session_state.get('aviso_sem_planilha') and not st.session_state.get('faltantes_temp'):
        st.markdown('<div id="resultado-anchor"></div>', unsafe_allow_html=True)
        render_section_header(
            "6. Resultado do sorteio",
            "Confira os times abaixo e, quando estiver tudo certo, copie ou compartilhe o resultado."
        )

        if st.session_state.get("scroll_para_resultado", False):
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
            st.session_state.scroll_para_resultado = False
        times = st.session_state.resultado
        contexto_resultado = st.session_state.get('resultado_contexto', {})
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
            lista_revisada_atual = st.session_state.get('lista_revisada') or []
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
