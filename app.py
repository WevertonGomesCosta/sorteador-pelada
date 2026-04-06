"""App principal do Sorteador Pelada PRO.

Arquivo organizado por blocos funcionais para facilitar manutenção, auditoria
e refatorações futuras da camada visual.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import hashlib
import json

from core.logic import PeladaLogic
from state.session import (
    atualizar_integridade_base_no_estado,
    diagnosticar_lista_no_estado,
    init_session_state,
    limpar_estado_revisao_lista,
)
from ui.components import botao_copiar_js, botao_instalar_app
from ui.manual_card import render_manual_card
from ui.result_view import (
    render_result_summary_panel,
    render_sort_ready_panel,
    render_team_cards,
)
from ui.styles import apply_app_styles
from core.validators import (
    diagnosticar_nomes_bloqueados_para_sorteio,
    listar_bloqueios_base_atual,
    normalizar_nome_comparacao,
    preparar_df_sorteio,
    registro_valido_para_sorteio,
    valor_slider_corrigir,
)
from ui.sections import (
    formatar_df_visual_numeros_inteiros,
    obter_criterios_ativos,
    render_base_inconsistencias_expander,
    render_base_integrity_alert,
    render_base_preview,
    render_base_summary,
    render_correcao_inline_bloqueios_base,
    render_correcao_inline_etapa2,
    render_revisao_lista,
    render_section_header,
    resumo_criterios_ativos,
    resumo_expander_cadastro_manual,
    resumo_expander_configuracao,
    resumo_expander_criterios,
)

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

# ============================================================================
# BLOCO 1 — UTILITÁRIOS GERAIS E NORMALIZAÇÃO
# ============================================================================

# ============================================================================
# BLOCO 2 — ESTADO DA BASE E INTEGRIDADE
# ============================================================================

def registrar_base_carregada_no_estado(logic, df_base: pd.DataFrame, *, is_admin: bool, ultimo_arquivo: str | None):
    st.session_state.df_base = df_base
    st.session_state.novos_jogadores = []
    st.session_state.is_admin = is_admin
    st.session_state.base_admin_carregada = is_admin
    st.session_state.ultimo_arquivo = ultimo_arquivo
    st.session_state.qtd_jogadores_adicionados_manualmente = 0
    atualizar_integridade_base_no_estado(logic)

# ============================================================================
# BLOCO 3 — REVISÃO E CORREÇÃO DA BASE / LISTA
# ============================================================================

def construir_assinatura_entrada_sorteio(lista_texto: str, n_times: int) -> str:
    cols = ["Nome", "Nota", "Posição", "Velocidade", "Movimentação"]

    df_base = st.session_state.get("df_base")
    if df_base is None or df_base.empty:
        base_json = "[]"
    else:
        df_base_ref = df_base.copy()
        for col in cols:
            if col not in df_base_ref.columns:
                df_base_ref[col] = ""
        base_json = (
            df_base_ref[cols]
            .astype(str)
            .to_json(orient="split", force_ascii=False)
        )

    novos_jogadores = st.session_state.get("novos_jogadores", [])
    if novos_jogadores:
        df_novos = pd.DataFrame(novos_jogadores)
        for col in cols:
            if col not in df_novos.columns:
                df_novos[col] = ""
        novos_json = (
            df_novos[cols]
            .astype(str)
            .to_json(orient="split", force_ascii=False)
        )
    else:
        novos_json = "[]"

    payload = {
        "lista_texto": lista_texto or "",
        "n_times": int(n_times),
        "criterios": obter_criterios_ativos(),
        "base_json": base_json,
        "novos_json": novos_json,
    }
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

def invalidar_resultado_se_entrada_mudou(lista_texto: str, n_times: int):
    if "resultado" not in st.session_state:
        return

    assinatura_anterior = st.session_state.get("resultado_assinatura")
    if not assinatura_anterior:
        return

    assinatura_atual = construir_assinatura_entrada_sorteio(lista_texto, n_times)
    if assinatura_atual == assinatura_anterior:
        return

    st.session_state.pop("resultado", None)
    st.session_state.pop("resultado_contexto", None)
    st.session_state.resultado_assinatura = None
    st.session_state.scroll_para_resultado = False
    st.session_state.resultado_invalidado_msg = True

# ============================================================================
# BLOCO 4 — SESSION STATE LOCAL E CONTROLES DE UI
# ============================================================================

def ensure_local_session_state():
    if "base_admin_carregada" not in st.session_state:
        st.session_state.base_admin_carregada = False
    if "base_inconsistencias_carregamento" not in st.session_state:
        st.session_state.base_inconsistencias_carregamento = {}
    if "base_registros_inconsistentes_carregamento" not in st.session_state:
        st.session_state.base_registros_inconsistentes_carregamento = []
    if "senha_admin_confirmada" not in st.session_state:
        st.session_state.senha_admin_confirmada = False
    if "ultima_senha_digitada" not in st.session_state:
        st.session_state.ultima_senha_digitada = ""
    if "qtd_jogadores_adicionados_manualmente" not in st.session_state:
        st.session_state.qtd_jogadores_adicionados_manualmente = 0
    if "cadastro_manual_expanded" not in st.session_state:
        st.session_state.cadastro_manual_expanded = False
    if "cadastro_manual_nome_existente" not in st.session_state:
        st.session_state.cadastro_manual_nome_existente = ""
    if "criterio_posicao" not in st.session_state:
        st.session_state.criterio_posicao = True
    if "criterio_nota" not in st.session_state:
        st.session_state.criterio_nota = True
    if "criterio_velocidade" not in st.session_state:
        st.session_state.criterio_velocidade = True
    if "criterio_movimentacao" not in st.session_state:
        st.session_state.criterio_movimentacao = True
    if "scroll_para_resultado" not in st.session_state:
        st.session_state.scroll_para_resultado = False
    if "resultado_assinatura" not in st.session_state:
        st.session_state.resultado_assinatura = None
    if "resultado_invalidado_msg" not in st.session_state:
        st.session_state.resultado_invalidado_msg = False

def abrir_expander_grupo():
    st.session_state.grupo_config_expanded = True

def abrir_expander_cadastro_manual():
    st.session_state.cadastro_manual_expanded = True

def grupo_config_deve_abrir() -> bool:
    return bool(
        st.session_state.get("grupo_config_expanded", False)
        or str(st.session_state.get("grupo_nome_pelada", "")).strip()
        or str(st.session_state.get("grupo_senha_admin", "")).strip()
        or st.session_state.get("senha_admin_confirmada", False)
    )

def render_action_button(
    label: str,
    *,
    key: str,
    role: str = "secondary",
    disabled: bool = False,
    use_primary_type: bool = False,
):
    with st.container(key=f"action-{role}-{key}"):
        button_type = "primary" if use_primary_type else "secondary"
        return st.button(
            label,
            key=key,
            disabled=disabled,
            type=button_type,
        )

# ============================================================================
# BLOCO 5 — RENDERIZAÇÃO DA BASE E AUDITORIA DE DADOS
# ============================================================================

# ============================================================================
# BLOCO 6 — FLUXO DE CONFIGURAÇÃO, CARGA E CADASTRO
# ============================================================================

def render_group_config_expander(logic, nome_pelada_adm: str, senha_adm: str) -> str:
    if "grupo_config_expanded" not in st.session_state:
        st.session_state.grupo_config_expanded = False

    with st.expander(
        resumo_expander_configuracao(nome_pelada_adm),
        expanded=grupo_config_deve_abrir(),
    ):
        st.markdown("**🔐 Configuração do grupo**")
        nome_pelada = st.text_input(
            "Nome da Pelada (opcional):",
            placeholder="Ex: Pelada de Domingo",
            key="grupo_nome_pelada",
        )

        nome_informado = nome_pelada.strip()
        grupo_admin = nome_informado.upper() == str(nome_pelada_adm).upper()
        origem_base = "Excel próprio"
        senha = ""
        uploaded_file = None

        if not (st.session_state.base_admin_carregada and st.session_state.is_admin):
            st.button(
                "🔎 Verificar grupo",
                key="grupo_verificar_nome",
                on_click=abrir_expander_grupo,
            )

        if grupo_admin:
            if st.session_state.base_admin_carregada and st.session_state.is_admin:
                st.success("Base admin carregada com sucesso.")
                origem_base = "Base original (Admin)"
            else:
                st.success("Base administrada encontrada para este grupo.")
                origem_base = st.radio(
                    "Como deseja iniciar a base?",
                    ["Base original (Admin)", "Excel próprio"],
                    key="grupo_origem_base",
                )
                st.caption("Para usar a base do grupo, informe a senha e clique em **Carregar base de dados**.")
        else:
            if nome_informado:
                st.warning(
                    "Base não encontrada para esse nome. Corrija o nome, envie uma planilha própria ou siga para a etapa 3."
                )
            else:
                st.info(
                    "Não tem uma base pronta? Você pode enviar uma planilha própria agora ou seguir direto para a etapa 3."
                )
            st.caption("Preencha esse campo apenas se quiser usar uma base administrada.")

        admin_base_carregada = st.session_state.base_admin_carregada

        if origem_base == "Base original (Admin)":
            senha_atual = st.session_state.get("grupo_senha_admin", "")
            if st.session_state.ultima_senha_digitada != senha_atual:
                st.session_state.senha_admin_confirmada = False
                st.session_state.ultima_senha_digitada = senha_atual

        if not admin_base_carregada:
            st.markdown("---")
            st.markdown("**📂 Banco de dados**")
            st.caption("Escolha como carregar sua base ou siga para a etapa 3.")

            if origem_base == "Base original (Admin)":
                senha = st.text_input(
                    "Senha de Acesso:",
                    type="password",
                    key="grupo_senha_admin",
                )
                if st.button(
                    "🔐 Confirmar senha e carregar base",
                    key="grupo_confirmar_senha",
                    on_click=abrir_expander_grupo,
                ):
                    if not grupo_admin:
                        st.session_state.is_admin = False
                        st.session_state.base_admin_carregada = False
                        st.session_state.senha_admin_confirmada = False
                        if nome_informado:
                            st.error(
                                "Base não encontrada para esse nome. Corrija o nome, envie uma planilha própria ou siga para a etapa 3."
                            )
                        else:
                            st.warning(
                                "Informe um grupo válido para usar a base administrada ou siga para a etapa 3."
                            )
                    elif senha != str(senha_adm):
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
                if not st.session_state.senha_admin_confirmada:
                    st.caption("Depois de informar a senha, toque em **Confirmar senha e carregar base**.")
            else:
                st.write("Já tem uma planilha? Envie o arquivo abaixo e depois clique em **Carregar base de dados**.")
                uploaded_file = st.file_uploader(
                    "Enviar planilha Excel",
                    type=["xlsx"],
                    label_visibility="collapsed",
                    key="grupo_upload_planilha",
                )

        if (
            not admin_base_carregada
            and origem_base != "Base original (Admin)"
            and st.button(
                "📥 Carregar base de dados",
                key="grupo_carregar_base",
                on_click=abrir_expander_grupo,
            )
        ):
            if uploaded_file is None:
                if nome_informado and not grupo_admin:
                    st.warning(
                        "Base não encontrada para esse nome e nenhuma planilha foi enviada. Envie uma planilha própria ou siga para a etapa 3."
                    )
                else:
                    st.info(
                        "Você ainda não selecionou uma base para carregar. Envie uma planilha própria ou siga para a etapa 3."
                    )
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
                    st.session_state.grupo_config_expanded = True
                    st.rerun()

    return nome_pelada

# --- FRONTEND ---
# ============================================================================
# BLOCO 7 — FLUXO PRINCIPAL
# ============================================================================

def main():
    logic = PeladaLogic()
    st.title("⚽ Sorteador Pelada PRO")
    botao_instalar_app()

    init_session_state(logic)
    ensure_local_session_state()

    render_section_header(
        "1. Configuração do grupo e base de dados",
        "Escolha como iniciar sua base: usar a base do grupo, enviar uma planilha própria ou seguir para a etapa 3."
    )
    nome_pelada = render_group_config_expander(logic, NOME_PELADA_ADM, SENHA_ADM)

    render_section_header(
        "2. Base de jogadores",
        "Confira a base atual. Se ela estiver vazia, siga pela etapa 3 para cadastrar jogadores manualmente."
    )
    render_base_summary()
    render_base_integrity_alert()
    render_base_inconsistencias_expander()
    render_correcao_inline_etapa2(
        logic,
        render_correcao_inline_bloqueios_base=render_correcao_inline_bloqueios_base,
        atualizar_integridade_base_no_estado=atualizar_integridade_base_no_estado,
        diagnosticar_lista_no_estado=diagnosticar_lista_no_estado,
        render_action_button=render_action_button,
    )

    render_section_header(
        "3. Adicionar jogadores manualmente",
        "Use esta etapa para montar sua base do zero ou complementar a base atual com novos jogadores."
    )
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

    render_section_header(
        "4. Lista da pelada",
        "Cole aqui os nomes confirmados para o sorteio. Eles serão comparados com a base carregada e, se necessário, você poderá completar os jogadores manualmente."
    )
    st.markdown(f"**Modo:** {'🔐 ADMIN (Download Bloqueado)' if st.session_state.is_admin else '👤 Público (Base Própria)'}")
    lista_texto = st.text_area("Cole a lista (Numerada ou não):", height=120, placeholder="1. Jogador A\n2. Jogador B...")

    processamento_previa = logic.processar_lista(
        lista_texto,
        return_metadata=True,
        emit_warning=False,
    )
    qtd_nomes_informados = len(processamento_previa["jogadores"])
    qtd_itens_ignorados = len(processamento_previa.get("ignorados", []))

    if qtd_nomes_informados > 0 or qtd_itens_ignorados > 0:
        if qtd_itens_ignorados == 0:
            st.caption(
                f"Leitura atual: {qtd_nomes_informados} nomes reconhecidos · nenhuma linha ignorada."
            )
        else:
            st.caption(
                f"Leitura atual: {qtd_nomes_informados} nomes reconhecidos · {qtd_itens_ignorados} linhas ignoradas."
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

    lista_alterada_pos_revisao = (
        bool(st.session_state.lista_texto_revisado)
        and lista_texto != st.session_state.lista_texto_revisado
    )
    if lista_alterada_pos_revisao and (
        st.session_state.diagnostico_lista is not None
        or st.session_state.lista_revisada_confirmada
    ):
        limpar_estado_revisao_lista()
        st.info("A lista foi alterada após a última revisão. Revise novamente antes de sortear.")

    review_role = "primary"
    if st.session_state.diagnostico_lista is not None or st.session_state.lista_revisada_confirmada:
        review_role = "secondary"

    revisar_lista = render_action_button(
        "🔎 Revisar lista",
        key="acao_revisar_lista",
        role=review_role,
        use_primary_type=(review_role == "primary"),
    )

    if revisar_lista:
        diagnostico = diagnosticar_lista_no_estado(logic, lista_texto)
        if diagnostico is None:
            st.warning("Cole uma lista de jogadores para revisar antes do sorteio.")

    render_revisao_lista(
        logic,
        lista_texto,
        render_action_button=render_action_button,
        diagnosticar_lista_no_estado=diagnosticar_lista_no_estado,
        atualizar_integridade_base_no_estado=atualizar_integridade_base_no_estado,
        render_correcao_inline_bloqueios_base=render_correcao_inline_bloqueios_base,
    )

    render_section_header(
        "5. Critérios do sorteio",
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

    lista_revisada_ok = bool(st.session_state.diagnostico_lista is not None)
    lista_confirmada_ok = bool(
        st.session_state.lista_revisada_confirmada and st.session_state.lista_revisada
    )
    base_pronta_ok = bool(
        not st.session_state.df_base.empty or st.session_state.novos_jogadores
    )

    render_sort_ready_panel(lista_revisada_ok, lista_confirmada_ok, base_pronta_ok)

    pode_sortear_agora = bool(
        st.session_state.lista_revisada_confirmada
        and st.session_state.lista_revisada
        and not st.session_state.cadastro_guiado_ativo
        and st.session_state.diagnostico_lista
        and not st.session_state.diagnostico_lista.get("tem_nao_encontrados", False)
        and not st.session_state.diagnostico_lista.get("tem_bloqueio_base", False)
    )
    diagnostico_atual = st.session_state.diagnostico_lista or {}

    if st.session_state.get("resultado_invalidado_msg", False):
        st.info("O resultado anterior foi invalidado porque os dados de entrada mudaram. Faça um novo sorteio.")
        st.session_state.resultado_invalidado_msg = False

    if st.session_state.cadastro_guiado_ativo:
        st.caption('Próximo passo: conclua o cadastro guiado dos jogadores faltantes e depois revise a lista novamente.')
    elif not lista_revisada_ok:
        st.caption('Próximo passo: clique em "🔎 Revisar lista" para verificar nomes e pendências.')
    elif diagnostico_atual.get("tem_nao_encontrados", False):
        st.caption('Próximo passo: clique em "➕ Cadastrar faltantes agora", conclua o cadastro e depois revise a lista novamente.')
    elif diagnostico_atual.get("tem_bloqueio_base", False):
        st.caption('Próximo passo: corrija os registros duplicados ou inconsistentes da base atual e depois revise a lista novamente.')
    elif not lista_confirmada_ok:
        st.caption('Próximo passo: em "🔎 Revisão da lista", clique em "✅ Confirmar lista final".')
    elif not base_pronta_ok:
        st.caption("Próximo passo: carregue uma base na etapa 1 ou complete os jogadores na etapa 3.")
    else:
        st.markdown(
            "<div class='action-hint'>Tudo pronto. Ajuste os critérios, se quiser, e clique em “🎲 SORTEAR TIMES”.</div>",
            unsafe_allow_html=True,
        )

    sortear_times = render_action_button(
        "🎲 SORTEAR TIMES",
        key="acao_sortear_times",
        role="primary",
        disabled=not pode_sortear_agora,
        use_primary_type=True,
    )

    if sortear_times:
        revisao_atual_valida = (
            st.session_state.lista_revisada_confirmada
            and st.session_state.lista_revisada is not None
            and lista_texto == st.session_state.lista_texto_revisado
        )

        if not revisao_atual_valida:
            diagnostico = diagnosticar_lista_no_estado(logic, lista_texto)
            if diagnostico is None:
                st.warning("Cole uma lista de jogadores para revisar antes do sorteio.")
            elif diagnostico["tem_nao_encontrados"]:
                st.warning("Existem nomes não encontrados. Cadastre esses jogadores na etapa 3 e confirme a revisão antes de sortear.")
            elif diagnostico.get("tem_bloqueio_base", False):
                st.error("Existem nomes da lista com duplicidade ou inconsistência na base atual. Corrija a base e revise a lista novamente antes de sortear.")
            else:
                st.info("Revise a lista e clique em **✅ Confirmar lista final** antes de sortear.")
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
            "6. Resultado",
            "Veja os times sorteados e copie o resultado para compartilhar."
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
        odds = logic.calcular_odds(times)
        contexto_resultado = st.session_state.get('resultado_contexto', {})
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

        modo_criterios = 'Padrão' if all(criterios_resultado.values()) else 'Personalizado'
        if modo_criterios == 'Padrão':
            criterios_ativos_texto = 'Todos os 4 critérios'
        else:
            criterios_ativos_texto = ', '.join(criterios_ativos_legiveis) if criterios_ativos_legiveis else 'Nenhum'
        qtd_jogadores_resultado = contexto_resultado.get('qtd_jogadores', len(st.session_state.get('lista_revisada', [])))
        qtd_times_resultado = contexto_resultado.get('qtd_times', len([time for time in times if time]))

        render_result_summary_panel(
            qtd_jogadores_resultado=qtd_jogadores_resultado,
            qtd_times_resultado=qtd_times_resultado,
            modo_criterios=modo_criterios,
            criterios_ativos_texto=criterios_ativos_texto,
        )

        texto_copiar = ""
        st.markdown("---")
        for i, time in enumerate(times):
            if not time:
                continue
            ordem = {'G': 0, 'D': 1, 'M': 2, 'A': 3}
            time.sort(key=lambda x: (ordem.get(x[2], 99), x[0]))
            texto_copiar += f"*Time {i+1}:*\n"
            for p in time:
                texto_copiar += f"{p[0]}\n"
            texto_copiar += "\n"

        botao_copiar_js(texto_copiar)

        render_team_cards(times, odds)

if __name__ == "__main__":
    main()
