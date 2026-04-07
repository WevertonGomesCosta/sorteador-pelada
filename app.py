"""App principal do Sorteador Pelada PRO.

Arquivo organizado por blocos funcionais para facilitar manutenção, auditoria
e refatorações futuras da camada visual.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import hashlib
import json
import random
from datetime import datetime

from core.logic import PeladaLogic
from state.session import (
    atualizar_integridade_base_no_estado,
    diagnosticar_lista_no_estado,
    init_session_state,
    limpar_estado_revisao_lista,
    registrar_base_carregada_no_estado,
)
from ui.components import botao_compartilhar_js, botao_copiar_js, botao_instalar_app
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
    render_group_config_expander,
    render_revisao_lista,
    render_section_header,
    resumo_criterios_ativos,
    resumo_expander_cadastro_manual,
    resumo_expander_configuracao,
    resumo_expander_criterios,
    resumo_inconsistencias_base,
    total_inconsistencias_base,
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

def extrair_nomes_unicos_da_lista(logic, lista_texto: str) -> tuple[list[str], int]:
    processamento = logic.processar_lista(
        lista_texto,
        return_metadata=True,
        emit_warning=False,
    )
    nomes_lidos = processamento.get("jogadores", [])
    nomes_unicos = list(dict.fromkeys(nomes_lidos))
    qtd_duplicados = max(0, len(nomes_lidos) - len(nomes_unicos))
    return nomes_unicos, qtd_duplicados

def sortear_times_aleatorios_por_lista(nomes: list[str], n_times: int) -> list[list[list]]:
    nomes_embaralhados = nomes.copy()
    random.shuffle(nomes_embaralhados)

    times = [[] for _ in range(n_times)]
    for idx, nome in enumerate(nomes_embaralhados):
        time_idx = idx % n_times
        times[time_idx].append([nome, None, "", None, None])

    return times

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

def contar_duplicados_base_atual(df_base: pd.DataFrame) -> int:
    if df_base is None or df_base.empty or "Nome" not in df_base.columns:
        return 0

    nomes = (
        df_base["Nome"]
        .fillna("")
        .astype(str)
        .apply(normalizar_nome_comparacao)
    )
    nomes = nomes[nomes.ne("")]
    if nomes.empty:
        return 0
    return int(nomes[nomes.duplicated(keep=False)].nunique())

def construir_gate_pre_sorteio(logic, lista_texto: str, qtd_nomes_informados: int, n_times: int) -> dict:
    df_base = st.session_state.get("df_base", pd.DataFrame())
    diagnostico_atual = st.session_state.get("diagnostico_lista") or {}
    lista_texto_revisado = st.session_state.get("lista_texto_revisado", "")
    lista_revisada_atual = bool(diagnostico_atual) and lista_texto == lista_texto_revisado
    lista_confirmada_atual = bool(
        st.session_state.get("lista_revisada_confirmada")
        and st.session_state.get("lista_revisada")
        and lista_texto == lista_texto_revisado
    )
    cadastro_guiado_ativo = bool(st.session_state.get("cadastro_guiado_ativo", False))
    base_pronta = bool(not df_base.empty or st.session_state.get("novos_jogadores"))

    nomes_lista_unicos, qtd_duplicados_lista = extrair_nomes_unicos_da_lista(logic, lista_texto)
    qtd_nomes_unicos = len(nomes_lista_unicos)
    sorteio_aleatorio_lista = bool(not base_pronta and qtd_nomes_unicos > 0)

    if hasattr(logic, "diagnosticar_inconsistencias_base"):
        inconsistencias_base = logic.diagnosticar_inconsistencias_base(df_base)
    else:
        inconsistencias_base = st.session_state.get("base_inconsistencias_carregamento", {})

    total_inconsistencias = total_inconsistencias_base(inconsistencias_base)
    resumo_incons = resumo_inconsistencias_base(inconsistencias_base)
    qtd_duplicados_base = contar_duplicados_base_atual(df_base)

    faltantes = len(diagnostico_atual.get("nao_encontrados", [])) if lista_revisada_atual else 0
    bloqueios_base = len(diagnostico_atual.get("nomes_bloqueados_base", [])) if lista_revisada_atual else 0

    pendencias = []
    avisos = []

    if qtd_nomes_informados == 0:
        pendencias.append("nenhum nome foi informado na lista")

    if sorteio_aleatorio_lista:
        if qtd_nomes_unicos < 2:
            pendencias.append("o modo aleatório exige pelo menos 2 nomes únicos na lista")
        if qtd_nomes_unicos < n_times:
            pendencias.append(f"há apenas {qtd_nomes_unicos} nome(s) único(s) para {n_times} time(s)")

        avisos.append("Modo aleatório por lista ativo: nenhuma base foi carregada.")
        avisos.append("Os critérios de equilíbrio, métricas e odds não serão aplicados neste modo.")
        avisos.append(f"O sorteio usará {qtd_nomes_unicos} nome(s) único(s) informados na lista.")
        if qtd_duplicados_lista > 0:
            avisos.append(f"Há {qtd_duplicados_lista} repetição(ões) na lista; os nomes repetidos serão unificados antes do sorteio.")
    else:
        if not base_pronta:
            pendencias.append("nenhuma base foi carregada ou construída")
        elif not lista_revisada_atual:
            pendencias.append("a lista ainda não foi revisada com a versão atual dos dados")

        if cadastro_guiado_ativo:
            pendencias.append("há um cadastro guiado em andamento")
        if lista_revisada_atual and faltantes > 0:
            pendencias.append(f"há {faltantes} nome(s) faltante(s) na base")
        if lista_revisada_atual and bloqueios_base > 0:
            pendencias.append(f"há {bloqueios_base} nome(s) com duplicidade ou inconsistência na base atual")
        if lista_revisada_atual and not lista_confirmada_atual:
            pendencias.append("a lista revisada ainda não foi confirmada")

    if qtd_duplicados_base > 0:
        avisos.append(f"Base atual com {qtd_duplicados_base} nome(s) duplicado(s).")
    if total_inconsistencias > 0:
        detalhe = f": {resumo_incons}" if resumo_incons else ""
        avisos.append(f"Base atual com {total_inconsistencias} inconsistência(s){detalhe}.")

    nomes_referencia_alerta = qtd_nomes_unicos if sorteio_aleatorio_lista else qtd_nomes_informados
    if nomes_referencia_alerta > 0 and nomes_referencia_alerta < n_times * 2:
        avisos.append("Há poucos nomes para a quantidade de times escolhida; o sorteio pode ficar menos equilibrado.")

    if sorteio_aleatorio_lista:
        base_status = "sem base carregada · modo aleatório pela lista"
        lista_status = f"{qtd_nomes_unicos} nome(s) único(s)"
        if qtd_duplicados_lista > 0:
            lista_status += f" · {qtd_duplicados_lista} repetição(ões) unificada(s)"
        criterios_status = "Ignorados · sorteio apenas pelos nomes da lista"
        prontidao_status = "Pronto para sortear · modo aleatório" if len(pendencias) == 0 else f"Bloqueado · {len(pendencias)} pendência(s)"
        modo_sorteio = "aleatorio_lista"
        modo_status = "Aleatório por lista"
    else:
        base_status = f"{len(df_base)} jogador(es)" if base_pronta else "sem base carregada"
        if qtd_duplicados_base > 0 or total_inconsistencias > 0:
            partes_base = [base_status]
            if qtd_duplicados_base > 0:
                partes_base.append(f"{qtd_duplicados_base} duplicidade(s)")
            if total_inconsistencias > 0:
                partes_base.append(f"{total_inconsistencias} inconsistência(s)")
            base_status = " · ".join(partes_base)

        lista_status = f"{qtd_nomes_informados} nome(s) lido(s)"
        if lista_revisada_atual:
            lista_status += " · revisada"
            if lista_confirmada_atual:
                lista_status += " · confirmada"
        elif qtd_nomes_informados > 0:
            lista_status += " · pendente de revisão"

        criterios_status = resumo_criterios_ativos()
        prontidao_status = "Pronto para sortear" if len(pendencias) == 0 else f"Bloqueado · {len(pendencias)} pendência(s)"
        modo_sorteio = "balanceado"
        modo_status = "Balanceado com base"

    return {
        "pronto_para_sortear": len(pendencias) == 0,
        "base_status": base_status,
        "lista_status": lista_status,
        "criterios_status": criterios_status,
        "prontidao_status": prontidao_status,
        "pendencias": pendencias,
        "avisos": avisos,
        "modo_sorteio": modo_sorteio,
        "modo_status": modo_status,
        "sorteio_aleatorio_lista": sorteio_aleatorio_lista,
        "qtd_nomes_unicos_lista": qtd_nomes_unicos,
        "qtd_duplicados_lista": qtd_duplicados_lista,
    }

def render_resumo_operacional_pre_sorteio(gate_pre_sorteio: dict):
    st.markdown(
        f"""
        <div class="theme-panel theme-panel--summary">
            <div class="theme-panel__title">Resumo operacional pré-sorteio</div>
            <div class="theme-panel__line">🎲 <span class="theme-panel__label">Modo:</span> <span class="theme-panel__strong">{gate_pre_sorteio['modo_status']}</span></div>
            <div class="theme-panel__line">📋 <span class="theme-panel__label">Base:</span> <span class="theme-panel__strong">{gate_pre_sorteio['base_status']}</span></div>
            <div class="theme-panel__line">📝 <span class="theme-panel__label">Lista:</span> <span class="theme-panel__strong">{gate_pre_sorteio['lista_status']}</span></div>
            <div class="theme-panel__line">⚙️ <span class="theme-panel__label">Critérios ativos:</span> <span class="theme-panel__strong">{gate_pre_sorteio['criterios_status']}</span></div>
            <div class="theme-panel__line">🚦 <span class="theme-panel__label">Prontidão:</span> <span class="theme-panel__strong">{gate_pre_sorteio['prontidao_status']}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if gate_pre_sorteio["pendencias"]:
        pendencias_md = "\n".join([f"- {item.capitalize()}." for item in gate_pre_sorteio["pendencias"]])
        st.warning(f"O sorteio está bloqueado até a resolução das pendências abaixo:\n{pendencias_md}")
    elif gate_pre_sorteio["modo_sorteio"] == "aleatorio_lista":
        avisos_md = "\n".join([f"- {item}" for item in gate_pre_sorteio["avisos"]])
        st.warning(f"Modo aleatório por lista ativo. Confira abaixo antes de sortear:\n{avisos_md}")
    elif gate_pre_sorteio["avisos"]:
        avisos_md = "\n".join([f"- {item}" for item in gate_pre_sorteio["avisos"]])
        st.info(f"Situação geral estável. Pontos de atenção:\n{avisos_md}")
    else:
        st.success("Tudo conferido. O app está pronto para realizar o sorteio.")

# ============================================================================
# BLOCO 4 — EXPORTAÇÃO E COMPARTILHAMENTO DO RESULTADO
# ============================================================================

def formatar_timestamp_sorteio_para_exibicao(timestamp_iso: str) -> str:
    if not timestamp_iso:
        return ""
    try:
        return datetime.strptime(timestamp_iso, "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y %H:%M")
    except Exception:
        return timestamp_iso

def formatar_timestamp_sorteio_para_arquivo(timestamp_iso: str) -> str:
    if not timestamp_iso:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        return datetime.strptime(timestamp_iso, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d_%H%M%S")
    except Exception:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

def construir_cabecalho_padronizado_sorteio(
    *,
    timestamp_iso: str,
    modo_sorteio_resultado: str,
    qtd_jogadores_resultado: int,
    qtd_times_resultado: int,
    modo_criterios: str,
    criterios_ativos_texto: str,
) -> dict:
    data_hora_exibicao = formatar_timestamp_sorteio_para_exibicao(timestamp_iso)
    modo_legivel = "Aleatório por lista" if modo_sorteio_resultado == "aleatorio_lista" else "Balanceado com base"
    titulo = "Sorteio aleatório pela lista" if modo_sorteio_resultado == "aleatorio_lista" else "Sorteio balanceado com base"
    cabecalho_txt = (
        f"{titulo}\n"
        f"Data/Hora: {data_hora_exibicao}\n"
        f"Modo: {modo_legivel}\n"
        f"Times: {qtd_times_resultado}\n"
        f"Jogadores: {qtd_jogadores_resultado}\n"
        f"Critérios: {modo_criterios}\n"
        f"Ativos: {criterios_ativos_texto}"
    )
    cabecalho_curto = (
        f"{titulo} · {data_hora_exibicao} · {qtd_times_resultado} time(s) · "
        f"{qtd_jogadores_resultado} jogador(es) · {modo_legivel}"
    )
    return {
        "titulo": titulo,
        "modo_legivel": modo_legivel,
        "data_hora_exibicao": data_hora_exibicao,
        "cabecalho_txt": cabecalho_txt,
        "cabecalho_curto": cabecalho_curto,
        "timestamp_arquivo": formatar_timestamp_sorteio_para_arquivo(timestamp_iso),
    }

def construir_texto_compartilhamento_resultado(*, times) -> str:
    linhas = []
    for i, time in enumerate(times):
        if not time:
            continue
        linhas.append(f"*Time {i+1}:*")
        for p in time:
            linhas.append(str(p[0]))
        linhas.append("")
    return "\n".join(linhas).strip() + "\n"

def render_acoes_resultado(texto_copiar: str):
    col_copy, col_share = st.columns(2)
    with col_copy:
        botao_copiar_js(texto_copiar)
    with col_share:
        botao_compartilhar_js(texto_copiar)

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

def abrir_expander_cadastro_manual():
    st.session_state.cadastro_manual_expanded = True

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

# ============================================================================
# BLOCO 7 — FLUXO PRINCIPAL
# ============================================================================

def main():
    logic = PeladaLogic()
    st.title("⚽ Sorteador Pelada PRO")
    botao_instalar_app()

    init_session_state(logic)
    ensure_local_session_state()

    base_carregada_via_secao1 = bool(
        st.session_state.get("base_admin_carregada", False)
        or st.session_state.get("ultimo_arquivo")
    )

    render_section_header(
        "1. Configuração do grupo e base de dados",
        "Escolha se deseja carregar a base do grupo ou enviar uma planilha própria."
    )
    nome_pelada = render_group_config_expander(logic, NOME_PELADA_ADM, SENHA_ADM)

    if base_carregada_via_secao1:
        render_section_header(
            "2. Base de jogadores",
            "Confira a base atual carregada a partir da etapa 1."
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
        titulo_secao_manual = "3. Adicionar jogadores manualmente"
        titulo_secao_lista = "4. Lista da pelada"
        subtitulo_lista = "Cole aqui os nomes confirmados para o sorteio. Eles serão comparados com a base carregada e, se necessário, você poderá completar os jogadores manualmente."
    else:
        titulo_secao_manual = "2. Adicionar jogadores manualmente"
        titulo_secao_lista = "3. Lista da pelada"
        subtitulo_lista = "Cole aqui os nomes confirmados para o sorteio. Sem base carregada, o app poderá sortear aleatoriamente entre os nomes únicos da lista."

    render_section_header(
        titulo_secao_manual,
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
        titulo_secao_lista,
        subtitulo_lista,
    )
    st.markdown(f"**Modo:** {'🔐 ADMIN (Download Bloqueado)' if st.session_state.is_admin else '👤 Público (Base Própria)'}")
    if "lista_texto_input" not in st.session_state:
        st.session_state.lista_texto_input = ""
    lista_texto = st.text_area(
        "Cole a lista (Numerada ou não):",
        height=120,
        placeholder="1. Jogador A\n2. Jogador B...",
        key="lista_texto_input",
    )

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
        lista_input_key="lista_texto_input",
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

    gate_pre_sorteio = construir_gate_pre_sorteio(logic, lista_texto, qtd_nomes_informados, n_times)
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

    if st.session_state.cadastro_guiado_ativo:
        st.caption('Próximo passo: conclua o cadastro guiado dos jogadores faltantes e depois revise a lista novamente.')
    elif bool(gate_pre_sorteio.get("sorteio_aleatorio_lista", False)):
        st.warning("Modo aleatório por lista ativo: sem base carregada, sem critérios de equilíbrio e com uso apenas dos nomes únicos informados na lista.")
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
                            'qtd_duplicados_unificados': int(gate_pre_sorteio.get('qtd_duplicados_lista', 0)),
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
            qtd_duplicados_unificados = int(contexto_resultado.get('qtd_duplicados_unificados', 0))
            if qtd_duplicados_unificados > 0:
                observacao_resultado += f' Repetições unificadas: {qtd_duplicados_unificados}.'
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

        render_result_summary_panel(
            qtd_jogadores_resultado=qtd_jogadores_resultado,
            qtd_times_resultado=qtd_times_resultado,
            modo_criterios=modo_criterios,
            criterios_ativos_texto=criterios_ativos_texto,
            modo_sorteio=modo_sorteio_resultado,
            observacao_resultado=observacao_resultado,
        )

        timestamp_sorteio_iso = contexto_resultado.get('timestamp_sorteio_iso', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        cabecalho_padronizado = construir_cabecalho_padronizado_sorteio(
            timestamp_iso=timestamp_sorteio_iso,
            modo_sorteio_resultado=modo_sorteio_resultado,
            qtd_jogadores_resultado=qtd_jogadores_resultado,
            qtd_times_resultado=qtd_times_resultado,
            modo_criterios=modo_criterios,
            criterios_ativos_texto=criterios_ativos_texto,
        )
        st.caption(cabecalho_padronizado['cabecalho_curto'])

        st.markdown("---")
        for i, time in enumerate(times):
            if not time:
                continue
            ordem = {'G': 0, 'D': 1, 'M': 2, 'A': 3}
            time.sort(key=lambda x: (ordem.get(x[2], 99), x[0]))

        texto_copiar = construir_texto_compartilhamento_resultado(
            times=times,
        )
        render_acoes_resultado(
            texto_copiar=texto_copiar,
        )

        render_team_cards(times, odds)

if __name__ == "__main__":
    main()
