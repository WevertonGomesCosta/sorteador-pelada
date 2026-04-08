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
from ui.group_config_view import render_group_config_expander
from ui.summary_strings import (
    obter_criterios_ativos,
    resumo_criterios_ativos,
    resumo_expander_cadastro_manual,
    resumo_expander_configuracao,
    resumo_expander_criterios,
)
from ui.panels import render_session_status_panel

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
            avisos.append(f"Há {qtd_duplicados_lista} repetição(ões) na lista; revise e corrija os nomes repetidos antes do sorteio.")
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
            lista_status += f" · {qtd_duplicados_lista} repetição(ões) para revisar"
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
    if "scroll_para_lista" not in st.session_state:
        st.session_state.scroll_para_lista = False
    if "scroll_para_revisao" not in st.session_state:
        st.session_state.scroll_para_revisao = False
    if "scroll_destino_revisao" not in st.session_state:
        st.session_state.scroll_destino_revisao = "top"
    if "scroll_para_sorteio" not in st.session_state:
        st.session_state.scroll_para_sorteio = False
    if "scroll_para_confirmar_senha" not in st.session_state:
        st.session_state.scroll_para_confirmar_senha = False
    if "resultado_assinatura" not in st.session_state:
        st.session_state.resultado_assinatura = None
    if "resultado_invalidado_msg" not in st.session_state:
        st.session_state.resultado_invalidado_msg = False
    if "manual_section_visible" not in st.session_state:
        st.session_state.manual_section_visible = False

def abrir_expander_cadastro_manual():
    st.session_state.cadastro_manual_expanded = True
    st.session_state.manual_section_visible = True

def main():
    logic = PeladaLogic()
    st.title("⚽ Sorteador Pelada PRO")
    botao_instalar_app()

    init_session_state(logic)
    ensure_local_session_state()

    if "lista_texto_input" not in st.session_state:
        st.session_state.lista_texto_input = ""
    pending_lista_key = "lista_texto_input__pending"
    if pending_lista_key in st.session_state:
        st.session_state.lista_texto_input = st.session_state.pop(pending_lista_key)

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

    auto_revisar_lista = bool(st.session_state.pop("lista_texto_input__revisar", False))

    revisar_lista = False
    if not st.session_state.get("cadastro_guiado_ativo", False) and precisa_revisar_lista:
        render_step_cta_panel(
            "Revisar lista antes de continuar",
            "Confira os nomes reconhecidos, veja os ajustes automáticos e libere a próxima etapa do fluxo.",
            tone="info",
            eyebrow="Etapa atual",
        )
        revisar_lista = render_action_button(
            "🔎 Revisar lista",
            key="acao_revisar_lista",
            role="primary",
            use_primary_type=True,
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
