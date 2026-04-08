"""Funções puras para interpretação do estado visual do app.

Estas funções não renderizam componentes e não escrevem em ``st.session_state``.
Elas centralizam a leitura do estado visual do fluxo para reduzir condicionais
espalhadas em ``app.py``.
"""

from __future__ import annotations


def determinar_visibilidade_revisao(
    *,
    diagnostico_disponivel: bool,
    lista_confirmada: bool,
    cadastro_guiado_ativo: bool,
    revisao_pendente_pos_cadastro: bool,
    faltantes_revisao_qtd: int,
    faltantes_cadastrados_qtd: int,
) -> bool:
    return bool(
        diagnostico_disponivel
        or lista_confirmada
        or cadastro_guiado_ativo
        or revisao_pendente_pos_cadastro
        or faltantes_revisao_qtd > 0
        or faltantes_cadastrados_qtd > 0
    )



def determinar_etapa_visual_ativa(
    *,
    escolha_inicial_pendente: bool,
    qtd_nomes: int,
    draft_lista: str,
    lista_confirmada: bool,
    resultado_disponivel: bool,
    review_stage_visible: bool,
    manual_section_visible: bool,
    cadastro_guiado_ativo: bool,
    revisao_pendente_pos_cadastro: bool,
) -> str:
    if escolha_inicial_pendente:
        return "config"
    if resultado_disponivel:
        return "resultado"
    if lista_confirmada:
        return "sorteio"
    if cadastro_guiado_ativo or revisao_pendente_pos_cadastro or review_stage_visible:
        return "revisao"
    if manual_section_visible and qtd_nomes == 0:
        return "cadastro_manual"
    if qtd_nomes > 0 or str(draft_lista or "").strip():
        return "lista"
    return "config"



def construir_status_sessao_visual(
    *,
    origem_fluxo: str | None,
    base_carregada_via_secao1: bool,
    qtd_nomes: int,
    qtd_ignorados: int,
    diagnostico: dict,
    lista_revisada_ok: bool,
    lista_confirmada_ok: bool,
    base_pronta_ok: bool,
    resultado_disponivel: bool,
    cadastro_guiado_ativo: bool,
    is_admin: bool,
    ultimo_arquivo: str,
    df_base_len: int,
    novos_jogadores_len: int,
) -> dict[str, object]:
    fluxo_somente_lista = bool(
        origem_fluxo == "lista" and not base_carregada_via_secao1
    )
    escolha_inicial_pendente = bool(
        origem_fluxo is None
        and not base_carregada_via_secao1
        and qtd_nomes == 0
        and not cadastro_guiado_ativo
    )

    if cadastro_guiado_ativo:
        fluxo_status = "Cadastro guiado em andamento"
    elif escolha_inicial_pendente:
        fluxo_status = "Escolha como iniciar"
    elif qtd_nomes == 0:
        fluxo_status = "Aguardando lista"
    elif not lista_revisada_ok:
        fluxo_status = "Revisão pendente"
    elif diagnostico.get("tem_nao_encontrados", False):
        fluxo_status = "Faltantes pendentes"
    elif diagnostico.get("tem_duplicados", False):
        fluxo_status = "Nomes repetidos na lista"
    elif diagnostico.get("tem_bloqueio_base", False):
        fluxo_status = "Base com bloqueios"
    elif not lista_confirmada_ok:
        fluxo_status = "Confirmação pendente"
    elif resultado_disponivel:
        fluxo_status = "Resultado disponível"
    else:
        fluxo_status = "Pronto para sortear"

    if is_admin:
        modo_atual = "Base do grupo"
    elif bool(ultimo_arquivo):
        modo_atual = "Excel próprio"
    elif fluxo_somente_lista:
        modo_atual = "Apenas sorteio com lista"
    elif escolha_inicial_pendente:
        modo_atual = "Escolha inicial"
    else:
        modo_atual = "Público / base própria"

    if base_carregada_via_secao1 and is_admin:
        base_status = f"Base do grupo carregada · {df_base_len} jogador(es)"
    elif bool(ultimo_arquivo):
        base_status = f"Excel próprio carregado · {df_base_len} jogador(es)"
    elif base_pronta_ok:
        qtd_base_total = df_base_len + novos_jogadores_len
        base_status = f"Base disponível · {qtd_base_total} jogador(es)"
    else:
        base_status = "Nenhuma base carregada"

    if qtd_nomes == 0:
        lista_status = "Nenhuma lista informada"
    else:
        lista_status = f"{qtd_nomes} nome(s) reconhecido(s)"
        if qtd_ignorados > 0:
            lista_status += f" · {qtd_ignorados} linha(s) ignorada(s)"
        if lista_confirmada_ok:
            lista_status += " · confirmada"
        elif lista_revisada_ok:
            lista_status += " · revisada"

    if escolha_inicial_pendente:
        proxima_acao = "Escolha como iniciar: apenas lista, base do grupo ou Excel próprio"
    elif qtd_nomes == 0:
        proxima_acao = "Cole a lista de jogadores"
    elif cadastro_guiado_ativo:
        proxima_acao = "Concluir cadastro guiado dos faltantes"
    elif not lista_revisada_ok:
        proxima_acao = "Corrigir a lista na etapa de revisão"
    elif diagnostico.get("tem_nao_encontrados", False):
        proxima_acao = "Cadastrar faltantes na revisão"
    elif diagnostico.get("tem_duplicados", False):
        proxima_acao = "Corrigir nomes repetidos na revisão"
    elif diagnostico.get("tem_bloqueio_base", False):
        proxima_acao = "Corrigir inconsistências da base"
    elif not lista_confirmada_ok:
        proxima_acao = "Clicar em ✅ Confirmar lista final"
    elif resultado_disponivel:
        proxima_acao = "Copiar, compartilhar ou ajustar e sortear novamente"
    else:
        proxima_acao = "Clicar em 🎲 SORTEAR TIMES"

    return {
        "fluxo_somente_lista": fluxo_somente_lista,
        "escolha_inicial_pendente": escolha_inicial_pendente,
        "modo_atual": modo_atual,
        "base_status": base_status,
        "lista_status": lista_status,
        "fluxo_status": fluxo_status,
        "proxima_acao": proxima_acao,
    }



def construir_estado_blocos_visuais(
    *,
    etapa_visual_ativa: str,
    scroll_para_revisao: bool,
    cadastro_guiado_ativo: bool,
    revisao_pendente_pos_cadastro: bool,
    cadastro_manual_nome_existente: str,
) -> dict[str, bool]:
    grupo_config_expanded = etapa_visual_ativa == "config"
    cadastro_manual_expanded = bool(
        etapa_visual_ativa == "cadastro_manual"
        or cadastro_guiado_ativo
        or revisao_pendente_pos_cadastro
        or cadastro_manual_nome_existente
    )
    review_stage_active_ui = etapa_visual_ativa == "revisao"
    atualiza_revisao_lista_expandida = not scroll_para_revisao and not cadastro_guiado_ativo
    revisao_lista_expandida = etapa_visual_ativa == "revisao"

    return {
        "grupo_config_expanded": grupo_config_expanded,
        "cadastro_manual_expanded": cadastro_manual_expanded,
        "review_stage_active_ui": review_stage_active_ui,
        "atualiza_revisao_lista_expandida": atualiza_revisao_lista_expandida,
        "revisao_lista_expandida": revisao_lista_expandida,
    }
