"""Renderização dos resultados do sorteio.

Concentra os painéis de status/resumo e os cards finais dos times,
mantendo o app principal mais enxuto e preparado para futuras melhorias
de tema e apresentação.
"""

from datetime import datetime

import html
import numpy as np
import streamlit as st

from ui.components import botao_compartilhar_js, botao_copiar_js


def render_sort_ready_panel(
    lista_revisada_ok: bool,
    lista_confirmada_ok: bool,
    base_pronta_ok: bool,
    sorteio_aleatorio_lista: bool = False,
):
    if sorteio_aleatorio_lista:
        linha_1 = "✅ Lista válida para sorteio"
        linha_2 = "⚠️ Sem base carregada"
        linha_3 = "🎲 Sorteio aleatório entre nomes únicos"
    else:
        linha_1 = f"{'✅' if lista_revisada_ok else '❌'} Lista revisada"
        linha_2 = f"{'✅' if lista_confirmada_ok else '❌'} Lista confirmada"
        linha_3 = f"{'✅' if base_pronta_ok else '❌'} Base pronta"

    st.markdown(
        f"""
        <div class="theme-panel theme-panel--status">
            <div class="theme-panel__title">Pronto para sortear?</div>
            <div class="theme-panel__line">{linha_1}</div>
            <div class="theme-panel__line">{linha_2}</div>
            <div class="theme-panel__line">{linha_3}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _contexto_resultado_sessao() -> dict:
    return dict(st.session_state.get("resultado_contexto", {}) or {})


def _diagnostico_lista_sessao() -> dict:
    return dict(st.session_state.get("diagnostico_lista", {}) or {})


def _resolver_status_parametros_resultado(
    *,
    sortear_capitao: bool | None,
    sortear_goleiros: bool | None,
    goleiros_incluidos: bool | None,
    qtd_goleiros_lidos: int | None,
) -> tuple[bool, bool, bool, int]:
    contexto = _contexto_resultado_sessao()
    diagnostico = _diagnostico_lista_sessao()

    if sortear_capitao is None:
        sortear_capitao = bool(
            contexto.get(
                "sortear_capitao",
                resultado_tem_capitao(st.session_state.get("resultado", [])),
            )
        )
    if sortear_goleiros is None:
        sortear_goleiros = bool(contexto.get("sortear_goleiros", diagnostico.get("sortear_goleiros", False)))
    if goleiros_incluidos is None:
        goleiros_incluidos = bool(contexto.get("goleiros_incluidos", diagnostico.get("goleiros_incluidos", False)))
    if qtd_goleiros_lidos is None:
        qtd_goleiros_lidos = int(contexto.get("qtd_goleiros_lidos", diagnostico.get("qtd_goleiros_lidos", 0)) or 0)

    return bool(sortear_capitao), bool(sortear_goleiros), bool(goleiros_incluidos), int(qtd_goleiros_lidos)


def render_result_summary_panel(
    qtd_jogadores_resultado: int,
    qtd_times_resultado: int,
    modo_criterios: str,
    criterios_ativos_texto: str,
    modo_sorteio: str = "balanceado",
    observacao_resultado: str = "",
    sortear_capitao: bool | None = None,
    sortear_goleiros: bool | None = None,
    goleiros_incluidos: bool | None = None,
    qtd_goleiros_lidos: int | None = None,
):
    sortear_capitao, sortear_goleiros, goleiros_incluidos, qtd_goleiros_lidos = _resolver_status_parametros_resultado(
        sortear_capitao=sortear_capitao,
        sortear_goleiros=sortear_goleiros,
        goleiros_incluidos=goleiros_incluidos,
        qtd_goleiros_lidos=qtd_goleiros_lidos,
    )
    titulo = "Detalhes do sorteio aleatório" if modo_sorteio == "aleatorio_lista" else "Detalhes do sorteio"
    linha_modo = "🎲 Aleatório por lista" if modo_sorteio == "aleatorio_lista" else "⚖️ Balanceado com base"
    observacao_html = (
        f"<div class='theme-panel__line'>ℹ️ <span class='theme-panel__label'>Observação:</span> <span class='theme-panel__strong'>{observacao_resultado}</span></div>"
        if observacao_resultado else ""
    )
    capitao_status = "Ativo" if sortear_capitao else "Desativado"
    if sortear_goleiros:
        goleiros_status = f"Ativo · {qtd_goleiros_lidos} lido(s)"
        if goleiros_incluidos:
            goleiros_status += " · incluídos"
        else:
            goleiros_status += " · não incluídos"
    else:
        goleiros_status = "Desativado"
    st.markdown(
        f"""
        <div class="theme-panel theme-panel--summary">
            <div class="theme-panel__title">{titulo}</div>
            <div class="theme-panel__line">🎯 <span class="theme-panel__label">Modo:</span> <span class="theme-panel__strong">{linha_modo}</span></div>
            <div class="theme-panel__line">👥 <span class="theme-panel__label">Jogadores:</span> <span class="theme-panel__strong">{qtd_jogadores_resultado}</span></div>
            <div class="theme-panel__line">🧩 <span class="theme-panel__label">Times:</span> <span class="theme-panel__strong">{qtd_times_resultado}</span></div>
            <div class="theme-panel__line">⚙️ <span class="theme-panel__label">Perfil:</span> <span class="theme-panel__strong">{modo_criterios}</span></div>
            <div class="theme-panel__line">✅ <span class="theme-panel__label">Equilíbrio usado:</span> <span class="theme-panel__strong">{criterios_ativos_texto}</span></div>
            <div class="theme-panel__line">(C) <span class="theme-panel__label">Capitão:</span> <span class="theme-panel__strong">{capitao_status}</span></div>
            <div class="theme-panel__line">🧤 <span class="theme-panel__label">Goleiros:</span> <span class="theme-panel__strong">{goleiros_status}</span></div>
            {observacao_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def jogador_eh_capitao(jogador) -> bool:
    return bool(isinstance(jogador, (list, tuple)) and len(jogador) >= 6 and jogador[5])


def resultado_tem_capitao(times) -> bool:
    return any(
        jogador_eh_capitao(jogador)
        for time in (times or [])
        for jogador in (time or [])
    )


def formatar_nome_jogador_resultado(jogador) -> str:
    nome = str(jogador[0]) if isinstance(jogador, (list, tuple)) and jogador else str(jogador)
    return f"{nome} (C)" if jogador_eh_capitao(jogador) else nome


def construir_texto_compartilhamento_resultado(*, times) -> str:
    linhas = []
    for i, time in enumerate(times):
        if not time:
            continue
        linhas.append(f"*Time {i+1}:*")
        for p in time:
            linhas.append(formatar_nome_jogador_resultado(p))
        linhas.append("")
    return "\n".join(linhas).strip() + "\n"


def _serializar_times_para_snapshot(times) -> list[list[list]]:
    times_snapshot = []
    for time in times or []:
        time_snapshot = []
        for jogador in time or []:
            if isinstance(jogador, (list, tuple)):
                linha = []
                for valor in jogador:
                    if hasattr(valor, "item"):
                        try:
                            valor = valor.item()
                        except Exception:
                            pass
                    linha.append(valor)
                time_snapshot.append(linha)
            else:
                time_snapshot.append([jogador])
        times_snapshot.append(time_snapshot)
    return times_snapshot


def _parametros_opcionais_para_snapshot(times) -> dict:
    contexto = _contexto_resultado_sessao()
    diagnostico = _diagnostico_lista_sessao()
    return {
        "sortear_capitao": bool(contexto.get("sortear_capitao", resultado_tem_capitao(times))),
        "sortear_goleiros": bool(contexto.get("sortear_goleiros", diagnostico.get("sortear_goleiros", False))),
        "goleiros_incluidos": bool(contexto.get("goleiros_incluidos", diagnostico.get("goleiros_incluidos", False))),
        "qtd_goleiros_lidos": int(contexto.get("qtd_goleiros_lidos", diagnostico.get("qtd_goleiros_lidos", 0)) or 0),
    }


def build_resultado_snapshot(
    *,
    times,
    odds,
    contexto_resultado: dict,
    cabecalho_padronizado: dict,
    texto_compartilhar: str,
    qtd_jogadores_resultado: int,
    qtd_times_resultado: int,
    modo_sorteio_resultado: str,
    modo_criterios: str,
    criterios_ativos_texto: str,
    observacao_resultado: str = "",
    resultado_assinatura: str | None = None,
) -> dict:
    contexto_snapshot = dict(contexto_resultado or {})
    contexto_snapshot.update(_parametros_opcionais_para_snapshot(times))
    timestamp_iso = contexto_snapshot.get("timestamp_sorteio_iso", "")
    snapshot_id = f"{timestamp_iso}::{resultado_assinatura or 'sem_assinatura'}"

    return {
        "snapshot_id": snapshot_id,
        "timestamp_iso": timestamp_iso,
        "timestamp_exibicao": cabecalho_padronizado.get("data_hora_exibicao", ""),
        "titulo_curto": cabecalho_padronizado.get("titulo", "Resultado do sorteio"),
        "resumo_curto": cabecalho_padronizado.get("cabecalho_curto", ""),
        "texto_compartilhar": texto_compartilhar,
        "payload_resultado": {
            "times": _serializar_times_para_snapshot(times),
            "odds": [float(x) if hasattr(x, "item") or isinstance(x, (int, float)) else x for x in (odds or [])],
            "contexto_resultado": contexto_snapshot,
            "qtd_jogadores_resultado": qtd_jogadores_resultado,
            "qtd_times_resultado": qtd_times_resultado,
            "modo_sorteio_resultado": modo_sorteio_resultado,
            "modo_criterios": modo_criterios,
            "criterios_ativos_texto": criterios_ativos_texto,
            "observacao_resultado": observacao_resultado,
            "cabecalho_padronizado": dict(cabecalho_padronizado or {}),
        },
    }


def registrar_snapshot_resultado_na_sessao(
    *,
    snapshot: dict,
    historico_atual: list[dict] | None,
    ultimo_snapshot_id: str | None,
    max_itens: int = 5,
) -> tuple[list[dict], str]:
    snapshot_id = snapshot.get("snapshot_id")

    if not snapshot_id:
        return list(historico_atual or []), str(ultimo_snapshot_id or "")

    if snapshot_id == ultimo_snapshot_id:
        return list(historico_atual or []), snapshot_id

    historico = [
        dict(item)
        for item in (historico_atual or [])
        if item.get("snapshot_id") != snapshot_id
    ]
    historico.insert(0, snapshot)
    historico = historico[:max_itens]

    return historico, snapshot_id


def obter_snapshot_resultado_por_id(
    historico: list[dict] | None,
    snapshot_id: str | None,
) -> dict | None:
    if not historico or not snapshot_id:
        return None

    for item in historico:
        if item.get("snapshot_id") == snapshot_id:
            return item
    return None


def render_historico_resultados_sessao(
    historico: list[dict] | None,
    *,
    snapshot_ativo_id: str | None = None,
    max_itens_visiveis: int = 3,
) -> str | None:
    if not historico or len(historico) <= 1:
        return None

    itens_visiveis = historico[1 : 1 + max_itens_visiveis]
    if not itens_visiveis:
        return None

    st.markdown("### Últimos sorteios desta sessão")
    st.caption("Resultados anteriores gerados nesta sessão.")

    selected_snapshot_id = None

    for idx, item in enumerate(itens_visiveis):
        snapshot_id = item.get("snapshot_id")
        horario = str(item.get("timestamp_exibicao") or item.get("timestamp_iso") or "Sem horário").strip()
        titulo = str(item.get("titulo_curto") or "Resultado do sorteio").strip()
        resumo = str(item.get("resumo_curto") or "").strip()

        col_info, col_action = st.columns([5, 1])
        with col_info:
            st.markdown(f"**{horario} · {titulo}**")
            if resumo:
                st.caption(resumo)
            if snapshot_id == snapshot_ativo_id:
                st.caption("Em exibição")
        with col_action:
            if st.button(
                "Ver resultado",
                key=f"historico_resultado_{snapshot_id}",
                use_container_width=True,
            ):
                selected_snapshot_id = snapshot_id

        if idx < len(itens_visiveis) - 1:
            st.divider()

    return selected_snapshot_id


def render_acoes_resultado(texto_copiar: str):
    col_copy, col_share = st.columns(2)
    with col_copy:
        botao_copiar_js(texto_copiar)
    with col_share:
        botao_compartilhar_js(texto_copiar)


def ordenar_jogadores_do_time(time):
    ordem = {'G': 0, 'D': 1, 'M': 2, 'A': 3}
    time.sort(key=lambda x: (ordem.get(x[2], 99), x[0]))
    return time


def montar_html_jogadores_do_time(time) -> str:
    rows = ""
    metrica_disponivel = all(
        len(p) >= 5 and p[1] is not None and p[3] is not None and p[4] is not None
        for p in time
    )

    for p in time:
        nome_html = html.escape(formatar_nome_jogador_resultado(p))
        pos_html = f" <span class='team-card__player-pos'>{p[2]}</span>" if len(p) >= 3 and p[2] else ""
        if metrica_disponivel:
            metricas_html = (
                "<div class='team-card__metrics'>"
                f"<span class='team-card__metric--star'>⭐{p[1]:.1f}</span> "
                f"<span class='team-card__metric--speed'>⚡{p[3]:.1f}</span> "
                f"<span class='team-card__metric--move'>🔄{p[4]:.1f}</span>"
                "</div>"
            )
        else:
            metricas_html = ""

        rows += (
            "<div class='team-card__player-row'>"
            f"<div class='team-card__player-main'><span class='team-card__player-name'>{nome_html}</span>{pos_html}</div>"
            f"{metricas_html}</div>"
        )
    return rows


def render_team_cards(times, odds):
    for i, time in enumerate(times):
        if not time:
            continue

        ordenar_jogadores_do_time(time)
        metricas_disponiveis = all(
            len(p) >= 5 and p[1] is not None and p[3] is not None and p[4] is not None
            for p in time
        )
        rows = montar_html_jogadores_do_time(time)

        if metricas_disponiveis:
            m_nota = np.mean([p[1] for p in time])
            m_vel = np.mean([p[3] for p in time])
            m_mov = np.mean([p[4] for p in time])
            stats_html = (
                "<div class='team-card__stats'>"
                f"<span>⭐ <b>{m_nota:.1f}</b></span>"
                f"<span>⚡ <b>{m_vel:.1f}</b></span>"
                f"<span>🔄 <b>{m_mov:.1f}</b></span>"
                "</div>"
            )
        else:
            stats_html = "<div class='team-card__stats'><span>Distribuição aleatória pela lista</span></div>"

        odd_val = odds[i] if i < len(odds) else None
        odd_html = f"<span class='team-card__odd'>Odd: {odd_val:.2f}</span>" if odd_val is not None else "<span class='team-card__odd'>Modo aleatório</span>"

        card_html = (
            "<div class='team-card'>"
            "<div class='team-card__header'>"
            f"<h3 class='team-card__title'>TIME {i+1}</h3>"
            f"{odd_html}"
            "</div>"
            f"{stats_html}"
            f"{rows}"
            "</div>"
        )

        st.markdown(card_html, unsafe_allow_html=True)
