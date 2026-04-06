"""Renderização dos resultados do sorteio.

Concentra os painéis de status/resumo e os cards finais dos times,
mantendo o app principal mais enxuto e preparado para futuras melhorias
de tema e apresentação.
"""

import numpy as np
import streamlit as st


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


def render_result_summary_panel(
    qtd_jogadores_resultado: int,
    qtd_times_resultado: int,
    modo_criterios: str,
    criterios_ativos_texto: str,
    modo_sorteio: str = "balanceado",
    observacao_resultado: str = "",
):
    titulo = "Resumo do sorteio aleatório" if modo_sorteio == "aleatorio_lista" else "Resumo do sorteio"
    linha_modo = "🎲 Aleatório por lista" if modo_sorteio == "aleatorio_lista" else "⚖️ Balanceado com base"
    observacao_html = (
        f"<div class='theme-panel__line'>ℹ️ <span class='theme-panel__label'>Observação:</span> <span class='theme-panel__strong'>{observacao_resultado}</span></div>"
        if observacao_resultado else ""
    )
    st.markdown(
        f"""
        <div class="theme-panel theme-panel--summary">
            <div class="theme-panel__title">{titulo}</div>
            <div class="theme-panel__line">🎯 <span class="theme-panel__label">Modo:</span> <span class="theme-panel__strong">{linha_modo}</span></div>
            <div class="theme-panel__line">👥 <span class="theme-panel__label">Jogadores:</span> <span class="theme-panel__strong">{qtd_jogadores_resultado}</span></div>
            <div class="theme-panel__line">🧩 <span class="theme-panel__label">Times:</span> <span class="theme-panel__strong">{qtd_times_resultado}</span></div>
            <div class="theme-panel__line">⚙️ <span class="theme-panel__label">Critérios:</span> <span class="theme-panel__strong">{modo_criterios}</span></div>
            <div class="theme-panel__line">✅ <span class="theme-panel__label">Ativos:</span> <span class="theme-panel__strong">{criterios_ativos_texto}</span></div>
            {observacao_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


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
            f"<div class='team-card__player-main'><span class='team-card__player-name'>{p[0]}</span>{pos_html}</div>"
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
            stats_html = f"""
                <div class='team-card__stats'>
                    <span>⭐ <b>{m_nota:.1f}</b></span>
                    <span>⚡ <b>{m_vel:.1f}</b></span>
                    <span>🔄 <b>{m_mov:.1f}</b></span>
                </div>
            """
        else:
            stats_html = "<div class='team-card__stats'><span>Sorteio aleatório pela lista</span></div>"

        odd_val = odds[i] if i < len(odds) else None
        odd_html = f"<span class='team-card__odd'>Odd: {odd_val:.2f}</span>" if odd_val is not None else "<span class='team-card__odd'>Aleatório</span>"

        st.markdown(
            f"""
            <div class='team-card'>
                <div class='team-card__header'>
                    <h3 class='team-card__title'>TIME {i+1}</h3>
                    {odd_html}
                </div>
                {stats_html}
                {rows}
            </div>
            """,
            unsafe_allow_html=True,
        )
