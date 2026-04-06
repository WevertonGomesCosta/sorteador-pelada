"""Renderização dos resultados do sorteio.

Concentra os painéis de status/resumo e os cards finais dos times,
mantendo o app principal mais enxuto e preparado para futuras melhorias
de tema e apresentação.
"""

import numpy as np
import streamlit as st


def render_sort_ready_panel(lista_revisada_ok: bool, lista_confirmada_ok: bool, base_pronta_ok: bool):
    st.markdown(
        f"""
        <div class="theme-panel theme-panel--status">
            <div class="theme-panel__title">Pronto para sortear?</div>
            <div class="theme-panel__line">{"✅" if lista_revisada_ok else "❌"} Lista revisada</div>
            <div class="theme-panel__line">{"✅" if lista_confirmada_ok else "❌"} Lista confirmada</div>
            <div class="theme-panel__line">{"✅" if base_pronta_ok else "❌"} Base pronta</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_summary_panel(
    qtd_jogadores_resultado: int,
    qtd_times_resultado: int,
    modo_criterios: str,
    criterios_ativos_texto: str,
):
    st.markdown(
        f"""
        <div class="theme-panel theme-panel--summary">
            <div class="theme-panel__title">Resumo do sorteio</div>
            <div class="theme-panel__line">👥 <span class="theme-panel__label">Jogadores:</span> <span class="theme-panel__strong">{qtd_jogadores_resultado}</span></div>
            <div class="theme-panel__line">🧩 <span class="theme-panel__label">Times:</span> <span class="theme-panel__strong">{qtd_times_resultado}</span></div>
            <div class="theme-panel__line">⚙️ <span class="theme-panel__label">Critérios:</span> <span class="theme-panel__strong">{modo_criterios}</span></div>
            <div class="theme-panel__line">✅ <span class="theme-panel__label">Ativos:</span> <span class="theme-panel__strong">{criterios_ativos_texto}</span></div>
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
    for p in time:
        rows += (
            "<div class='team-card__player-row'>"
            f"<div class='team-card__player-main'><span class='team-card__player-name'>{p[0]}</span> "
            f"<span class='team-card__player-pos'>{p[2]}</span></div>"
            "<div class='team-card__metrics'>"
            f"<span class='team-card__metric--star'>⭐{p[1]:.1f}</span> "
            f"<span class='team-card__metric--speed'>⚡{p[3]:.1f}</span> "
            f"<span class='team-card__metric--move'>🔄{p[4]:.1f}</span>"
            "</div></div>"
        )
    return rows


def render_team_cards(times, odds):
    for i, time in enumerate(times):
        if not time:
            continue

        ordenar_jogadores_do_time(time)
        m_nota = np.mean([p[1] for p in time])
        m_vel = np.mean([p[3] for p in time])
        m_mov = np.mean([p[4] for p in time])
        rows = montar_html_jogadores_do_time(time)

        st.markdown(
            f"""
            <div class='team-card'>
                <div class='team-card__header'>
                    <h3 class='team-card__title'>TIME {i+1}</h3>
                    <span class='team-card__odd'>Odd: {odds[i]:.2f}</span>
                </div>
                <div class='team-card__stats'>
                    <span>⭐ <b>{m_nota:.1f}</b></span>
                    <span>⚡ <b>{m_vel:.1f}</b></span>
                    <span>🔄 <b>{m_mov:.1f}</b></span>
                </div>
                {rows}
            </div>
            """,
            unsafe_allow_html=True,
        )
