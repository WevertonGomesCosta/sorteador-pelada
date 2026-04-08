"""Painéis de resumo operacional antes do sorteio."""

import streamlit as st


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
