import streamlit as st
import pandas as pd
import numpy as np

from core.logic import PeladaLogic
from state.session import init_session_state
from ui.components import botao_copiar_js, botao_instalar_app
from ui.manual_card import render_manual_card
from ui.sidebar import render_sidebar

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
st.markdown("""
    <style>
    .stButton>button {
        width: 100%; height: 3.5em; font-weight: bold;
        background-color: #ff4b4b; color: white; border-radius: 8px; border: none;
    }
    .stButton>button:hover { background-color: #ff3333; }
    .stTextArea textarea { font-size: 16px; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    .stAlert { font-weight: bold; }

    .section-title {
        margin-top: 1.2rem;
        margin-bottom: 0.45rem;
        font-size: 1.08rem;
        font-weight: 700;
        color: #f3f4f6;
    }

    .section-subtitle {
        margin-top: -0.10rem;
        margin-bottom: 0.85rem;
        font-size: 0.93rem;
        color: #cbd5e1;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 0.5rem 0 1rem 0;
    }

    .summary-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.96) 0%, rgba(17, 24, 39, 0.92) 100%);
        border: 1px solid #253247;
        border-top: 3px solid #22c55e;
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.16);
    }

    .summary-label {
        font-size: 0.76rem;
        color: #93c5fd;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .summary-value {
        font-size: 1.2rem;
        font-weight: 800;
        color: #f8fafc;
    }

    @media (max-width: 900px) {
        .summary-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    </style>
""", unsafe_allow_html=True)


def render_section_header(titulo: str, subtitulo: str | None = None):
    st.markdown(f"<div class='section-title'>{titulo}</div>", unsafe_allow_html=True)
    if subtitulo:
        st.markdown(f"<div class='section-subtitle'>{subtitulo}</div>", unsafe_allow_html=True)


def render_base_summary():
    df_base = st.session_state.df_base

    qtd_jogadores = len(df_base)

    if st.session_state.is_admin:
        origem = "Admin"
    elif qtd_jogadores == 0:
        origem = "Vazia"
    else:
        origem = "Sua base"

    modo = "ADMIN" if st.session_state.is_admin else "Público"

    if df_base.empty:
        posicoes = "—"
    else:
        cont_pos = df_base["Posição"].value_counts()
        posicoes = " / ".join(
            [
                f"D {cont_pos.get('D', 0)}",
                f"M {cont_pos.get('M', 0)}",
                f"A {cont_pos.get('A', 0)}",
            ]
        )

    st.markdown(
        f"""
        <div class=\"summary-grid\">
            <div class=\"summary-card\">
                <div class=\"summary-label\">⚽ Modo</div>
                <div class=\"summary-value\">{modo}</div>
            </div>
            <div class=\"summary-card\">
                <div class=\"summary-label\">👥 Jogadores</div>
                <div class=\"summary-value\">{qtd_jogadores} jogadores</div>
            </div>
            <div class=\"summary-card\">
                <div class=\"summary-label\">📋 Base</div>
                <div class=\"summary-value\">{origem}</div>
            </div>
            <div class=\"summary-card\">
                <div class=\"summary-label\">🧩 D / M / A</div>
                <div class=\"summary-value\">{posicoes}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_base_preview():
    df_base = st.session_state.df_base

    if df_base.empty:
        return

    render_section_header(
        "2. Prévia da base atual",
        "Confira rapidamente os jogadores atualmente disponíveis para o sorteio."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        ordenar_por = st.selectbox(
            "Ordenar por",
            ["Nome", "Posição", "Nota"],
            key="preview_ordenar_por"
        )
    with col2:
        max_linhas = st.selectbox(
            "Mostrar",
            [10, 20, 50, 100],
            index=1,
            key="preview_max_linhas"
        )

    ascending = True
    if ordenar_por == "Nota":
        ascending = False

    df_preview = df_base.sort_values(by=ordenar_por, ascending=ascending).reset_index(drop=True)

    st.dataframe(
        df_preview.head(max_linhas),
        width="stretch",
        hide_index=True
    )

# --- FRONTEND ---
def main():
    logic = PeladaLogic()
    st.title("⚽ Sorteador Pelada PRO")
    botao_instalar_app()

    init_session_state(logic)

    render_section_header(
        "1. Base de jogadores",
        "Carregue sua base pela sidebar, use a base admin ou complemente manualmente."
    )
    render_base_summary()

    # --- SIDEBAR ---
    nome_pelada = render_sidebar(logic, NOME_PELADA_ADM, SENHA_ADM)

    # --- CADASTRO MANUAL ---
    render_manual_card(logic, nome_pelada)

    render_base_preview()

    # --- INPUT PRINCIPAL ---
    render_section_header(
        "3. Lista da pelada",
        "Cole os nomes confirmados para montar os times."
    )
    st.markdown(f"**Modo:** {'🔐 ADMIN (Download Bloqueado)' if st.session_state.is_admin else '👤 Público (Base Própria)'}")
    lista_texto = st.text_area("Cole a lista (Numerada ou não):", height=120, placeholder="1. Jogador A\n2. Jogador B...")
    col1, col2 = st.columns(2)
    n_times = col1.selectbox("Nº Times:", range(2, 11), index=1)

    render_section_header(
        "4. Critérios do sorteio",
        "Escolha quais dimensões devem ser equilibradas entre os times."
    )
    with st.expander("⚙️ Critérios", expanded=False):
        c_pos = st.checkbox("Equilibrar Posição", value=True)
        c_nota = st.checkbox("Equilibrar Nota", value=True)
        c_vel = st.checkbox("Equilibrar Velocidade", value=True)
        c_mov = st.checkbox("Equilibrar Movimentação", value=True)

    if st.button("🎲 SORTEAR TIMES"):
        nomes_brutos = logic.processar_lista(lista_texto)
        if not nomes_brutos:
            st.warning("Lista vazia!")
            st.stop()

        nomes_corrigidos = logic.corrigir_nomes_pela_base(nomes_brutos, st.session_state.df_base)

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

            df_jogar = df_final[df_final['Nome'].isin(nomes_corrigidos)].drop_duplicates(subset=['Nome'], keep='last')

            try:
                with st.spinner('Sorteando...'):
                    times = logic.otimizar(df_jogar, n_times, {'pos': c_pos, 'nota': c_nota, 'vel': c_vel, 'mov': c_mov})
                    st.session_state.resultado = times
            except Exception as e:
                st.error(f"Erro: {e}")

    if st.session_state.get('aviso_sem_planilha'):
        st.warning("⚠️ NENHUMA PLANILHA DETECTADA!")
        st.markdown(f"""
        Você não carregou a base Admin e nem fez Upload de uma planilha própria.

        Isso significa que você terá que **adicionar notas manualmente para todos os {len(st.session_state.nomes_pendentes)} jogadores** da lista.
        """)

        col_conf1, col_conf2 = st.columns(2)
        if col_conf1.button("✅ Sim, quero cadastrar manualmente"):
            st.session_state.faltantes_temp = st.session_state.nomes_pendentes
            st.session_state.aviso_sem_planilha = False
            st.rerun()

        if col_conf2.button("❌ Não, vou carregar a planilha"):
            st.session_state.aviso_sem_planilha = False
            st.rerun()

    if 'faltantes_temp' in st.session_state and st.session_state.faltantes_temp:
        nome_atual = st.session_state.faltantes_temp[0]
        total_f = len(st.session_state.faltantes_temp) + len(st.session_state.novos_jogadores)
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
        render_section_header(
            "5. Resultado",
            "Veja os times gerados e copie rapidamente o resultado para compartilhar."
        )
        times = st.session_state.resultado
        odds = logic.calcular_odds(times)
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

        for i, time in enumerate(times):
            if not time:
                continue
            ordem = {'G': 0, 'D': 1, 'M': 2, 'A': 3}
            time.sort(key=lambda x: (ordem.get(x[2], 99), x[0]))
            m_nota = np.mean([p[1] for p in time])
            m_vel = np.mean([p[3] for p in time])
            m_mov = np.mean([p[4] for p in time])
            rows = ""
            for p in time:
                rows += f"<div style='display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #eee;'><div><span style='font-weight:bold; color:black'>{p[0]}</span> <span style='font-size:12px; background:#eee; padding:2px 5px; border-radius:4px; color:#333'>{p[2]}</span></div><div style='font-family:monospace; font-size:14px'><span style='color:#d39e00'>⭐{p[1]:.1f}</span> <span style='color:#0056b3'>⚡{p[3]:.1f}</span> <span style='color:#28a745'>🔄{p[4]:.1f}</span></div></div>"
            st.markdown(f"<div style='background:white; padding:15px; border-radius:10px; margin-bottom:20px; border:1px solid #ddd; box-shadow:0 2px 5px rgba(0,0,0,0.1);'><div style='display:flex; justify-content:space-between; margin-bottom:10px; border-bottom:2px solid #333; padding-bottom:10px;'><h3 style='margin:0; color:black'>TIME {i+1}</h3><span style='background:#ffc107; padding:2px 8px; border-radius:10px; font-weight:bold; color:black'>Odd: {odds[i]:.2f}</span></div><div style='background:#f8f9fa; padding:8px; border-radius:8px; display:flex; justify-content:space-around; color:#333; margin-bottom:10px;'><span>⭐ <b>{m_nota:.1f}</b></span><span>⚡ <b>{m_vel:.1f}</b></span><span>🔄 <b>{m_mov:.1f}</b></span></div>{rows}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
