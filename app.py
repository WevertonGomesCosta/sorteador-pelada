import streamlit as st
import pandas as pd
import numpy as np

from core.logic import PeladaLogic
from state.session import init_session_state
from ui.components import botao_copiar_js, botao_instalar_app

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
    .block-container { padding-top: 1.15rem; padding-bottom: 3rem; }
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

    h1 {
        margin-top: 0.1rem !important;
        margin-bottom: 0.2rem !important;
        line-height: 1.05 !important;
    }

    #install-app-container {
        margin: 0.15rem 0 0.12rem 0 !important;
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


def render_group_config_expander(logic, nome_pelada_adm: str, senha_adm: str) -> str:
    with st.expander("⚙️ Configuração do grupo e base de dados", expanded=False):
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

        if grupo_admin:
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

        st.markdown("---")
        st.markdown("**📂 Banco de dados**")
        st.caption("Escolha como carregar sua base ou siga para a etapa 3.")

        df_exemplo = logic.criar_exemplo()
        excel_exemplo = logic.converter_df_para_excel(df_exemplo)
        st.download_button(
            label="📥 Baixar planilha modelo",
            data=excel_exemplo,
            file_name="modelo_pelada.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Baixe este arquivo para ver como preencher a planilha no formato correto.",
            key="grupo_baixar_modelo_planilha",
        )

        if origem_base == "Base original (Admin)":
            senha = st.text_input(
                "Senha de Acesso:",
                type="password",
                key="grupo_senha_admin",
            )
            st.caption("Depois de informar a senha, clique em **Carregar base de dados**.")
        else:
            st.write("Já tem uma planilha? Envie o arquivo abaixo e depois clique em **Carregar base de dados**.")
            uploaded_file = st.file_uploader(
                "Enviar planilha Excel",
                type=["xlsx"],
                label_visibility="collapsed",
                key="grupo_upload_planilha",
            )

        if st.button("📥 Carregar base de dados", key="grupo_carregar_base"):
            if origem_base == "Base original (Admin)":
                if not grupo_admin:
                    if nome_informado:
                        st.error(
                            "Base não encontrada para esse nome. Corrija o nome, envie uma planilha própria ou siga para a etapa 3."
                        )
                    else:
                        st.warning(
                            "Informe um grupo válido para usar a base administrada ou siga para a etapa 3."
                        )
                elif senha != str(senha_adm):
                    st.session_state.is_admin = False
                    st.error("Senha incorreta")
                else:
                    st.session_state.df_base = logic.carregar_dados_originais()
                    st.session_state.novos_jogadores = []
                    st.session_state.is_admin = True
                    st.session_state.ultimo_arquivo = None
                    st.success(f"Base carregada: {len(st.session_state.df_base)} jogadores.")
                    st.rerun()
            else:
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
                        st.session_state.df_base = df_novo
                        st.session_state.novos_jogadores = []
                        st.session_state.is_admin = False
                        st.session_state.ultimo_arquivo = uploaded_file.name
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
                    st.session_state.ultimo_arquivo = None
                    st.rerun()

    return nome_pelada


def render_manual_card(logic, nome_pelada: str):
    with st.expander("📝 Adicionar jogadores manualmente", expanded=False):
        st.caption(
            "Use esta etapa para montar sua base do zero ou complementar a base atual com novos jogadores."
        )

        with st.form("form_add_manual"):
            col_a, col_b = st.columns(2)
            nome_m = col_a.text_input("Nome")
            p_m = col_b.selectbox("Posição", ["M", "A", "D"])
            n_m = st.slider("Nota", 1.0, 10.0, 6.0, 0.5)
            v_m = st.slider("Velocidade", 1, 5, 3)
            mv_m = st.slider("Movimentação", 1, 5, 3)
            if st.form_submit_button("Adicionar à Base"):
                if nome_m:
                    novo_nome = logic.formatar_nome_visual(nome_m)
                    novo = {
                        'Nome': novo_nome,
                        'Nota': n_m,
                        'Posição': p_m,
                        'Velocidade': v_m,
                        'Movimentação': mv_m,
                    }
                    st.session_state.df_base.loc[len(st.session_state.df_base)] = novo
                    st.success(f"{novo_nome} salvo!")
                else:
                    st.error("Digite um nome.")

        st.markdown("---")
        if not st.session_state.df_base.empty:
            st.caption("Baixe a planilha atual com os jogadores já adicionados à base.")
            if st.session_state.is_admin:
                st.info("🔒 O download da Base Mestra é bloqueado por segurança.")
            else:
                nome_arquivo = nome_pelada.strip()
                if not nome_arquivo:
                    nome_arquivo = "minha_pelada"
                if not nome_arquivo.endswith(".xlsx"):
                    nome_arquivo += ".xlsx"
                excel_data = logic.converter_df_para_excel(st.session_state.df_base)
                st.download_button(
                    label="💾 Baixar Minha Planilha",
                    data=excel_data,
                    file_name=nome_arquivo,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            if not st.session_state.is_admin:
                st.info("Sem base carregada? Você pode adicionar jogadores aqui e montar sua base manualmente.")


def render_base_preview():
    df_base = st.session_state.df_base

    if df_base.empty:
        return

    render_section_header(
        "Prévia da base atual",
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
        "1. Configuração do grupo e base de dados",
        "Escolha como iniciar sua base: usar a base do grupo, enviar uma planilha própria ou seguir para a etapa 3."
    )
    nome_pelada = render_group_config_expander(logic, NOME_PELADA_ADM, SENHA_ADM)

    render_section_header(
        "2. Base de jogadores",
        "Confira a base atual. Se ela estiver vazia, siga pela etapa 3 para cadastrar jogadores manualmente."
    )
    render_base_summary()

    render_section_header(
        "3. Adicionar jogadores manualmente",
        "Use esta etapa para montar sua base do zero ou complementar a base atual com novos jogadores."
    )
    render_manual_card(logic, nome_pelada)

    render_base_preview()

    render_section_header(
        "4. Lista da pelada",
        "Cole aqui os nomes confirmados para o sorteio. Eles serão comparados com a base carregada e, se necessário, você poderá completar os jogadores manualmente."
    )
    st.markdown(f"**Modo:** {'🔐 ADMIN (Download Bloqueado)' if st.session_state.is_admin else '👤 Público (Base Própria)'}")
    lista_texto = st.text_area("Cole a lista (Numerada ou não):", height=120, placeholder="1. Jogador A\n2. Jogador B...")
    col1, col2 = st.columns(2)
    n_times = col1.selectbox("Nº Times:", range(2, 11), index=1)

    render_section_header(
        "5. Critérios do sorteio",
        "Escolha quais características devem ser equilibradas entre os times."
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
        render_section_header(
            "7. Resultado",
            "Veja os times sorteados e copie o resultado para compartilhar."
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
