import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
import numpy as np
import random
import pulp
import json
import io

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Sorteador Pelada PRO",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- SEGREDOS (VIA ST.SECRETS) ---
# Certifique-se de criar o arquivo .streamlit/secrets.toml localmente
# ou configurar os Secrets no painel do Streamlit Cloud.
try:
    NOME_PELADA_ADM = st.secrets["nome_admin"]
    SENHA_ADM = st.secrets["senha_admin"]
except Exception:
    # Fallback apenas para evitar erro se o arquivo não existir na primeira execução local
    # O ideal é não ter isso em produção
    NOME_PELADA_ADM = "QUARTA 18:30" 
    SENHA_ADM = "1234"
    # st.warning("⚠️ Usando credenciais padrão. Configure o secrets.toml para segurança.")

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
    </style>
""", unsafe_allow_html=True)

# --- LÓGICA (BACKEND) ---
class PeladaLogic:
    def __init__(self):
        self.url_padrao = "https://docs.google.com/spreadsheets/d/1a9qMYNWI_9l4AhQG9whGl11C-ZsXFFZivBKRoD5GyPU/export?format=xlsx"

    def criar_base_vazia(self):
        return pd.DataFrame(columns=["Nome", "Nota", "Posição", "Velocidade", "Movimentação"])

    def criar_exemplo(self):
        dados_exemplo = [
            {"Nome": "Exemplo Atacante", "Nota": 8.5, "Posição": "A", "Velocidade": 5, "Movimentação": 4},
            {"Nome": "Exemplo Meio", "Nota": 6.0, "Posição": "M", "Velocidade": 3, "Movimentação": 3},
            {"Nome": "Exemplo Zagueiro", "Nota": 7.0, "Posição": "D", "Velocidade": 2, "Movimentação": 2}
        ]
        return pd.DataFrame(dados_exemplo)

    def converter_df_para_excel(self, df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Página12')
        return output.getvalue()

    def carregar_dados_originais(self):
        try:
            df = pd.read_excel(self.url_padrao, sheet_name="Notas pelada")
            return self.limpar_df(df)
        except Exception as e:
            st.error(f"Erro ao conectar com Google Sheets: {e}")
            return self.criar_base_vazia()

    def processar_upload(self, arquivo_upload):
        try:
            df = pd.read_excel(arquivo_upload)
            df = self.limpar_df(df)
            return df
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            return None

    def limpar_df(self, df):
        cols = ["Nome", "Nota", "Posição", "Velocidade", "Movimentação"]
        if df is None or df.empty: return self.criar_base_vazia()
        
        for col in cols:
            if col not in df.columns: df[col] = 0 if col != "Nome" and col != "Posição" else ""

        df = df[cols]
        df = df[df["Posição"].str.upper() != "G"].reset_index(drop=True)
        df = df.dropna(subset=["Nota"])
        df["Nome"] = df["Nome"].astype(str).str.strip().str.title()
        
        duplicados = df[df.duplicated(subset=['Nome'], keep=False)]['Nome'].unique()
        if len(duplicados) > 0:
            st.error(f"⛔ ERRO: Nomes repetidos na base: {', '.join(duplicados)}")
            return self.criar_base_vazia()
        return df

    def processar_lista(self, texto):
        jogadores = []
        texto_lower = texto.lower()
        for kw in ['goleiros', 'lista de espera']:
            if kw in texto_lower: texto = texto[:texto_lower.find(kw)]; break

        linhas = texto.split('\n')
        pattern = r'^\s*\d+[\.\-\)]?\s+(.+)' 
        for linha in linhas:
            match = re.search(pattern, linha)
            if match:
                nome = match.group(1).split('(')[0].strip().title()
                if len(nome) > 1 and nome not in ['.', '-', '...']: jogadores.append(nome)
        
        if len(jogadores) != len(set(jogadores)):
            st.error("⛔ Nomes repetidos na lista colada.")
            st.stop()
        return jogadores

    def calcular_odds(self, times):
        odd = []
        for time in times:
            if not time: odd.append(1.0); continue
            notas = [p[1] for p in time]; vels = [p[3] for p in time]; movs = [p[4] for p in time]
            forca = (np.mean(notas)*1.0) + (np.mean(vels)*0.8) + (np.mean(movs)*0.6)
            odd.append(100 / (forca ** 1.5) if forca > 0 else 0)
        media = sum(odd)/len(odd) if odd else 1
        fator = 3.0/media if media > 0 else 1
        return [o * fator for o in odd]

    def otimizar(self, df, n_times, params):
        dados = []
        for j in df.to_dict('records'):
            dados.append({
                'Nome': j['Nome'],
                'Nota': max(1, min(10, j['Nota'] + random.uniform(-0.7, 0.7))),
                'Posição': j['Posição'],
                'Velocidade': max(1, min(5, j['Velocidade'] + random.uniform(-0.4, 0.4))),
                'Movimentação': max(1, min(5, j['Movimentação'] + random.uniform(-0.4, 0.4)))
            })
        n_jog = len(dados)
        if n_jog < n_times: st.error("Jogadores insuficientes."); st.stop()
        
        ids_j, ids_t = range(n_jog), range(n_times)
        t_vals = {'Nota': sum(d['Nota'] for d in dados), 'Vel': sum(d['Velocidade'] for d in dados), 'Mov': sum(d['Movimentação'] for d in dados)}
        medias = {k: v/n_times for k,v in t_vals.items()}

        prob = pulp.LpProblem("Pelada", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", ((i, j) for i in ids_j for j in ids_t), cat='Binary')

        for i in ids_j: prob += pulp.lpSum(x[i, j] for j in ids_t) == 1
        min_p = n_jog // n_times
        for j in ids_t: 
            prob += pulp.lpSum(x[i, j] for i in ids_j) >= min_p
            prob += pulp.lpSum(x[i, j] for i in ids_j) <= min_p + 1

        if params['pos']:
            for pos in ['D', 'M', 'A']:
                idxs = [i for i, p in enumerate(dados) if p['Posição'] == pos]
                if idxs:
                    mp = len(idxs)//n_times
                    for j in ids_t: prob += pulp.lpSum(x[i, j] for i in idxs) >= mp

        devs = {k: pulp.LpVariable.dicts(f"d_{k}", ids_t, lowBound=0) for k in ['Nota', 'Vel', 'Mov']}
        k_map = {'Nota':'Nota', 'Vel':'Velocidade', 'Mov':'Movimentação'}
        for j in ids_t:
            for k_abv, k_full in k_map.items():
                soma = pulp.lpSum(x[i, j] * dados[i][k_full] for i in ids_j)
                prob += soma - medias[k_abv] <= devs[k_abv][j]
                prob += medias[k_abv] - soma <= devs[k_abv][j]

        obj = pulp.lpSum(0.1 * devs['Nota'][j] for j in ids_t)
        if params['nota']: obj += pulp.lpSum(10 * devs['Nota'][j] for j in ids_t)
        if params['vel']: obj += pulp.lpSum(4 * devs['Vel'][j] for j in ids_t)
        if params['mov']: obj += pulp.lpSum(3 * devs['Mov'][j] for j in ids_t)

        prob += obj
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

        times = [[] for _ in range(n_times)]
        for i in ids_j:
            for j in ids_t:
                if pulp.value(x[i, j]) == 1:
                    times[j].append([dados[i]['Nome'], dados[i]['Nota'], dados[i]['Posição'], dados[i]['Velocidade'], dados[i]['Movimentação']])
                    break
        return times

def botao_copiar_js(texto_para_copiar):
    texto_js = json.dumps(texto_para_copiar)
    html_code = f"""
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <button onclick="copiarTexto()" style="width: 100%; height: 50px; background-color: #25D366; color: white; border: none; border-radius: 8px; font-weight: bold; font-size: 16px; cursor: pointer; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);">📋 COPIAR PARA WHATSAPP</button>
        <script>
            function copiarTexto() {{
                const texto = {texto_js};
                const el = document.createElement('textarea'); el.value = texto; document.body.appendChild(el); el.select(); document.execCommand('copy'); document.body.removeChild(el);
                const btn = document.querySelector('button'); const originalText = btn.innerText; btn.innerText = '✅ COPIADO!'; btn.style.backgroundColor = '#128C7E';
                setTimeout(() => {{ btn.innerText = originalText; btn.style.backgroundColor = '#25D366'; }}, 2000);
            }}
        </script>
    </div>
    """
    components.html(html_code, height=70)

# --- FRONTEND ---
def main():
    logic = PeladaLogic()
    st.title("⚽ Sorteador Pelada PRO")

    if 'df_base' not in st.session_state: st.session_state.df_base = logic.criar_base_vazia()
    if 'novos_jogadores' not in st.session_state: st.session_state.novos_jogadores = []
    if 'is_admin' not in st.session_state: st.session_state.is_admin = False
    if 'aviso_sem_planilha' not in st.session_state: st.session_state.aviso_sem_planilha = False
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🔐 Configuração do Grupo")
        nome_pelada = st.text_input("Nome da Pelada:", placeholder="Ex: Pelada de Domingo")
        
        # VERIFICAÇÃO COM DADOS DO SECRETS
        if nome_pelada.strip().upper() == str(NOME_PELADA_ADM).upper():
            st.success("Grupo identificado!")
            opcao = st.radio("Selecione a ação:", ["Acessar Base Original (Admin)", "Criar Nova Lista (Limpar)"])
            
            if opcao == "Acessar Base Original (Admin)":
                senha = st.text_input("Senha de Acesso:", type="password")
                # VERIFICAÇÃO DA SENHA DO SECRETS
                if senha == str(SENHA_ADM):
                    st.session_state.is_admin = True
                    st.success("🔓 Acesso Liberado")
                else:
                    st.session_state.is_admin = False
                    if senha: st.error("Senha incorreta")
            else:
                st.session_state.is_admin = False
                if st.button("🗑️ Confirmar Limpeza"):
                    st.session_state.df_base = logic.criar_base_vazia()
                    st.session_state.novos_jogadores = []
                    st.rerun()
        else:
            st.session_state.is_admin = False
            if st.button("🗑️ Limpar / Começar do Zero"):
                st.session_state.df_base = logic.criar_base_vazia()
                st.session_state.novos_jogadores = []
                st.rerun()

        st.markdown("---")
        st.subheader("📂 Banco de Dados")
        
        # AÇÕES ADMIN
        if st.session_state.is_admin:
            if st.button("🔄 Carregar Planilha Original"):
                st.session_state.df_base = logic.carregar_dados_originais()
                st.session_state.novos_jogadores = []
                st.success(f"Base carregada: {len(st.session_state.df_base)} jogadores.")
        
        # --- UPLOAD E EXEMPLO ---
        st.write("Substituir por Excel Próprio:")
        
        df_exemplo = logic.criar_exemplo()
        excel_exemplo = logic.converter_df_para_excel(df_exemplo)
        st.download_button(
            label="📥 Baixar Modelo de Planilha",
            data=excel_exemplo,
            file_name="modelo_pelada.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Baixe este arquivo para ver como preencher os dados corretamente."
        )

        uploaded_file = st.file_uploader("", type=["xlsx"], label_visibility="collapsed")
        if uploaded_file:
            if 'ultimo_arquivo' not in st.session_state or st.session_state.ultimo_arquivo != uploaded_file.name:
                df_novo = logic.processar_upload(uploaded_file)
                if df_novo is not None:
                    st.session_state.df_base = df_novo
                    st.session_state.novos_jogadores = []
                    st.session_state.ultimo_arquivo = uploaded_file.name
                    st.success("Arquivo carregado!")

        # --- DOWNLOAD (RESULTADO) ---
        st.markdown("---")
        if not st.session_state.df_base.empty:
            st.write("Salvar dados atuais:")
            if st.session_state.is_admin:
                st.info("🔒 O download da Base Mestra é bloqueado por segurança.")
            else:
                nome_arquivo = nome_pelada.strip()
                if not nome_arquivo: nome_arquivo = "minha_pelada"
                if not nome_arquivo.endswith(".xlsx"): nome_arquivo += ".xlsx"
                excel_data = logic.converter_df_para_excel(st.session_state.df_base)
                st.download_button(label="💾 Baixar Minha Planilha", data=excel_data, file_name=nome_arquivo, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            if not st.session_state.is_admin:
                st.info("Adicione jogadores para baixar a planilha.")

    # --- CADASTRO MANUAL ---
    with st.expander("📝 Adicionar Jogador Manualmente", expanded=False):
        with st.form("form_add_manual"):
            col_a, col_b = st.columns(2)
            nome_m = col_a.text_input("Nome")
            p_m = col_b.selectbox("Posição", ["M", "A", "D"])
            n_m = st.slider("Nota", 1.0, 10.0, 6.0, 0.5)
            v_m = st.slider("Velocidade", 1, 5, 3)
            mv_m = st.slider("Movimentação", 1, 5, 3)
            if st.form_submit_button("Adicionar à Base"):
                if nome_m:
                    novo = {'Nome': nome_m.title(), 'Nota': n_m, 'Posição': p_m, 'Velocidade': v_m, 'Movimentação': mv_m}
                    st.session_state.df_base = pd.concat([st.session_state.df_base, pd.DataFrame([novo])], ignore_index=True)
                    st.success(f"{nome_m} salvo!")
                else: st.error("Digite um nome.")

    # --- INPUT PRINCIPAL ---
    st.markdown(f"**Modo:** {'🔐 ADMIN (Download Bloqueado)' if st.session_state.is_admin else '👤 Público (Base Própria)'}")
    lista_texto = st.text_area("Cole a lista numerada:", height=120, placeholder="1. Jogador A\n2. Jogador B...")
    col1, col2 = st.columns(2)
    n_times = col1.selectbox("Nº Times:", range(2, 11), index=1)
    
    with st.expander("⚙️ Critérios", expanded=False):
        c_pos = st.checkbox("Equilibrar Posição", value=True)
        c_nota = st.checkbox("Equilibrar Nota", value=True)
        c_vel = st.checkbox("Equilibrar Velocidade", value=True)
        c_mov = st.checkbox("Equilibrar Movimentação", value=True)

    if st.button("🎲 SORTEAR TIMES"):
        nomes = logic.processar_lista(lista_texto)
        if not nomes: st.warning("Lista vazia!"); st.stop()

        # --- VERIFICAÇÃO SE EXISTE PLANILHA CARREGADA ---
        if st.session_state.df_base.empty:
            st.session_state.aviso_sem_planilha = True
            st.session_state.nomes_pendentes = nomes
            st.rerun()
        
        # Se tem planilha, segue fluxo normal
        conhecidos = st.session_state.df_base['Nome'].tolist()
        faltantes = [n for n in nomes if n not in conhecidos and n not in [x['Nome'] for x in st.session_state.novos_jogadores]]
        
        if faltantes:
            st.session_state.faltantes_temp = faltantes
            st.rerun()
        else:
            df_final = st.session_state.df_base.copy()
            if st.session_state.novos_jogadores: df_final = pd.concat([df_final, pd.DataFrame(st.session_state.novos_jogadores)], ignore_index=True)
            df_jogar = df_final[df_final['Nome'].isin(nomes)].drop_duplicates(subset=['Nome'], keep='last')
            try:
                with st.spinner('Sorteando...'):
                    times = logic.otimizar(df_jogar, n_times, {'pos': c_pos, 'nota': c_nota, 'vel': c_vel, 'mov': c_mov})
                    st.session_state.resultado = times
            except Exception as e: st.error(f"Erro: {e}")

    # --- BLOCO DE AVISO: SEM PLANILHA ---
    if st.session_state.get('aviso_sem_planilha'):
        st.warning("⚠️ NENHUMA PLANILHA DETECTADA!")
        st.markdown(f"""
        Você não carregou a base Admin e nem fez Upload de uma planilha própria.
        
        Isso significa que você terá que **adicionar notas manualmente para todos os {len(st.session_state.nomes_pendentes)} jogadores** da lista.
        """)
        
        col_conf1, col_conf2 = st.columns(2)
        if col_conf1.button("✅ Sim, quero cadastrar manualmente"):
            # Passa a lista inteira para o sistema de cadastro individual
            st.session_state.faltantes_temp = st.session_state.nomes_pendentes
            st.session_state.aviso_sem_planilha = False
            st.rerun()
        
        if col_conf2.button("❌ Não, vou carregar a planilha"):
            st.session_state.aviso_sem_planilha = False
            st.rerun()

    # --- FALTANTES (CADASTRO INDIVIDUAL) ---
    if 'faltantes_temp' in st.session_state and st.session_state.faltantes_temp:
        nome_atual = st.session_state.faltantes_temp[0]
        # Mostra contador de progresso
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
                st.session_state.df_base = pd.concat([st.session_state.df_base, pd.DataFrame([novo])], ignore_index=True)
                st.session_state.faltantes_temp.pop(0)
                st.rerun()

    # --- RESULTADO ---
    if 'resultado' in st.session_state and not st.session_state.get('aviso_sem_planilha') and not st.session_state.get('faltantes_temp'):
        times = st.session_state.resultado
        odds = logic.calcular_odds(times)
        texto_copiar = ""
        st.markdown("---")
        for i, time in enumerate(times):
            if not time: continue
            ordem = {'G': 0, 'D': 1, 'M': 2, 'A': 3}
            time.sort(key=lambda x: (ordem.get(x[2], 99), x[0]))
            texto_copiar += f"*Time {i+1}:*\n"; 
            for p in time: texto_copiar += f"{p[0]}\n"
            texto_copiar += "\n"
        botao_copiar_js(texto_copiar)

        for i, time in enumerate(times):
            if not time: continue
            ordem = {'G': 0, 'D': 1, 'M': 2, 'A': 3}
            time.sort(key=lambda x: (ordem.get(x[2], 99), x[0]))
            m_nota = np.mean([p[1] for p in time])
            m_vel = np.mean([p[3] for p in time])
            m_mov = np.mean([p[4] for p in time])
            rows = ""
            for p in time: rows += f"<div style='display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #eee;'><div><span style='font-weight:bold; color:black'>{p[0]}</span> <span style='font-size:12px; background:#eee; padding:2px 5px; border-radius:4px; color:#333'>{p[2]}</span></div><div style='font-family:monospace; font-size:14px'><span style='color:#d39e00'>⭐{p[1]:.1f}</span> <span style='color:#0056b3'>⚡{p[3]:.1f}</span> <span style='color:#28a745'>🔄{p[4]:.1f}</span></div></div>"
            st.markdown(f"<div style='background:white; padding:15px; border-radius:10px; margin-bottom:20px; border:1px solid #ddd; box-shadow:0 2px 5px rgba(0,0,0,0.1);'><div style='display:flex; justify-content:space-between; margin-bottom:10px; border-bottom:2px solid #333; padding-bottom:10px;'><h3 style='margin:0; color:black'>TIME {i+1}</h3><span style='background:#ffc107; padding:2px 8px; border-radius:10px; font-weight:bold; color:black'>Odd: {odds[i]:.2f}</span></div><div style='background:#f8f9fa; padding:8px; border-radius:8px; display:flex; justify-content:space-around; color:#333; margin-bottom:10px;'><span>⭐ <b>{m_nota:.1f}</b></span><span>⚡ <b>{m_vel:.1f}</b></span><span>🔄 <b>{m_mov:.1f}</b></span></div>{rows}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
