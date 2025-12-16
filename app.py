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
    initial_sidebar_state="collapsed"
)

# --- CSS (VISUAL MOBILE & DARK MODE FIX) ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        height: 3.5em;
        font-weight: bold;
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    /* Estilo para alertas */
    .stAlert {
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- LÓGICA (BACKEND) ---
class PeladaLogic:
    def __init__(self):
        # NOVA URL ATUALIZADA
        self.url_padrao = "https://docs.google.com/spreadsheets/d/1gCQFG_mYX5DXjh1LRI_UdgrPtkYbkBVLoI3LeOjk5ak/export?format=xlsx"

    def converter_df_para_excel(self, df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Notas pelada')
        return output.getvalue()

    def carregar_dados_iniciais(self):
        try:
            df = pd.read_excel(self.url_padrao, sheet_name="Notas pelada")
            return self.limpar_df(df)
        except Exception as e:
            st.error(f"Erro ao conectar com Google Sheets: {e}")
            return pd.DataFrame(columns=["Nome", "Nota", "Posição", "Velocidade", "Movimentação"])

    def processar_upload(self, arquivo_upload):
        try:
            df = pd.read_excel(arquivo_upload)
            df = self.limpar_df(df)
            return df
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            return None

    def limpar_df(self, df):
        cols_obrigatorias = ["Nome", "Nota", "Posição", "Velocidade", "Movimentação"]
        
        # Validação de colunas
        if not all(col in df.columns for col in cols_obrigatorias):
            st.error(f"❌ O arquivo deve ter as colunas: {', '.join(cols_obrigatorias)}")
            return pd.DataFrame()

        # Filtragem básica
        df = df[df["Posição"].str.upper() != "G"].reset_index(drop=True)
        df = df[cols_obrigatorias].dropna(subset=["Nota"])
        df["Nome"] = df["Nome"].astype(str).str.strip().str.title()
        
        # Trava de Duplicidade
        duplicados = df[df.duplicated(subset=['Nome'], keep=False)]['Nome'].unique()
        if len(duplicados) > 0:
            st.error(f"⛔ ERRO CRÍTICO: Existem nomes repetidos na base de dados!")
            st.write("Corrija os seguintes nomes no arquivo:")
            for d in duplicados:
                st.markdown(f"- 🔴 **{d}**")
            return pd.DataFrame() # Retorna vazio para travar
            
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
            seen = set()
            dupes = [x for x in jogadores if x in seen or seen.add(x)]
            st.error(f"⛔ ERRO: Nomes repetidos na lista colada: **{', '.join(dupes)}**")
            st.stop()
            
        return jogadores

    def calcular_odds(self, times):
        odd = []
        for time in times:
            if not time: 
                odd.append(1.0); continue
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
        <button onclick="copiarTexto()" style="
            width: 100%; height: 50px; background-color: #25D366; color: white; border: none; 
            border-radius: 8px; font-weight: bold; font-size: 16px; cursor: pointer;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);">
            📋 COPIAR PARA WHATSAPP
        </button>
        <script>
            function copiarTexto() {{
                const texto = {texto_js};
                const el = document.createElement('textarea');
                el.value = texto;
                document.body.appendChild(el);
                el.select();
                document.execCommand('copy');
                document.body.removeChild(el);
                const btn = document.querySelector('button');
                const originalText = btn.innerText;
                btn.innerText = '✅ COPIADO!';
                btn.style.backgroundColor = '#128C7E';
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

    # --- INICIALIZAÇÃO DE DADOS ---
    if 'df_base' not in st.session_state:
        # Carrega URL padrão na primeira vez
        st.session_state.df_base = logic.carregar_dados_iniciais()
    
    if 'novos_jogadores' not in st.session_state:
        st.session_state.novos_jogadores = []

    # --- SIDEBAR (CONFIGURAÇÕES E DADOS) ---
    with st.sidebar:
        st.header("📂 Gerenciar Banco de Dados")
        
        # 1. UPLOAD
        uploaded_file = st.file_uploader("Substituir dados (Upload Excel)", type=["xlsx"])
        if uploaded_file is not None:
            # Verifica se já carregamos este arquivo específico para não recarregar no rerun
            if 'ultimo_arquivo' not in st.session_state or st.session_state.ultimo_arquivo != uploaded_file.name:
                df_novo = logic.processar_upload(uploaded_file)
                if df_novo is not None and not df_novo.empty:
                    st.session_state.df_base = df_novo
                    st.session_state.novos_jogadores = [] # Limpa temporários pois assumimos que o excel é a verdade
                    st.session_state.ultimo_arquivo = uploaded_file.name
                    st.success("✅ Arquivo enviado com sucesso!")
        
        st.markdown("---")

        # 2. CADASTRO MANUAL DIRETO
        with st.expander("📝 Adicionar Jogador Manualmente"):
            with st.form("form_add_manual"):
                nome_m = st.text_input("Nome")
                n_m = st.slider("Nota", 1.0, 10.0, 6.0, 0.5)
                p_m = st.selectbox("Posição", ["M", "A", "D"])
                v_m = st.slider("Velocidade", 1, 5, 3)
                mv_m = st.slider("Movimentação", 1, 5, 3)
                
                if st.form_submit_button("Adicionar à Base"):
                    if nome_m:
                        novo_jogador = {'Nome': nome_m.title(), 'Nota': n_m, 'Posição': p_m, 'Velocidade': v_m, 'Movimentação': mv_m}
                        # Adiciona ao dataframe principal na sessão
                        st.session_state.df_base = pd.concat([st.session_state.df_base, pd.DataFrame([novo_jogador])], ignore_index=True)
                        st.success(f"✅ {nome_m} adicionado!")
                    else:
                        st.error("Nome é obrigatório.")

        st.markdown("---")

        # 3. DOWNLOAD DA BASE ATUALIZADA
        st.write("Salvar dados atuais (inclui manuais):")
        excel_data = logic.converter_df_para_excel(st.session_state.df_base)
        st.download_button(
            label="💾 Baixar Planilha Atualizada",
            data=excel_data,
            file_name="pelada_atualizada.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # --- INPUT PRINCIPAL ---
    lista_texto = st.text_area("Cole a lista numerada:", height=120, placeholder="1. Jogador A\n2. Jogador B...")
    
    col1, col2 = st.columns(2)
    n_times = col1.selectbox("Nº Times:", range(2, 11), index=1)
    
    with st.expander("⚙️ Ajustar Critérios", expanded=False):
        c_pos = st.checkbox("Equilibrar Posição", value=True)
        c_nota = st.checkbox("Equilibrar Nota", value=True)
        c_vel = st.checkbox("Equilibrar Velocidade", value=True)
        c_mov = st.checkbox("Equilibrar Movimentação", value=True)

    if st.button("🎲 SORTEAR TIMES"):
        nomes = logic.processar_lista(lista_texto)
        if not nomes:
            st.warning("Lista vazia!")
            return

        # Verifica quem está na base atual (que já inclui manuais e uploads)
        conhecidos = st.session_state.df_base['Nome'].tolist()
        
        # Faltantes ainda podem ocorrer se alguém colar um nome na lista que não cadastrou nem na base nem no upload
        faltantes = [n for n in nomes if n not in conhecidos and n not in [x['Nome'] for x in st.session_state.novos_jogadores]]
        
        if faltantes:
            st.session_state.faltantes_temp = faltantes
            st.rerun()
        else:
            # Junta base principal + temporários da rodada (se houver)
            df_final = st.session_state.df_base.copy()
            if st.session_state.novos_jogadores:
                df_final = pd.concat([df_final, pd.DataFrame(st.session_state.novos_jogadores)], ignore_index=True)
            
            df_jogar = df_final[df_final['Nome'].isin(nomes)]
            
            # Remove duplicatas caso existam (prioriza o último cadastro)
            df_jogar = df_jogar.drop_duplicates(subset=['Nome'], keep='last')

            params = {'pos': c_pos, 'nota': c_nota, 'vel': c_vel, 'mov': c_mov}
            try:
                with st.spinner('Calculando a melhor combinação...'):
                    times = logic.otimizar(df_jogar, n_times, params)
                    st.session_state.resultado = times
            except Exception as e:
                st.error(f"Erro na otimização: {e}")

    # --- CADASTRO DE FALTANTES (DA LISTA COLADA) ---
    if 'faltantes_temp' in st.session_state and st.session_state.faltantes_temp:
        nome_atual = st.session_state.faltantes_temp[0]
        st.warning(f"⚠️ Jogador na lista mas sem nota: **{nome_atual}**")
        
        with st.form("form_cadastro_faltante"):
            n_val = st.slider("Nota (⭐)", 1.0, 10.0, 6.0, 0.5)
            p_val = st.selectbox("Posição", ["M", "A", "D"])
            v_val = st.select_slider("Velocidade (⚡)", options=[1, 2, 3, 4, 5], value=3)
            m_val = st.select_slider("Movimentação (🔄)", options=[1, 2, 3, 4, 5], value=3)
            
            if st.form_submit_button("Salvar e Continuar"):
                novo = {'Nome': nome_atual, 'Nota': n_val, 'Posição': p_val, 'Velocidade': v_val, 'Movimentação': m_val}
                # Adiciona tanto à lista temporária quanto à base permanente para poder baixar depois
                st.session_state.novos_jogadores.append(novo)
                st.session_state.df_base = pd.concat([st.session_state.df_base, pd.DataFrame([novo])], ignore_index=True)
                
                st.session_state.faltantes_temp.pop(0)
                st.rerun()

    # --- EXIBIÇÃO RESULTADO ---
    if 'resultado' in st.session_state:
        times = st.session_state.resultado
        odds = logic.calcular_odds(times)
        texto_copiar = ""
        st.markdown("---")
        
        for i, time in enumerate(times):
            if not time: continue
            ordem = {'G': 0, 'D': 1, 'M': 2, 'A': 3}
            time.sort(key=lambda x: (ordem.get(x[2], 99), x[0]))
            texto_copiar += f"*Time {i+1}:*\n"
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

            rows_html = ""
            for p in time:
                rows_html += f"""<div style='display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #eee; padding:8px 0;'>
<div style="display:flex; align-items:center; gap:8px;"><span style="font-weight:700; font-size:16px; color:#222;">{p[0]}</span><span style="background:#eee; color:#444; font-size:12px; padding:2px 6px; border-radius:4px; font-weight:bold;">{p[2]}</span></div>
<div style="font-family:monospace; font-size:14px; display:flex; gap:10px;"><span style="color:#d39e00;">⭐{p[1]:.1f}</span><span style="color:#0056b3;">⚡{p[3]:.1f}</span><span style="color:#28a745;">🔄{p[4]:.1f}</span></div>
</div>"""

            card = f"""
<div style="background-color:white; padding:15px; border-radius:12px; margin-bottom:20px; border:1px solid #ddd; box-shadow:0 2px 5px rgba(0,0,0,0.1);">
<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid #333; padding-bottom:10px; margin-bottom:10px;">
<h3 style="margin:0; color:#000; font-weight:800;">TIME {i+1}</h3>
<span style="background:#ffc107; padding:4px 10px; border-radius:15px; font-weight:bold; color:#000; font-size:14px;">Odd: {odds[i]:.2f}</span>
</div>
<div style="background:#f8f9fa; padding:8px; border-radius:8px; display:flex; justify-content:space-around; margin-bottom:10px; color:#333; font-size:14px;">
<span>⭐ <b>{m_nota:.1f}</b></span><span>⚡ <b>{m_vel:.1f}</b></span><span>🔄 <b>{m_mov:.1f}</b></span>
</div>
<div>{rows_html}</div>
</div>
"""
            st.markdown(card, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
