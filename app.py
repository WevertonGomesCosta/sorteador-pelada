import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
import numpy as np
import random
import pulp
import json

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Sorteador Pelada",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Estilo CSS Global
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        height: 3em;
        font-weight: bold;
        background-color: #ff4b4b;
        color: white;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    iframe { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- LÓGICA (BACKEND) ---
class PeladaLogic:
    def __init__(self):
        # URL da planilha
        self.url = "https://docs.google.com/spreadsheets/d/1Dy5Zu8DsM4H-6eHSu_1RAfEB3UoOAEl8GhIVoFgk76A/export?format=xlsx"

    @st.cache_data(ttl=600)
    def carregar_dados(_self):
        try:
            df = pd.read_excel(_self.url, sheet_name="Notas pelada")
            cols = ["Nome", "Nota", "Posição", "Velocidade", "Movimentação"]
            df = df[df["Posição"].str.upper() != "G"].reset_index(drop=True)
            df = df[cols].dropna(subset=["Nota"])
            df["Nome"] = df["Nome"].astype(str).str.strip().str.title()
            return df
        except Exception as e:
            st.error(f"Erro ao carregar Excel: {e}")
            return pd.DataFrame()

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
            border-radius: 8px; font-weight: bold; font-size: 18px; cursor: pointer;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1); transition: background-color 0.2s;">
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
    if 'df_base' not in st.session_state:
        st.session_state.df_base = logic.carregar_dados()
    if 'novos_jogadores' not in st.session_state:
        st.session_state.novos_jogadores = []

    st.title("⚽ Sorteador Mobile")

    lista_texto = st.text_area("Cole a lista numerada:", height=150, placeholder="1. Jogador A\n2. Jogador B...")
    
    col1, col2 = st.columns(2)
    n_times = col1.selectbox("Nº Times:", range(2, 11), index=1)
    
    st.markdown("**Critérios:**")
    c_pos = st.checkbox("Equilibrar Posição", value=True)
    c_nota = st.checkbox("Equilibrar Nota", value=True)
    c_vel = st.checkbox("Equilibrar Velocidade", value=True)
    c_mov = st.checkbox("Equilibrar Movimentação", value=True)

    if st.button("🎲 SORTEAR TIMES"):
        nomes = logic.processar_lista(lista_texto)
        if not nomes:
            st.warning("Lista vazia!")
            return

        conhecidos = st.session_state.df_base['Nome'].tolist()
        faltantes = [n for n in nomes if n not in conhecidos and n not in [x['Nome'] for x in st.session_state.novos_jogadores]]
        
        if faltantes:
            st.session_state.faltantes_temp = faltantes
            st.rerun()
        else:
            df_final = st.session_state.df_base.copy()
            if st.session_state.novos_jogadores:
                df_final = pd.concat([df_final, pd.DataFrame(st.session_state.novos_jogadores)], ignore_index=True)
            
            df_jogar = df_final[df_final['Nome'].isin(nomes)]
            
            params = {'pos': c_pos, 'nota': c_nota, 'vel': c_vel, 'mov': c_mov}
            try:
                with st.spinner('Calculando...'):
                    times = logic.otimizar(df_jogar, n_times, params)
                    st.session_state.resultado = times
            except Exception as e:
                st.error(f"Erro: {e}")

    # CADASTRO DE FALTANTES
    if 'faltantes_temp' in st.session_state and st.session_state.faltantes_temp:
        nome_atual = st.session_state.faltantes_temp[0]
        st.info(f"🆕 Cadastrando: **{nome_atual}**")
        with st.form("form_cadastro"):
            n_val = st.slider("Nota", 1.0, 10.0, 6.0, 0.5)
            p_val = st.selectbox("Posição", ["M", "A", "D"])
            v_val = st.select_slider("Velocidade", options=[1, 2, 3, 4, 5], value=3)
            m_val = st.select_slider("Movimentação", options=[1, 2, 3, 4, 5], value=3)
            if st.form_submit_button("Salvar Jogador"):
                novo = {'Nome': nome_atual, 'Nota': n_val, 'Posição': p_val, 'Velocidade': v_val, 'Movimentação': m_val}
                st.session_state.novos_jogadores.append(novo)
                st.session_state.faltantes_temp.pop(0)
                st.rerun()

    # EXIBIÇÃO RESULTADO
    if 'resultado' in st.session_state:
        times = st.session_state.resultado
        odds = logic.calcular_odds(times)
        texto_copiar = ""
        st.markdown("---")
        
        # Gera texto para cópia
        for i, time in enumerate(times):
            if not time: continue
            ordem = {'G': 0, 'D': 1, 'M': 2, 'A': 3}
            time.sort(key=lambda x: (ordem.get(x[2], 99), x[0]))
            texto_copiar += f"*Time {i+1}:*\n"
            for p in time: texto_copiar += f"{p[0]}\n"
            texto_copiar += "\n"
            
        botao_copiar_js(texto_copiar)

        # Loop de Exibição dos Cards
        for i, time in enumerate(times):
            if not time: continue
            ordem = {'G': 0, 'D': 1, 'M': 2, 'A': 3}
            time.sort(key=lambda x: (ordem.get(x[2], 99), x[0]))
            
            m_nota = np.mean([p[1] for p in time])
            m_vel = np.mean([p[3] for p in time])
            m_mov = np.mean([p[4] for p in time])

            with st.container():
                # CSS Inline forçado para garantir fundo branco e texto preto
                card_html = f"""
                <div style="background-color:#ffffff; padding:15px; border-radius:12px; margin-bottom:20px; border:1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color:#000000; font-family: sans-serif;">
                    
                    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid #333; padding-bottom:10px; margin-bottom:10px;">
                        <h3 style="margin:0; color:#000000; font-weight:800;">TIME {i+1}</h3>
                        <span style="background:#ffc107; padding:4px 10px; border-radius:20px; font-weight:bold; font-size:0.9em; color:#000;">Odd: {odds[i]:.2f}</span>
                    </div>
                    
                    <div style="font-size:0.9em; color:#333; display:flex; justify-content:space-around; margin-bottom:15px; background-color:#f8f9fa; padding:8px; border-radius:8px;">
                        <span title="Nota">⭐ <b>{m_nota:.1f}</b></span> 
                        <span title="Velocidade">⚡ <b>{m_vel:.1f}</b></span> 
                        <span title="Movimentação">🔄 <b>{m_mov:.1f}</b></span>
                    </div>

                    <div style="display:flex; flex-direction:column; gap:8px;">
                """
                
                for p in time:
                    # p = [Nome, Nota, Pos, Vel, Mov]
                    card_html += f"""
                    <div style='display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #eee; padding-bottom:5px;'>
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span style="font-weight:600; font-size:1em; color:#000;">{p[0]}</span>
                            <span style="background:#e0e0e0; color:#333; font-size:0.7em; padding:2px 6px; border-radius:4px; font-weight:bold;">{p[2]}</span>
                        </div>
                        <div style="font-size:0.85em; font-family:monospace; color:#555; display:flex; gap:8px;">
                            <span style="color:#d39e00;" title="Nota">⭐{p[1]:.1f}</span>
                            <span style="color:#0056b3;" title="Velocidade">⚡{p[3]:.1f}</span>
                            <span style="color:#28a745;" title="Movimentação">🔄{p[4]:.1f}</span>
                        </div>
                    </div>
                    """
                
                card_html += "</div></div>"
                st.markdown(card_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
