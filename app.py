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
    </style>
""", unsafe_allow_html=True)

# --- LÓGICA (BACKEND) ---
class PeladaLogic:
    def __init__(self):
        self.url_padrao = "https://docs.google.com/spreadsheets/d/1gCQFG_mYX5DXjh1LRI_UdgrPtkYbkBVLoI3LeOjk5ak/export?format=xlsx"

    def criar_base_vazia(self):
        return pd.DataFrame(columns=["Nome", "Nota", "Posição", "Velocidade", "Movimentação"])

    def criar_exemplo(self):
        dados_exemplo = [
            {"Nome": "Exemplo Atacante", "Nota": 8.5, "Posição": "A", "Velocidade": 5, "Movimentação": 4},
            {"Nome": "Exemplo Meio", "Nota": 6.0, "Posição": "M", "Velocidade": 3, "Movimentação": 3},
            {"Nome": "Exemplo Zagueiro", "Nota": 7.0, "Posição": "D", "Velocidade": 2, "Movimentação": 2},
            {"Nome": "Exemplo Goleiro", "Nota": 8.0, "Posição": "G", "Velocidade": 2, "Movimentação": 2}
        ]
        return pd.DataFrame(dados_exemplo)

    def converter_df_para_excel(self, df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Notas pelada')
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
        # REMOVIDO FILTRO DE GOLEIROS AQUI PARA PERMITIR QUE ELES ENTREM NA BASE
        df = df.dropna(subset=["Nota"])
        df["Nome"] = df["Nome"].astype(str).str.strip().str.title()
        
        duplicados = df[df.duplicated(subset=['Nome'], keep=False)]['Nome'].unique()
        if len(duplicados) > 0:
            st.error(f"⛔ ERRO: Nomes repetidos na base: {', '.join(duplicados)}")
            return self.criar_base_vazia()
        return df

    def processar_lista(self, texto):
        jogadores_linha = []
        goleiros = []
        
        # Separação rudimentar de blocos (Linha vs Goleiros)
        texto_lower = texto.lower()
        if 'goleiro' in texto_lower:
            partes = re.split(r'goleiros?:?', texto, flags=re.IGNORECASE)
            texto_linha = partes[0]
            texto_goleiros = partes[1] if len(partes) > 1 else ""
        else:
            texto_linha = texto
            texto_goleiros = ""

        def extrair_nomes(txt):
            lista_nomes = []
            linhas = txt.split('\n')
            # Regex 1: Tenta pegar lista numerada (1. Nome)
            pattern_num = r'^\s*\d+[\.\-\)]?\s+(.+)'
            for linha in linhas:
                match = re.search(pattern_num, linha)
                if match:
                    nome = match.group(1).split('(')[0].strip().title()
                    if len(nome) > 1: lista_nomes.append(nome)
            
            # Fallback: Se não achou nada numerado, pega a linha inteira (lista sem numero)
            if not lista_nomes:
                for linha in linhas:
                    nome = linha.split('(')[0].strip().title()
                    if len(nome) > 1 and nome.lower() not in ['goleiros', 'lista de espera', '...']:
                         lista_nomes.append(nome)
            return lista_nomes

        jogadores_linha = extrair_nomes(texto_linha)
        goleiros = extrair_nomes(texto_goleiros)
        
        todos = jogadores_linha + goleiros
        if len(todos) != len(set(todos)):
            st.warning("⚠️ Atenção: Há nomes duplicados na lista.")
        
        return jogadores_linha, goleiros

    def calcular_odds(self, times):
        odd = []
        for time in times:
            if not time: odd.append(1.0); continue
            # Filtra apenas jogadores de linha para calcular força, ou inclui goleiro com peso menor
            linha = [p for p in time if p[2] != 'G']
            if not linha: linha = time # Se só tiver goleiro
            
            notas = [p[1] for p in linha]; vels = [p[3] for p in linha]; movs = [p[4] for p in linha]
            
            if not notas: forca = 0
            else: forca = (np.mean(notas)*1.0) + (np.mean(vels)*0.8) + (np.mean(movs)*0.6)
            
            odd.append(100 / (forca ** 1.5) if forca > 0 else 0)
        media = sum(odd)/len(odd) if odd else 1
        fator = 3.0/media if media > 0 else 1
        return [o * fator for o in odd]

    def otimizar(self, df, n_times, params, goleiros_fixos=[]):
        # Separa Goleiros da Otimização Linear se já foram identificados
        jogadores_df = df[~df['Nome'].isin(goleiros_fixos)].copy()
        
        dados = []
        for j in jogadores_df.to_dict('records'):
            dados.append({
                'Nome': j['Nome'],
                'Nota': max(1, min(10, j['Nota'] + random.uniform(-0.5, 0.5))), # Randomização menor
                'Posição': j['Posição'],
                'Velocidade': j['Velocidade'],
                'Movimentação': j['Movimentação']
            })
        
        n_jog = len(dados)
        if n_jog < n_times and not goleiros_fixos: 
            st.error("Jogadores insuficientes para o número de times."); return []

        # TENTATIVA 1: OTIMIZAÇÃO COMPLETA
        try:
            times_idx = self._resolver_pulp(dados, n_times, params, n_jog)
        except Exception:
            # TENTATIVA 2: RELAXAR POSIÇÕES (FALLBACK)
            st.warning("⚠️ Não foi possível equilibrar perfeitamente as posições. Tentando focar apenas nas Notas...")
            params['pos'] = False
            try:
                times_idx = self._resolver_pulp(dados, n_times, params, n_jog)
            except Exception as e:
                st.error(f"Não foi possível sortear. Tente reduzir o número de times. Erro: {e}")
                return []

        # Montar estrutura final
        times = [[] for _ in range(n_times)]
        
        # 1. Distribui Goleiros (Round Robin simples ou Aleatório)
        if goleiros_fixos:
            random.shuffle(goleiros_fixos)
            for i, g_nome in enumerate(goleiros_fixos):
                # Busca dados do goleiro na base original
                g_dados = df[df['Nome'] == g_nome].iloc[0]
                # Se faltar goleiro para um time, paciência (ou rodízio). Se sobrar, vai pro banco.
                if i < n_times:
                    times[i].append([g_dados['Nome'], g_dados['Nota'], 'G', g_dados['Velocidade'], g_dados['Movimentação']])

        # 2. Distribui Jogadores de Linha baseada na Otimização
        for t_idx, membros in times_idx.items():
            for m in membros:
                times[t_idx].append([m['Nome'], m['Nota'], m['Posição'], m['Velocidade'], m['Movimentação']])
                
        return times

    def _resolver_pulp(self, dados, n_times, params, n_jog):
        ids_j, ids_t = range(n_jog), range(n_times)
        prob = pulp.LpProblem("Pelada", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", ((i, j) for i in ids_j for j in ids_t), cat='Binary')

        # Restrição: Todo jogador em 1 time
        for i in ids_j: prob += pulp.lpSum(x[i, j] for j in ids_t) == 1
        
        # Restrição: Tamanho dos times (equilibrado)
        min_p = n_jog // n_times
        for j in ids_t: 
            prob += pulp.lpSum(x[i, j] for i in ids_j) >= min_p
            prob += pulp.lpSum(x[i, j] for i in ids_j) <= min_p + 1

        # Restrição: Posições
        if params['pos']:
            for pos in ['D', 'M', 'A']:
                idxs = [i for i, p in enumerate(dados) if p['Posição'] == pos]
                if idxs:
                    mp = len(idxs) // n_times
                    # Relaxamento: permite variação de +1 para evitar infeasibility em números quebrados
                    for j in ids_t: 
                        prob += pulp.lpSum(x[i, j] for i in idxs) >= mp
                        if mp > 0: prob += pulp.lpSum(x[i, j] for i in idxs) <= mp + 2 

        # Variáveis de Desvio (Médias)
        t_vals = {'Nota': sum(d['Nota'] for d in dados), 'Vel': sum(d['Velocidade'] for d in dados)}
        medias = {k: v/n_times for k,v in t_vals.items()}
        devs = {k: pulp.LpVariable.dicts(f"d_{k}", ids_t, lowBound=0) for k in ['Nota', 'Vel']}

        for j in ids_t:
            # Desvio Nota
            soma_n = pulp.lpSum(x[i, j] * dados[i]['Nota'] for i in ids_j)
            prob += soma_n - medias['Nota'] <= devs['Nota'][j]
            prob += medias['Nota'] - soma_n <= devs['Nota'][j]
            
            # Desvio Velocidade (se ativado)
            if params['vel']:
                soma_v = pulp.lpSum(x[i, j] * dados[i]['Velocidade'] for i in ids_j)
                prob += soma_v - medias['Vel'] <= devs['Vel'][j]
                prob += medias['Vel'] - soma_v <= devs['Vel'][j]

        # Função Objetivo
        obj = pulp.lpSum(devs['Nota'][j] for j in ids_t) * 10
        if params['vel']: obj += pulp.lpSum(devs['Vel'][j] for j in ids_t) * 5
        
        prob += obj
        status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))
        
        if status != pulp.LpStatusOptimal:
            raise Exception("Solução inviável matematicamente.")

        times_res = {j: [] for j in ids_t}
        for i in ids_j:
            for j in ids_t:
                if pulp.value(x[i, j]) == 1:
                    times_res[j].append(dados[i])
                    break
        return times_res

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
        st.header("🔐 Configuração")
        nome_pelada = st.text_input("Nome da Pelada:")
        
        if nome_pelada.strip().upper() == str(NOME_PELADA_ADM).upper():
            st.success("Grupo identificado!")
            opcao = st.radio("Ação:", ["Admin (Base Mestra)", "Limpar"])
            if opcao == "Admin (Base Mestra)":
                if st.text_input("Senha:", type="password") == str(SENHA_ADM):
                    st.session_state.is_admin = True
                    st.success("🔓 Liberado")
            else:
                if st.button("🗑️ Limpar Tudo"):
                    st.session_state.clear()
                    st.rerun()
        
        st.markdown("---")
        # ADMIN: CARREGAR BASE
        if st.session_state.is_admin:
            if st.button("🔄 Carregar Google Sheets"):
                st.session_state.df_base = logic.carregar_dados_originais()
                st.session_state.novos_jogadores = []
                st.success(f"Carregado: {len(st.session_state.df_base)} nomes.")
        
        # USER: UPLOAD & MODELO
        st.write("📂 Arquivo Próprio")
        df_ex = logic.criar_exemplo()
        st.download_button("📥 Baixar Modelo", logic.converter_df_para_excel(df_ex), "modelo.xlsx")
        
        up = st.file_uploader("", type=["xlsx"], label_visibility="collapsed")
        if up:
            if 'last_up' not in st.session_state or st.session_state.last_up != up.name:
                df = logic.processar_upload(up)
                if df is not None:
                    st.session_state.df_base = df
                    st.session_state.last_up = up.name
                    st.success("Carregado!")

        # DOWNLOAD
        st.markdown("---")
        if not st.session_state.df_base.empty and not st.session_state.is_admin:
            n_arq = nome_pelada.strip() or "minha_pelada"
            st.download_button("💾 Salvar Planilha Atual", logic.converter_df_para_excel(st.session_state.df_base), f"{n_arq}.xlsx")

    # --- CADASTRO MANUAL ---
    with st.expander("📝 Cadastro Manual (Rápido)", expanded=False):
        with st.form("add_manual"):
            c1, c2 = st.columns(2)
            nm = c1.text_input("Nome")
            ps = c2.selectbox("Posição", ["M", "A", "D", "G"])
            nt = st.slider("Nota", 1.0, 10.0, 6.0)
            if st.form_submit_button("Adicionar"):
                if nm:
                    novo = {'Nome': nm.title(), 'Nota': nt, 'Posição': ps, 'Velocidade': 3, 'Movimentação': 3}
                    st.session_state.df_base = pd.concat([st.session_state.df_base, pd.DataFrame([novo])], ignore_index=True)
                    st.success(f"{nm} adicionado!")

    # --- INPUT LISTA ---
    st.info("Cole a lista abaixo. O sistema tenta separar 'Goleiros' se você escrever a palavra 'Goleiros' antes dos nomes deles.")
    lista_texto = st.text_area("Lista de Presença:", height=150, placeholder="1. Jogador A\n2. Jogador B\n\nGoleiros:\n1. Muralha")
    
    c1, c2 = st.columns(2)
    n_times = c1.selectbox("Nº Times:", range(2, 9), index=1)
    
    with st.expander("⚙️ Opções Avançadas"):
        c_pos = st.checkbox("Equilibrar Posição", True)
        c_vel = st.checkbox("Equilibrar Velocidade", True)

    if st.button("🎲 SORTEAR AGORA", type="primary"):
        # 1. PARSING
        nomes_linha, nomes_goleiro = logic.processar_lista(lista_texto)
        todos_nomes = nomes_linha + nomes_goleiro
        
        if not todos_nomes: st.warning("Lista vazia!"); st.stop()

        # 2. CHECK DE BASE
        if st.session_state.df_base.empty:
            st.session_state.aviso_sem_planilha = True
            st.session_state.nomes_pendentes = todos_nomes
            st.rerun()

        # 3. CHECK DE FALTANTES
        conhecidos = st.session_state.df_base['Nome'].tolist()
        faltantes = [n for n in todos_nomes if n not in conhecidos]
        
        if faltantes:
            st.session_state.faltantes_temp = faltantes
            st.rerun()

        # 4. OTIMIZAÇÃO
        df_full = st.session_state.df_base.drop_duplicates(subset=['Nome'], keep='last')
        df_jogar = df_full[df_full['Nome'].isin(nomes_linha)]
        
        # Garante que goleiros na lista entrem como "goleiros fixos"
        # Se o goleiro não estiver na base, ele caiu no passo 3 (faltantes). 
        # Aqui ele já está na base, mas precisamos passar a lista de nomes explicitamente.
        
        try:
            with st.spinner('Calculando melhores times...'):
                times = logic.otimizar(df_jogar, n_times, {'pos': c_pos, 'nota': True, 'vel': c_vel, 'mov': True}, goleiros_fixos=nomes_goleiro)
                st.session_state.resultado = times
        except Exception as e:
            st.error(f"Erro fatal: {e}")

    # --- POPUP: SEM PLANILHA ---
    if st.session_state.get('aviso_sem_planilha'):
        st.warning("⚠️ MODO MANUAL ATIVADO")
        st.write(f"Você vai cadastrar **{len(st.session_state.nomes_pendentes)}** jogadores agora.")
        c1, c2 = st.columns(2)
        if c1.button("✅ Começar"):
            st.session_state.faltantes_temp = st.session_state.nomes_pendentes
            st.session_state.aviso_sem_planilha = False
            st.rerun()
        if c2.button("❌ Cancelar"):
            st.session_state.aviso_sem_planilha = False
            st.rerun()

    # --- POPUP: FALTANTES ---
    if 'faltantes_temp' in st.session_state and st.session_state.faltantes_temp:
        atual = st.session_state.faltantes_temp[0]
        # Tenta adivinhar se é goleiro pelo nome na lista original ou se foi parseado como goleiro
        eh_goleiro_chute = "Goleiro" in lista_texto and atual in lista_texto.split("Goleiro")[1]
        idx_p = 3 if eh_goleiro_chute else 0 # Default para G se parecer goleiro

        st.markdown(f"### 🆕 Cadastrar: {atual}")
        with st.form("cad_faltante"):
            n = st.slider("Nota", 1.0, 10.0, 6.0)
            p = st.selectbox("Posição", ["M", "A", "D", "G"], index=idx_p)
            v = st.slider("Velocidade", 1, 5, 3)
            if st.form_submit_button("Salvar"):
                novo = {'Nome': atual, 'Nota': n, 'Posição': p, 'Velocidade': v, 'Movimentação': 3}
                st.session_state.df_base = pd.concat([st.session_state.df_base, pd.DataFrame([novo])], ignore_index=True)
                st.session_state.faltantes_temp.pop(0)
                st.rerun()

    # --- RESULTADO ---
    if 'resultado' in st.session_state and not st.session_state.get('faltantes_temp'):
        times = st.session_state.resultado
        if not times: st.error("Erro na geração dos times."); st.stop()
        
        odds = logic.calcular_odds(times)
        
        # TEXTO WHATSAPP OTIMIZADO
        txt_zap = f"⚽ *TIMES SORTEADOS - {nome_pelada or 'PELADA'}* ⚽\n\n"
        for i, t in enumerate(times):
            t.sort(key=lambda x: (0 if x[2]=='G' else 1, x[2], x[0])) # Goleiro primeiro
            txt_zap += f"*TIME {i+1}* (Força: {odds[i]:.0f}%)\n"
            for p in t: txt_zap += f"{'🧤' if p[2]=='G' else ''} {p[0]}\n"
            txt_zap += "\n"
        
        botao_copiar_js(txt_zap)
        
        # EXIBIÇÃO VISUAL
        cols = st.columns(n_times)
        for i, col in enumerate(cols):
            if i < len(times):
                t = times[i]
                t.sort(key=lambda x: (0 if x[2]=='G' else 1, x[2], x[0]))
                media = np.mean([x[1] for x in t])
                with col:
                    st.markdown(f"""
                    <div style="background:#f0f2f6; padding:10px; border-radius:8px; border-top: 5px solid #ff4b4b">
                        <h4 style="text-align:center; color:#31333F">Time {i+1}</h4>
                        <p style="text-align:center; font-size:12px; color:#666">Média: {media:.1f}</p>
                        <hr style="margin:5px 0">
                        {''.join([f"<div style='font-size:14px; display:flex; justify-content:space-between'><span>{'🧤' if p[2]=='G' else '🏃'} <b>{p[0]}</b></span> <span style='color:#555'>{p[2]}</span></div>" for p in t])}
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
