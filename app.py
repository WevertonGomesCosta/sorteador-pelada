
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import unicodedata
import hashlib
import json

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
    }

    .section-subtitle {
        margin-top: -0.10rem;
        margin-bottom: 0.85rem;
        font-size: 0.93rem;
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

    #install-app-container a,
    #install-app-container button,
    #install-app-container [role="button"],
    #install-app-container .stButton > button {
        background: rgba(15, 23, 42, 0.28) !important;
        color: #dbe7ef !important;
        border: 1px solid rgba(45, 212, 191, 0.55) !important;
        box-shadow: none !important;
        opacity: 0.94 !important;
    }

    #install-app-container a:hover,
    #install-app-container button:hover,
    #install-app-container [role="button"]:hover,
    #install-app-container .stButton > button:hover {
        background: rgba(15, 23, 42, 0.42) !important;
        color: #f8fafc !important;
        border-color: rgba(45, 212, 191, 0.75) !important;
        box-shadow: none !important;
        transform: none !important;
    }


    html[data-theme="light"] #install-app-container a,
    html[data-theme="light"] #install-app-container button,
    html[data-theme="light"] #install-app-container [role="button"],
    html[data-theme="light"] #install-app-container .stButton > button,
    html:not([data-theme="dark"]) #install-app-container a,
    html:not([data-theme="dark"]) #install-app-container button,
    html:not([data-theme="dark"]) #install-app-container [role="button"],
    html:not([data-theme="dark"]) #install-app-container .stButton > button {
        background: rgba(226, 232, 240, 0.98) !important;
        color: #0f172a !important;
        border: 1px solid rgba(100, 116, 139, 0.55) !important;
        opacity: 1 !important;
    }

    html[data-theme="light"] #install-app-container a:hover,
    html[data-theme="light"] #install-app-container button:hover,
    html[data-theme="light"] #install-app-container [role="button"]:hover,
    html[data-theme="light"] #install-app-container .stButton > button:hover,
    html:not([data-theme="dark"]) #install-app-container a:hover,
    html:not([data-theme="dark"]) #install-app-container button:hover,
    html:not([data-theme="dark"]) #install-app-container [role="button"]:hover,
    html:not([data-theme="dark"]) #install-app-container .stButton > button:hover {
        background: rgba(203, 213, 225, 0.98) !important;
        color: #0f172a !important;
        border-color: rgba(71, 85, 105, 0.7) !important;
    }

    html[data-theme="light"] #install-app-container a *,
    html[data-theme="light"] #install-app-container button *,
    html[data-theme="light"] #install-app-container [role="button"] *,
    html[data-theme="light"] #install-app-container .stButton > button *,
    html:not([data-theme="dark"]) #install-app-container a *,
    html:not([data-theme="dark"]) #install-app-container button *,
    html:not([data-theme="dark"]) #install-app-container [role="button"] *,
    html:not([data-theme="dark"]) #install-app-container .stButton > button * {
        color: #0f172a !important;
        fill: #0f172a !important;
        stroke: #0f172a !important;
    }

    @media (max-width: 900px) {
        .summary-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    :root {
        --section-title-color: #0f172a;
        --section-subtitle-color: #475569;

        --action-primary-bg: #14B8A6;
        --action-primary-bg-hover: #0F9F94;
        --action-primary-border: #2DD4BF;
        --action-primary-text: #F8FAFC;

        --action-secondary-bg: transparent;
        --action-secondary-bg-hover: rgba(20, 184, 166, 0.08);
        --action-secondary-border: #334155;
        --action-secondary-border-hover: #2DD4BF;
        --action-secondary-text: #E5E7EB;

        --action-danger-bg: #EF4444;
        --action-danger-bg-hover: #DC2626;
        --action-danger-border: #F87171;
        --action-danger-text: #FFFFFF;

        --action-disabled-bg: #111827;
        --action-disabled-border: #374151;
        --action-disabled-text: #6B7280;

        --action-radius: 14px;
        --action-height: 3.15rem;
        --action-font-weight: 700;
    }

    html[data-theme="dark"],
    body[data-theme="dark"],
    [data-theme="dark"] {
        --section-title-color: #f8fafc;
        --section-subtitle-color: #cbd5e1;
    }

    [class*="st-key-action-primary-"] div.stButton > button,
    [class*="st-key-action-primary-"] div[data-testid="stFormSubmitButton"] > button {
        background: var(--action-primary-bg) !important;
        color: var(--action-primary-text) !important;
        border: 1px solid var(--action-primary-border) !important;
        border-radius: var(--action-radius) !important;
        min-height: var(--action-height) !important;
        font-weight: var(--action-font-weight) !important;
        box-shadow: 0 6px 16px rgba(20, 184, 166, 0.18) !important;
    }

    [class*="st-key-action-primary-"] div.stButton > button:hover,
    [class*="st-key-action-primary-"] div[data-testid="stFormSubmitButton"] > button:hover {
        background: var(--action-primary-bg-hover) !important;
        border-color: var(--action-primary-border) !important;
    }

    [class*="st-key-action-secondary-"] div.stButton > button,
    [class*="st-key-action-secondary-"] div[data-testid="stFormSubmitButton"] > button {
        background: var(--action-secondary-bg) !important;
        color: var(--action-secondary-text) !important;
        border: 1px solid var(--action-secondary-border) !important;
        border-radius: var(--action-radius) !important;
        min-height: var(--action-height) !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    [class*="st-key-action-secondary-"] div.stButton > button:hover,
    [class*="st-key-action-secondary-"] div[data-testid="stFormSubmitButton"] > button:hover {
        background: var(--action-secondary-bg-hover) !important;
        border-color: var(--action-secondary-border-hover) !important;
        color: #F8FAFC !important;
    }

    [class*="st-key-action-danger-"] div.stButton > button,
    [class*="st-key-action-danger-"] div[data-testid="stFormSubmitButton"] > button {
        background: var(--action-danger-bg) !important;
        color: var(--action-danger-text) !important;
        border: 1px solid var(--action-danger-border) !important;
        border-radius: var(--action-radius) !important;
        min-height: var(--action-height) !important;
        font-weight: var(--action-font-weight) !important;
    }

    [class*="st-key-action-danger-"] div.stButton > button:hover,
    [class*="st-key-action-danger-"] div[data-testid="stFormSubmitButton"] > button:hover {
        background: var(--action-danger-bg-hover) !important;
        border-color: var(--action-danger-border) !important;
    }

    [class*="st-key-action-"] div.stButton > button:disabled,
    [class*="st-key-action-"] div[data-testid="stFormSubmitButton"] > button:disabled {
        background: var(--action-disabled-bg) !important;
        color: var(--action-disabled-text) !important;
        border: 1px solid var(--action-disabled-border) !important;
        opacity: 1 !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
    }

    .action-hint {
        margin-top: 0.35rem;
        margin-bottom: 0.6rem;
        font-size: 0.92rem;
        color: #CBD5E1;
    }
    </style>
""", unsafe_allow_html=True)


def normalizar_nome_comparacao(nome: str) -> str:
    nome = unicodedata.normalize("NFKD", str(nome))
    nome = "".join(ch for ch in nome if not unicodedata.combining(ch))
    nome = " ".join(nome.split())
    return nome.strip().upper()


def formatar_df_visual_numeros_inteiros(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df_fmt = df.copy()
    for col in ["Nota", "Velocidade", "Movimentação"]:
        if col in df_fmt.columns:
            def _to_int_visual(v):
                try:
                    if pd.isna(v):
                        return v
                except Exception:
                    pass
                try:
                    return int(round(float(v)))
                except Exception:
                    return v
            df_fmt[col] = df_fmt[col].apply(_to_int_visual)
    return df_fmt


def registro_valido_para_sorteio(row: pd.Series) -> bool:
    nome = str(row.get("Nome", "")).strip()
    posicao = str(row.get("Posição", "")).strip().upper()

    nota = pd.to_numeric(pd.Series([row.get("Nota")]), errors="coerce").iloc[0]
    velocidade = pd.to_numeric(pd.Series([row.get("Velocidade")]), errors="coerce").iloc[0]
    movimentacao = pd.to_numeric(pd.Series([row.get("Movimentação")]), errors="coerce").iloc[0]

    if not nome:
        return False
    if posicao not in ["D", "M", "A"]:
        return False
    if pd.isna(nota) or nota < 1 or nota > 10:
        return False
    if pd.isna(velocidade) or velocidade < 1 or velocidade > 5:
        return False
    if pd.isna(movimentacao) or movimentacao < 1 or movimentacao > 5:
        return False

    return True


def diagnosticar_nomes_bloqueados_para_sorteio(df_base: pd.DataFrame, nomes_confirmados: list[str]) -> list[dict]:
    if df_base is None or df_base.empty:
        return [{"nome": nome, "motivos": ["sem registro na base atual"]} for nome in nomes_confirmados]

    bloqueios = []
    for nome in nomes_confirmados:
        df_nome = df_base[df_base["Nome"] == nome].copy()
        if df_nome.empty:
            bloqueios.append({"nome": nome, "motivos": ["sem registro na base atual"]})
            continue

        total_registros = len(df_nome)
        registros_validos = int(df_nome.apply(registro_valido_para_sorteio, axis=1).sum())
        registros_invalidos = total_registros - registros_validos

        motivos = []
        if total_registros > 1:
            motivos.append("duplicado na base")
        if registros_invalidos > 0:
            motivos.append("com inconsistência na base")
        if registros_validos == 0:
            motivos.append("sem registro válido para sorteio")

        if motivos:
            bloqueios.append({"nome": nome, "motivos": motivos})

    return bloqueios


def preparar_df_sorteio(df_base: pd.DataFrame, nomes_confirmados: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    bloqueios = diagnosticar_nomes_bloqueados_para_sorteio(df_base, nomes_confirmados)
    if bloqueios:
        return pd.DataFrame(), bloqueios

    if df_base is None or df_base.empty:
        return pd.DataFrame(), [{"nome": nome, "motivos": ["sem registro na base atual"]} for nome in nomes_confirmados]

    df_lista = df_base[df_base["Nome"].isin(nomes_confirmados)].copy()
    if df_lista.empty:
        return pd.DataFrame(), [{"nome": nome, "motivos": ["sem registro na base atual"]} for nome in nomes_confirmados]

    df_validos = df_lista[df_lista.apply(registro_valido_para_sorteio, axis=1)].copy()
    df_validos = df_validos.drop_duplicates(subset=["Nome"], keep="last")

    return df_validos.reset_index(drop=True), []


def valor_slider_corrigir(v, minimo: int, maximo: int, fallback: int) -> int:
    num = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    if pd.isna(num):
        return fallback
    return max(minimo, min(maximo, int(round(float(num)))))


def atualizar_integridade_base_no_estado(logic):
    if hasattr(logic, "diagnosticar_inconsistencias_base"):
        st.session_state.base_inconsistencias_carregamento = logic.diagnosticar_inconsistencias_base(
            st.session_state.df_base
        )
    if hasattr(logic, "listar_registros_inconsistentes"):
        st.session_state.base_registros_inconsistentes_carregamento = (
            logic.listar_registros_inconsistentes(st.session_state.df_base).to_dict("records")
        )
    else:
        st.session_state.base_registros_inconsistentes_carregamento = []


def render_correcao_inline_bloqueios_base(logic, lista_texto: str, nomes_bloqueados_base: list[dict]):
    if not nomes_bloqueados_base:
        return

    st.caption("Você pode corrigir esses registros sem sair da revisão.")
    for item in nomes_bloqueados_base:
        nome = item["nome"]
        motivos = item.get("motivos", [])
        df_nome = st.session_state.df_base.copy()
        if df_nome.empty:
            continue

        df_nome = df_nome[df_nome["Nome"] == nome].copy()
        if df_nome.empty:
            continue

        df_nome = df_nome.reset_index().rename(columns={"index": "_orig_index"})
        df_nome["_registro_valido"] = df_nome.apply(registro_valido_para_sorteio, axis=1)
        df_nome = df_nome.sort_values(
            by=["_registro_valido", "_orig_index"],
            ascending=[True, True]
        ).reset_index(drop=True)

        with st.expander(f"🛠️ Corrigir agora: {nome}", expanded=False):
            st.markdown(f"**Motivos detectados:** {'; '.join(motivos)}")
            st.caption("Remova o registro duplicado indesejado ou corrija os critérios do registro inconsistente.")

            for i, row in df_nome.iterrows():
                idx_original = int(row["_orig_index"])
                registro_valido = registro_valido_para_sorteio(row)
                status = "✅ Registro válido" if registro_valido else "⚠️ Registro inconsistente"

                st.markdown(f"**Registro {i + 1}** — {status}")
                col_info1, col_info2, col_info3, col_info4, col_info5 = st.columns(5)
                col_info1.markdown(f"**Nome**\n\n{row.get('Nome', '')}")
                col_info2.markdown(f"**Posição**\n\n{row.get('Posição', '')}")
                col_info3.markdown(f"**Nota**\n\n{row.get('Nota', '')}")
                col_info4.markdown(f"**Velocidade**\n\n{row.get('Velocidade', '')}")
                col_info5.markdown(f"**Movimentação**\n\n{row.get('Movimentação', '')}")

                if render_action_button(
                    "🗑️ Remover este registro da base",
                    key=f"remover_registro_bloqueado_{nome}_{idx_original}",
                    role="secondary",
                ):
                    st.session_state.df_base = (
                        st.session_state.df_base.drop(index=idx_original).reset_index(drop=True)
                    )
                    atualizar_integridade_base_no_estado(logic)
                    if lista_texto:
                        diagnosticar_lista_no_estado(logic, lista_texto)
                        st.session_state.revisao_lista_expandida = True
                    st.rerun()

                with st.form(f"form_corrigir_registro_{nome}_{idx_original}"):
                    posicao_atual = str(row.get("Posição", "")).strip().upper()
                    if posicao_atual not in ["D", "M", "A"]:
                        posicao_atual = "M"

                    pos_corr = st.selectbox(
                        "Posição",
                        ["D", "M", "A"],
                        index=["D", "M", "A"].index(posicao_atual),
                        key=f"corrigir_posicao_{nome}_{idx_original}",
                    )
                    nota_corr = st.slider(
                        "Nota",
                        1,
                        10,
                        valor_slider_corrigir(row.get("Nota"), 1, 10, 6),
                        key=f"corrigir_nota_{nome}_{idx_original}",
                    )
                    vel_corr = st.slider(
                        "Velocidade",
                        1,
                        5,
                        valor_slider_corrigir(row.get("Velocidade"), 1, 5, 3),
                        key=f"corrigir_velocidade_{nome}_{idx_original}",
                    )
                    mov_corr = st.slider(
                        "Movimentação",
                        1,
                        5,
                        valor_slider_corrigir(row.get("Movimentação"), 1, 5, 3),
                        key=f"corrigir_movimentacao_{nome}_{idx_original}",
                    )
                    salvar = st.form_submit_button("💾 Salvar correção neste registro")

                    if salvar:
                        st.session_state.df_base.loc[idx_original, "Posição"] = pos_corr
                        st.session_state.df_base.loc[idx_original, "Nota"] = nota_corr
                        st.session_state.df_base.loc[idx_original, "Velocidade"] = vel_corr
                        st.session_state.df_base.loc[idx_original, "Movimentação"] = mov_corr
                        atualizar_integridade_base_no_estado(logic)
                        if lista_texto:
                            diagnosticar_lista_no_estado(logic, lista_texto)
                            st.session_state.revisao_lista_expandida = True
                        st.rerun()

                st.markdown("---")



def listar_bloqueios_base_atual(df_base: pd.DataFrame) -> list[dict]:
    if df_base is None or df_base.empty:
        return []

    nomes = [
        nome for nome in df_base["Nome"].astype(str).tolist()
        if str(nome).strip()
    ]
    nomes_unicos = list(dict.fromkeys(nomes))
    return diagnosticar_nomes_bloqueados_para_sorteio(df_base, nomes_unicos)


def render_correcao_inline_etapa2(logic):
    bloqueios = listar_bloqueios_base_atual(st.session_state.df_base)
    if not bloqueios:
        return

    with st.expander("🛠️ Corrigir base agora", expanded=False):
        render_correcao_inline_bloqueios_base(
            logic,
            st.session_state.get("lista_texto_revisado", ""),
            bloqueios,
        )


def construir_assinatura_entrada_sorteio(lista_texto: str, n_times: int) -> str:
    cols = ["Nome", "Nota", "Posição", "Velocidade", "Movimentação"]

    df_base = st.session_state.get("df_base")
    if df_base is None or df_base.empty:
        base_json = "[]"
    else:
        df_base_ref = df_base.copy()
        for col in cols:
            if col not in df_base_ref.columns:
                df_base_ref[col] = ""
        base_json = (
            df_base_ref[cols]
            .astype(str)
            .to_json(orient="split", force_ascii=False)
        )

    novos_jogadores = st.session_state.get("novos_jogadores", [])
    if novos_jogadores:
        df_novos = pd.DataFrame(novos_jogadores)
        for col in cols:
            if col not in df_novos.columns:
                df_novos[col] = ""
        novos_json = (
            df_novos[cols]
            .astype(str)
            .to_json(orient="split", force_ascii=False)
        )
    else:
        novos_json = "[]"

    payload = {
        "lista_texto": lista_texto or "",
        "n_times": int(n_times),
        "criterios": obter_criterios_ativos(),
        "base_json": base_json,
        "novos_json": novos_json,
    }
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def invalidar_resultado_se_entrada_mudou(lista_texto: str, n_times: int):
    if "resultado" not in st.session_state:
        return

    assinatura_anterior = st.session_state.get("resultado_assinatura")
    if not assinatura_anterior:
        return

    assinatura_atual = construir_assinatura_entrada_sorteio(lista_texto, n_times)
    if assinatura_atual == assinatura_anterior:
        return

    st.session_state.pop("resultado", None)
    st.session_state.pop("resultado_contexto", None)
    st.session_state.resultado_assinatura = None
    st.session_state.scroll_para_resultado = False
    st.session_state.resultado_invalidado_msg = True


def render_section_header(titulo: str, subtitulo: str | None = None):
    st.markdown(
        f"<div class='section-title' style='color: var(--section-title-color);'>{titulo}</div>",
        unsafe_allow_html=True,
    )
    if subtitulo:
        st.markdown(
            f"<div class='section-subtitle' style='color: var(--section-subtitle-color);'>{subtitulo}</div>",
            unsafe_allow_html=True,
        )


def ensure_local_session_state():
    if "base_admin_carregada" not in st.session_state:
        st.session_state.base_admin_carregada = False
    if "base_inconsistencias_carregamento" not in st.session_state:
        st.session_state.base_inconsistencias_carregamento = {}
    if "base_registros_inconsistentes_carregamento" not in st.session_state:
        st.session_state.base_registros_inconsistentes_carregamento = []
    if "senha_admin_confirmada" not in st.session_state:
        st.session_state.senha_admin_confirmada = False
    if "ultima_senha_digitada" not in st.session_state:
        st.session_state.ultima_senha_digitada = ""
    if "qtd_jogadores_adicionados_manualmente" not in st.session_state:
        st.session_state.qtd_jogadores_adicionados_manualmente = 0
    if "cadastro_manual_expanded" not in st.session_state:
        st.session_state.cadastro_manual_expanded = False
    if "cadastro_manual_nome_existente" not in st.session_state:
        st.session_state.cadastro_manual_nome_existente = ""
    if "criterio_posicao" not in st.session_state:
        st.session_state.criterio_posicao = True
    if "criterio_nota" not in st.session_state:
        st.session_state.criterio_nota = True
    if "criterio_velocidade" not in st.session_state:
        st.session_state.criterio_velocidade = True
    if "criterio_movimentacao" not in st.session_state:
        st.session_state.criterio_movimentacao = True
    if "scroll_para_resultado" not in st.session_state:
        st.session_state.scroll_para_resultado = False
    if "resultado_assinatura" not in st.session_state:
        st.session_state.resultado_assinatura = None
    if "resultado_invalidado_msg" not in st.session_state:
        st.session_state.resultado_invalidado_msg = False

    if "criterio_posicao" not in st.session_state:
        st.session_state.criterio_posicao = True
    if "criterio_nota" not in st.session_state:
        st.session_state.criterio_nota = True
    if "criterio_velocidade" not in st.session_state:
        st.session_state.criterio_velocidade = True
    if "criterio_movimentacao" not in st.session_state:
        st.session_state.criterio_movimentacao = True


def abrir_expander_grupo():
    st.session_state.grupo_config_expanded = True


def abrir_expander_cadastro_manual():
    st.session_state.cadastro_manual_expanded = True


def grupo_config_deve_abrir() -> bool:
    return bool(
        st.session_state.get("grupo_config_expanded", False)
        or str(st.session_state.get("grupo_nome_pelada", "")).strip()
        or str(st.session_state.get("grupo_senha_admin", "")).strip()
        or st.session_state.get("senha_admin_confirmada", False)
    )


def render_action_button(
    label: str,
    *,
    key: str,
    role: str = "secondary",
    disabled: bool = False,
    use_primary_type: bool = False,
):
    with st.container(key=f"action-{role}-{key}"):
        button_type = "primary" if use_primary_type else "secondary"
        return st.button(
            label,
            key=key,
            disabled=disabled,
            type=button_type,
        )


def _titulo_expander(rotulo: str, status: str) -> str:
    return f"{rotulo} · {status}"


def resumo_expander_configuracao() -> str:
    nome_pelada = str(st.session_state.get("grupo_nome_pelada", "")).strip()
    base_admin_carregada = bool(st.session_state.get("base_admin_carregada", False) and st.session_state.get("is_admin", False))
    base_upload_carregada = bool(st.session_state.get("ultimo_arquivo")) and not st.session_state.get("is_admin", False)
    grupo_encontrado = bool(nome_pelada) and nome_pelada.upper() == str(NOME_PELADA_ADM).upper()
    nome_nao_encontrado = bool(nome_pelada) and not grupo_encontrado and not base_admin_carregada and not base_upload_carregada

    if base_admin_carregada:
        status = "Base admin carregada"
    elif base_upload_carregada:
        status = "Planilha própria carregada"
    elif grupo_encontrado:
        status = "Grupo encontrado"
    elif nome_nao_encontrado:
        status = "Nome não encontrado"
    else:
        status = "Sem base"

    return _titulo_expander("⚙️ Grupo e base", status)


def _qtd_adicoes_manuais() -> int:
    return int(st.session_state.get("qtd_jogadores_adicionados_manualmente", 0))


def resumo_expander_cadastro_manual() -> str:
    cadastro_guiado_ativo = bool(st.session_state.get("cadastro_guiado_ativo", False))
    cadastro_guiado_concluido = bool(
        st.session_state.get("revisao_pendente_pos_cadastro", False)
        and len(st.session_state.get("faltantes_cadastrados_na_rodada", [])) > 0
        and not cadastro_guiado_ativo
    )
    qtd_manual = _qtd_adicoes_manuais()

    if cadastro_guiado_ativo:
        status = "Cadastro guiado ativo"
    elif cadastro_guiado_concluido:
        status = "Faltantes cadastrados"
    elif qtd_manual > 0:
        status = f"{qtd_manual} adicionados"
    else:
        status = "Opcional"

    return _titulo_expander("📝 Cadastro manual", status)


def obter_criterios_ativos() -> dict:
    return {
        "pos": bool(st.session_state.get("criterio_posicao", True)),
        "nota": bool(st.session_state.get("criterio_nota", True)),
        "vel": bool(st.session_state.get("criterio_velocidade", True)),
        "mov": bool(st.session_state.get("criterio_movimentacao", True)),
    }


def _criterios_estao_no_padrao() -> bool:
    criterios = obter_criterios_ativos()
    return (
        criterios["pos"],
        criterios["nota"],
        criterios["vel"],
        criterios["mov"],
    ) == (True, True, True, True)


def resumo_criterios_ativos() -> str:
    criterios = obter_criterios_ativos()
    ativos = []

    if criterios["pos"]:
        ativos.append("Posição")
    if criterios["nota"]:
        ativos.append("Nota")
    if criterios["vel"]:
        ativos.append("Velocidade")
    if criterios["mov"]:
        ativos.append("Movimentação")

    if len(ativos) == 4:
        return "Padrão · Posição, Nota, Velocidade e Movimentação"
    if not ativos:
        return "Personalizado · Nenhum critério ativo"

    return "Personalizado · " + ", ".join(ativos)


def resumo_expander_criterios() -> str:
    status = "Padrão" if _criterios_estao_no_padrao() else "Personalizado"
    return _titulo_expander("⚙️ Critérios", status)


def limpar_estado_revisao_lista():
    st.session_state.diagnostico_lista = None
    st.session_state.lista_revisada = None
    st.session_state.lista_revisada_confirmada = False
    st.session_state.lista_texto_revisado = ""
    st.session_state.revisao_lista_expandida = False


def diagnosticar_lista_no_estado(logic, lista_texto: str):
    processamento = logic.processar_lista(
        lista_texto,
        return_metadata=True,
        emit_warning=False,
    )

    if not processamento["jogadores"]:
        limpar_estado_revisao_lista()
        return None

    diagnostico = logic.diagnosticar_lista_para_sorteio(
        lista_texto,
        st.session_state.df_base,
        st.session_state.novos_jogadores,
    )

    df_final = st.session_state.df_base.copy()
    if st.session_state.novos_jogadores:
        df_final = pd.concat([df_final, pd.DataFrame(st.session_state.novos_jogadores)], ignore_index=True)

    nomes_bloqueados_base = diagnosticar_nomes_bloqueados_para_sorteio(
        df_final,
        diagnostico.get("lista_final_sugerida", []),
    )
    diagnostico["nomes_bloqueados_base"] = nomes_bloqueados_base
    diagnostico["tem_bloqueio_base"] = len(nomes_bloqueados_base) > 0

    st.session_state.diagnostico_lista = diagnostico
    st.session_state.lista_revisada = None
    st.session_state.lista_revisada_confirmada = False
    st.session_state.lista_texto_revisado = lista_texto
    st.session_state.revisao_lista_expandida = True
    return diagnostico


def render_revisao_lista(logic, lista_texto: str):
    diagnostico = st.session_state.diagnostico_lista
    pos_cadastro_pendente = (
        st.session_state.revisao_pendente_pos_cadastro
        and not st.session_state.cadastro_guiado_ativo
        and len(st.session_state.faltantes_revisao) == 0
        and len(st.session_state.faltantes_cadastrados_na_rodada) > 0
    )

    if not diagnostico and not pos_cadastro_pendente:
        return

    expanded = (
        st.session_state.revisao_lista_expandida
        or st.session_state.lista_revisada_confirmada
        or pos_cadastro_pendente
    )

    with st.expander("🔎 Revisão da lista", expanded=expanded):
        if pos_cadastro_pendente and not diagnostico:
            qtd_cadastrados = len(st.session_state.faltantes_cadastrados_na_rodada)
            if qtd_cadastrados == 1:
                st.success("O faltante desta revisão foi cadastrado com sucesso.")
            else:
                st.success(f"Os {qtd_cadastrados} faltantes desta revisão foram cadastrados com sucesso.")
            st.caption("Clique em **🔎 Revisar lista novamente** para atualizar a lista final e liberar a confirmação.")

            if render_action_button(
                "🔎 Revisar lista novamente",
                key="revisar_lista_pos_cadastro",
                role="primary",
                use_primary_type=True,
            ):
                diagnostico = diagnosticar_lista_no_estado(logic, lista_texto)
                st.session_state.revisao_pendente_pos_cadastro = False
                st.session_state.faltantes_cadastrados_na_rodada = []
                st.session_state.faltantes_revisao = []
                if diagnostico is None:
                    st.warning("Cole uma lista de jogadores para revisar novamente.")
                st.rerun()
            return

        tem_pendencia_revisao = (
            len(diagnostico.get("nao_encontrados", [])) > 0
            or len(diagnostico.get("duplicados", [])) > 0
            or len(diagnostico.get("correcoes_aplicadas", [])) > 0
            or diagnostico.get("tem_bloqueio_base", False)
            or st.session_state.cadastro_guiado_ativo
            or pos_cadastro_pendente
        )

        if diagnostico.get("tem_bloqueio_base", False):
            st.error("A lista não pode ser confirmada porque há nomes com duplicidade ou inconsistência na base atual.")
        elif st.session_state.lista_revisada_confirmada:
            st.success("Lista confirmada com sucesso. Agora você já pode sortear os times.")
        elif tem_pendencia_revisao:
            st.warning("Há pendências na lista. Resolva os pontos acima para continuar.")
        else:
            st.success("A lista está pronta para confirmação.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lidos", diagnostico["total_brutos"])
        col2.metric("Prontos", diagnostico["total_validos"])
        col3.metric("Correções", len(diagnostico["correcoes_aplicadas"]))
        col4.metric("Não encontrados", len(diagnostico["nao_encontrados"]))

        if diagnostico["correcoes_aplicadas"]:
            st.info("Alguns nomes foram ajustados com base na sua base atual.")
            for item in diagnostico["correcoes_aplicadas"]:
                st.markdown(f"- `{item['original']}` → `{item['corrigido']}`")

        if diagnostico["duplicados"]:
            st.warning("Encontramos nomes repetidos na lista. Apenas a primeira ocorrência será mantida na sugestão final.")
            for nome in diagnostico["duplicados"]:
                st.markdown(f"- {nome}")

        if diagnostico.get("nomes_bloqueados_base"):
            st.error("Os nomes abaixo têm duplicidade ou inconsistência na base atual e precisam ser corrigidos antes da confirmação:")
            for item in diagnostico["nomes_bloqueados_base"]:
                st.markdown(f"- **{item['nome']}** — {'; '.join(item['motivos'])}")
            render_correcao_inline_bloqueios_base(logic, lista_texto, diagnostico["nomes_bloqueados_base"])

        if diagnostico["nao_encontrados"]:
            st.error("Alguns nomes não foram encontrados na base atual.")
            for nome in diagnostico["nao_encontrados"]:
                st.markdown(f"- {nome}")
            st.caption("Cadastre esses jogadores agora no formulário abaixo e depois revise a lista novamente.")
            if render_action_button(
                "➕ Cadastrar faltantes agora",
                key="revisao_cadastrar_faltantes",
                role="primary",
                use_primary_type=True,
            ):
                st.session_state.faltantes_revisao = diagnostico["nao_encontrados"].copy()
                st.session_state.cadastro_guiado_ativo = True
                st.session_state.faltantes_cadastrados_na_rodada = []
                st.session_state.revisao_pendente_pos_cadastro = False
                st.session_state.lista_revisada_confirmada = False
                st.session_state.lista_revisada = None
                st.session_state.revisao_lista_expandida = True
                st.rerun()

        if st.session_state.cadastro_guiado_ativo and st.session_state.faltantes_revisao:
            faltantes_restantes = st.session_state.faltantes_revisao
            faltantes_feitos = st.session_state.faltantes_cadastrados_na_rodada
            nome_atual = faltantes_restantes[0]
            total_rodada = len(faltantes_restantes) + len(faltantes_feitos)
            indice_atual = len(faltantes_feitos) + 1
            ultimo_da_fila = len(faltantes_restantes) == 1

            st.info(
                f"Cadastro guiado iniciado — jogador {indice_atual} de {total_rodada}: **{nome_atual}**"
            )
            st.markdown(f"**Cadastrando agora:** {nome_atual}")

            with st.form("form_add_manual_guiado_inline"):
                p_m = st.selectbox("Posição", ["M", "A", "D"], key="guiado_inline_posicao")
                n_m = st.slider("Nota", 1, 10, 6, key="guiado_inline_nota")
                v_m = st.slider("Velocidade", 1, 5, 3, key="guiado_inline_velocidade")
                mv_m = st.slider("Movimentação", 1, 5, 3, key="guiado_inline_movimentacao")
                label_submit = "Salvar e concluir" if ultimo_da_fila else "Salvar e próximo faltante"

                with st.container(key="action-primary-form-salvar-faltante"):
                    submit_guiado = st.form_submit_button(label_submit)

                if submit_guiado:
                    novo_nome = logic.formatar_nome_visual(nome_atual)
                    novo = {
                        'Nome': novo_nome,
                        'Nota': n_m,
                        'Posição': p_m,
                        'Velocidade': v_m,
                        'Movimentação': mv_m,
                    }
                    st.session_state.df_base.loc[len(st.session_state.df_base)] = novo
                    st.session_state.faltantes_cadastrados_na_rodada.append(novo_nome)
                    st.session_state.faltantes_revisao.pop(0)
                    st.session_state.lista_revisada_confirmada = False
                    st.session_state.lista_revisada = None
                    st.session_state.diagnostico_lista = None
                    st.session_state.revisao_lista_expandida = True

                    if not st.session_state.faltantes_revisao:
                        st.session_state.cadastro_guiado_ativo = False
                        st.session_state.faltantes_revisao = []
                        st.session_state.revisao_pendente_pos_cadastro = True

                    st.rerun()

        if diagnostico["ignorados"]:
            with st.expander("Itens ignorados na leitura", expanded=False):
                for item in diagnostico["ignorados"]:
                    st.markdown(f"- {item}")

        st.text_area(
            "Lista final sugerida",
            value="\n".join(diagnostico["lista_final_sugerida"]),
            height=140,
            disabled=True,
            key="lista_final_sugerida_preview",
        )

        pode_confirmar = (
            diagnostico["total_validos"] > 0
            and not diagnostico["tem_nao_encontrados"]
            and not diagnostico.get("tem_bloqueio_base", False)
            and not st.session_state.cadastro_guiado_ativo
        )
        if pode_confirmar and not st.session_state.lista_revisada_confirmada:
            if render_action_button(
                "✅ Confirmar lista final",
                key="confirmar_lista_revisada",
                role="primary",
                use_primary_type=True,
            ):
                st.session_state.lista_revisada = diagnostico["lista_final_sugerida"]
                st.session_state.lista_revisada_confirmada = True
                st.session_state.revisao_lista_expandida = False
                st.rerun()


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





def estilo_celulas_inconsistentes(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(index=getattr(df, "index", []), columns=getattr(df, "columns", []))

    estilos = pd.DataFrame("", index=df.index, columns=df.columns)
    destaque = "background-color: rgba(248, 113, 113, 0.22); font-weight: 700;"

    if "Nome" in df.columns:
        nomes = df["Nome"].fillna("").astype(str).str.strip()
        estilos.loc[nomes.eq(""), "Nome"] = destaque

    if "Posição" in df.columns:
        posicoes = df["Posição"].fillna("").astype(str).str.strip().str.upper()
        estilos.loc[~posicoes.isin(["D", "M", "A", "G"]), "Posição"] = destaque

    if "Nota" in df.columns:
        nota = pd.to_numeric(df["Nota"], errors="coerce")
        estilos.loc[nota.isna() | (nota < 1) | (nota > 10), "Nota"] = destaque

    if "Velocidade" in df.columns:
        velocidade = pd.to_numeric(df["Velocidade"], errors="coerce")
        estilos.loc[velocidade.isna() | (velocidade < 1) | (velocidade > 5), "Velocidade"] = destaque

    if "Movimentação" in df.columns:
        movimentacao = pd.to_numeric(df["Movimentação"], errors="coerce")
        estilos.loc[movimentacao.isna() | (movimentacao < 1) | (movimentacao > 5), "Movimentação"] = destaque

    return estilos


def render_base_inconsistencias_expander():
    registros = st.session_state.get("base_registros_inconsistentes_carregamento", [])
    if not registros:
        return

    df_inconsistentes = pd.DataFrame(registros)
    if df_inconsistentes.empty:
        return

    with st.expander("⚠️ Registros com inconsistências", expanded=False):
        st.caption("Os registros abaixo foram carregados, mas merecem revisão antes do uso.")
        df_inconsistentes_display = df_inconsistentes.copy()
        styler = df_inconsistentes_display.style.apply(estilo_celulas_inconsistentes, axis=None)
        st.dataframe(
            styler,
            width="stretch",
            hide_index=True,
        )

def total_inconsistencias_base(inconsistencias: dict) -> int:
    if not inconsistencias:
        return 0
    return int(sum(v for v in inconsistencias.values() if isinstance(v, (int, float))))


def resumo_inconsistencias_base(inconsistencias: dict) -> str:
    if not inconsistencias:
        return ""

    mensagens = []
    if inconsistencias.get("nomes_vazios", 0) > 0:
        mensagens.append(f'{inconsistencias["nomes_vazios"]} nome(s) vazio(s)')
    if inconsistencias.get("posicoes_invalidas", 0) > 0:
        mensagens.append(f'{inconsistencias["posicoes_invalidas"]} posição(ões) inválida(s)')
    if inconsistencias.get("notas_invalidas", 0) > 0:
        mensagens.append(f'{inconsistencias["notas_invalidas"]} nota(s) fora da faixa 1–10')
    if inconsistencias.get("velocidades_invalidas", 0) > 0:
        mensagens.append(f'{inconsistencias["velocidades_invalidas"]} velocidade(s) fora da faixa 1–5')
    if inconsistencias.get("movimentacoes_invalidas", 0) > 0:
        mensagens.append(f'{inconsistencias["movimentacoes_invalidas"]} movimentação(ões) fora da faixa 1–5')

    return "; ".join(mensagens)

def render_base_integrity_alert():
    df_base = st.session_state.df_base

    if df_base.empty:
        return

    inconsistencias = st.session_state.get("base_inconsistencias_carregamento", {})
    total_inconsistencias = total_inconsistencias_base(inconsistencias)
    resumo_inconsistencias = resumo_inconsistencias_base(inconsistencias)

    nomes_normalizados = df_base["Nome"].astype(str).apply(normalizar_nome_comparacao)
    duplicados = nomes_normalizados[nomes_normalizados.duplicated(keep=False)]

    if not duplicados.empty:
        qtd_nomes_duplicados = duplicados.nunique()
        mensagem = (
            f"Atenção: a base atual contém {qtd_nomes_duplicados} nome(s) duplicado(s). "
            "Use o filtro “Mostrar apenas duplicados” para revisar esses registros."
        )
        if total_inconsistencias > 0 and resumo_inconsistencias:
            mensagem += f" Também foram detectadas inconsistências no carregamento: {resumo_inconsistencias}."
        st.warning(mensagem)
        return

    if total_inconsistencias > 0:
        st.warning(
            "Atenção: a base atual foi carregada com inconsistências nos dados. "
            f"Foram detectados: {resumo_inconsistencias}."
        )
        return

    st.caption("Integridade da base: limpa.")


def render_group_config_expander(logic, nome_pelada_adm: str, senha_adm: str) -> str:
    if "grupo_config_expanded" not in st.session_state:
        st.session_state.grupo_config_expanded = False

    with st.expander(
        resumo_expander_configuracao(),
        expanded=grupo_config_deve_abrir(),
    ):
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

        if not (st.session_state.base_admin_carregada and st.session_state.is_admin):
            st.button(
                "🔎 Verificar grupo",
                key="grupo_verificar_nome",
                on_click=abrir_expander_grupo,
            )

        if grupo_admin:
            if st.session_state.base_admin_carregada and st.session_state.is_admin:
                st.success("Base admin carregada com sucesso.")
                origem_base = "Base original (Admin)"
            else:
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

        admin_base_carregada = st.session_state.base_admin_carregada

        if origem_base == "Base original (Admin)":
            senha_atual = st.session_state.get("grupo_senha_admin", "")
            if st.session_state.ultima_senha_digitada != senha_atual:
                st.session_state.senha_admin_confirmada = False
                st.session_state.ultima_senha_digitada = senha_atual

        if not admin_base_carregada:
            st.markdown("---")
            st.markdown("**📂 Banco de dados**")
            st.caption("Escolha como carregar sua base ou siga para a etapa 3.")

            if origem_base == "Base original (Admin)":
                senha = st.text_input(
                    "Senha de Acesso:",
                    type="password",
                    key="grupo_senha_admin",
                )
                if st.button(
                    "🔐 Confirmar senha e carregar base",
                    key="grupo_confirmar_senha",
                    on_click=abrir_expander_grupo,
                ):
                    if not grupo_admin:
                        st.session_state.is_admin = False
                        st.session_state.base_admin_carregada = False
                        st.session_state.senha_admin_confirmada = False
                        if nome_informado:
                            st.error(
                                "Base não encontrada para esse nome. Corrija o nome, envie uma planilha própria ou siga para a etapa 3."
                            )
                        else:
                            st.warning(
                                "Informe um grupo válido para usar a base administrada ou siga para a etapa 3."
                            )
                    elif senha != str(senha_adm):
                        st.session_state.senha_admin_confirmada = False
                        st.session_state.ultima_senha_digitada = senha
                        st.session_state.is_admin = False
                        st.session_state.base_admin_carregada = False
                        st.error("Senha incorreta")
                    else:
                        st.session_state.senha_admin_confirmada = True
                        st.session_state.ultima_senha_digitada = senha
                        st.session_state.df_base = logic.carregar_dados_originais()
                        st.session_state.novos_jogadores = []
                        st.session_state.is_admin = True
                        st.session_state.base_admin_carregada = True
                        st.session_state.ultimo_arquivo = None
                        st.session_state.qtd_jogadores_adicionados_manualmente = 0
                        if hasattr(logic, "diagnosticar_inconsistencias_base"):
                            st.session_state.base_inconsistencias_carregamento = logic.diagnosticar_inconsistencias_base(
                                st.session_state.df_base
                            )
                            st.session_state.base_registros_inconsistentes_carregamento = (
                                logic.listar_registros_inconsistentes(st.session_state.df_base).to_dict("records")
                                if hasattr(logic, "listar_registros_inconsistentes")
                                else []
                            )
                        st.session_state.grupo_config_expanded = False
                        st.success(f"Base carregada: {len(st.session_state.df_base)} jogadores.")
                        st.rerun()
                if not st.session_state.senha_admin_confirmada:
                    st.caption("Depois de informar a senha, toque em **Confirmar senha e carregar base**.")
            else:
                st.write("Já tem uma planilha? Envie o arquivo abaixo e depois clique em **Carregar base de dados**.")
                uploaded_file = st.file_uploader(
                    "Enviar planilha Excel",
                    type=["xlsx"],
                    label_visibility="collapsed",
                    key="grupo_upload_planilha",
                )

        if (
            not admin_base_carregada
            and origem_base != "Base original (Admin)"
            and st.button(
                "📥 Carregar base de dados",
                key="grupo_carregar_base",
                on_click=abrir_expander_grupo,
            )
        ):
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
                    st.session_state.base_admin_carregada = False
                    st.session_state.ultimo_arquivo = uploaded_file.name
                    st.session_state.qtd_jogadores_adicionados_manualmente = 0
                    st.session_state.senha_admin_confirmada = False
                    if hasattr(logic, "diagnosticar_inconsistencias_base"):
                        st.session_state.base_inconsistencias_carregamento = logic.diagnosticar_inconsistencias_base(
                            st.session_state.df_base
                        )
                        st.session_state.base_registros_inconsistentes_carregamento = (
                            logic.listar_registros_inconsistentes(st.session_state.df_base).to_dict("records")
                            if hasattr(logic, "listar_registros_inconsistentes")
                            else []
                        )
                    st.session_state.grupo_config_expanded = False
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
                    st.session_state.base_admin_carregada = False
                    st.session_state.ultimo_arquivo = None
                    st.session_state.qtd_jogadores_adicionados_manualmente = 0
                    st.session_state.senha_admin_confirmada = False
                    st.session_state.base_inconsistencias_carregamento = {}
                    st.session_state.base_registros_inconsistentes_carregamento = []
                    st.session_state.grupo_config_expanded = True
                    st.rerun()

    return nome_pelada


def render_manual_card(logic, nome_pelada: str):
    with st.expander(
        resumo_expander_cadastro_manual(),
        expanded=st.session_state.get("cadastro_manual_expanded", False),
    ):
        st.caption(
            "Use esta etapa para montar sua base do zero ou complementar a base atual com novos jogadores."
        )
        st.caption("Quer montar ou editar a base fora do app? Baixe o modelo de planilha abaixo.")

        df_exemplo = logic.criar_exemplo()
        excel_exemplo = logic.converter_df_para_excel(df_exemplo)
        st.download_button(
            label="📥 Baixar planilha modelo",
            data=excel_exemplo,
            file_name="modelo_pelada.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Baixe este arquivo para ver como preencher a planilha no formato correto.",
            key="manual_baixar_modelo_planilha",
        )

        with st.form("form_add_manual"):
            col_a, col_b = st.columns(2)
            nome_m = col_a.text_input("Nome")
            p_m = col_b.selectbox("Posição", ["M", "A", "D"])
            n_m = st.slider("Nota", 1, 10, 6)
            v_m = st.slider("Velocidade", 1, 5, 3)
            mv_m = st.slider("Movimentação", 1, 5, 3)
            submit_manual = st.form_submit_button(
                "Adicionar à Base",
                on_click=abrir_expander_cadastro_manual,
            )
            if submit_manual:
                if nome_m:
                    novo_nome = logic.formatar_nome_visual(nome_m)
                    nomes_existentes = {
                        normalizar_nome_comparacao(nome)
                        for nome in st.session_state.df_base["Nome"].astype(str).tolist()
                    }
                    nomes_existentes.update(
                        {
                            normalizar_nome_comparacao(nome)
                            for nome in pd.Series(st.session_state.get("novos_jogadores", []))
                            .apply(lambda x: x.get("Nome") if isinstance(x, dict) else None)
                            .dropna()
                            .tolist()
                        }
                    )

                    if normalizar_nome_comparacao(novo_nome) in nomes_existentes:
                        st.session_state.cadastro_manual_expanded = True
                        st.session_state.cadastro_manual_nome_existente = novo_nome
                        st.error(
                            "Esse nome já existe na base atual. Revise a grafia ou edite o registro existente antes de adicionar novamente."
                        )
                    else:
                        novo = {
                            'Nome': novo_nome,
                            'Nota': n_m,
                            'Posição': p_m,
                            'Velocidade': v_m,
                            'Movimentação': mv_m,
                        }
                        st.session_state.df_base.loc[len(st.session_state.df_base)] = novo
                        st.session_state.qtd_jogadores_adicionados_manualmente += 1
                        st.session_state.cadastro_manual_expanded = False
                        st.session_state.cadastro_manual_nome_existente = ""
                        st.success(f"{novo_nome} salvo!")
                else:
                    st.session_state.cadastro_manual_expanded = True
                    st.session_state.cadastro_manual_nome_existente = ""
                    st.error("Digite um nome.")

        nome_existente = st.session_state.get("cadastro_manual_nome_existente", "")
        if nome_existente:
            st.warning("Você pode editar ou remover esse registro existente sem sair da etapa 3.")
            render_correcao_inline_bloqueios_base(
                logic,
                st.session_state.get("lista_texto_revisado", ""),
                [{"nome": nome_existente, "motivos": ["nome já existente na base"]}],
            )

        if (
            not st.session_state.cadastro_guiado_ativo
            and st.session_state.revisao_pendente_pos_cadastro
            and st.session_state.faltantes_cadastrados_na_rodada
        ):
            st.success(
                "Todos os faltantes desta revisão foram cadastrados. Agora revise a lista novamente para liberar o sorteio."
            )
            st.caption(
                f"Cadastrados nesta rodada: {', '.join(st.session_state.faltantes_cadastrados_na_rodada)}"
            )

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
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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

    busca_nome = st.text_input(
        "Buscar jogador na base",
        placeholder="Ex: Cleiton",
        key="preview_busca_nome",
    ).strip()

    mostrar_apenas_duplicados = st.checkbox(
        "Mostrar apenas duplicados",
        key="preview_mostrar_apenas_duplicados",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        ordenar_por = st.selectbox(
            "Ordenar por",
            ["Nome", "Posição", "Nota"],
            key="preview_ordenar_por"
        )
    with col2:
        opcoes_mostrar = ["Todos", 10, 20, 50, 100]
        max_linhas = st.selectbox(
            "Mostrar",
            opcoes_mostrar,
            index=0,
            key="preview_max_linhas"
        )

    ascending = True
    if ordenar_por == "Nota":
        ascending = False

    df_preview = df_base.copy()
    nomes_normalizados_base = df_base["Nome"].astype(str).apply(normalizar_nome_comparacao)
    nomes_duplicados_normalizados = set(
        nomes_normalizados_base[nomes_normalizados_base.duplicated(keep=False)].tolist()
    )

    if mostrar_apenas_duplicados:
        nomes_normalizados = df_preview["Nome"].astype(str).apply(normalizar_nome_comparacao)
        mascara_duplicados = nomes_normalizados.isin(nomes_duplicados_normalizados)
        df_preview = df_preview[mascara_duplicados]

    if busca_nome:
        df_preview = df_preview[
            df_preview["Nome"].astype(str).str.contains(busca_nome, case=False, na=False)
        ]

    if busca_nome and mostrar_apenas_duplicados:
        st.caption(f"{len(df_preview)} registro(s) encontrado(s) entre os nomes duplicados.")
    elif busca_nome:
        st.caption(f"{len(df_preview)} jogador(es) encontrado(s).")
    elif mostrar_apenas_duplicados:
        nomes_normalizados = df_preview["Nome"].astype(str).apply(normalizar_nome_comparacao)
        qtd_nomes_duplicados = nomes_normalizados.nunique()
        st.caption(f"{len(df_preview)} registro(s) exibido(s) · {qtd_nomes_duplicados} nome(s) duplicado(s).")

    df_preview["_registro_valido"] = df_preview.apply(registro_valido_para_sorteio, axis=1)
    if ordenar_por == "Nome":
        df_preview = df_preview.sort_values(
            by=["Nome", "_registro_valido"],
            ascending=[True, True]
        ).reset_index(drop=True)
    else:
        df_preview = df_preview.sort_values(
            by=[ordenar_por, "_registro_valido"],
            ascending=[ascending, True]
        ).reset_index(drop=True)

    df_preview = df_preview.drop(columns=["_registro_valido"], errors="ignore")

    if max_linhas != "Todos":
        df_preview = df_preview.head(int(max_linhas))

    def destacar_linha_duplicada(linha):
        chave = normalizar_nome_comparacao(linha["Nome"])
        if chave in nomes_duplicados_normalizados:
            return ["background-color: rgba(250, 204, 21, 0.12);"] * len(linha)
        return [""] * len(linha)

    df_preview_display = formatar_df_visual_numeros_inteiros(df_preview)

    st.dataframe(
        df_preview_display.style.apply(destacar_linha_duplicada, axis=1),
        width="stretch",
        hide_index=True
    )


# --- FRONTEND ---
def main():
    logic = PeladaLogic()
    st.title("⚽ Sorteador Pelada PRO")
    botao_instalar_app()

    init_session_state(logic)
    ensure_local_session_state()

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
    render_base_integrity_alert()
    render_base_inconsistencias_expander()
    render_correcao_inline_etapa2(logic)

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

    processamento_previa = logic.processar_lista(
        lista_texto,
        return_metadata=True,
        emit_warning=False,
    )
    qtd_nomes_informados = len(processamento_previa["jogadores"])
    qtd_itens_ignorados = len(processamento_previa.get("ignorados", []))

    if qtd_nomes_informados > 0 or qtd_itens_ignorados > 0:
        if qtd_itens_ignorados == 0:
            st.caption(
                f"Leitura atual: {qtd_nomes_informados} nomes reconhecidos · nenhuma linha ignorada."
            )
        else:
            st.caption(
                f"Leitura atual: {qtd_nomes_informados} nomes reconhecidos · {qtd_itens_ignorados} linhas ignoradas."
            )

    col1, col2 = st.columns(2)
    n_times = col1.selectbox("Nº Times:", range(2, 11), index=1)

    invalidar_resultado_se_entrada_mudou(lista_texto, n_times)

    if qtd_nomes_informados > 0:
        base_por_time = qtd_nomes_informados // n_times
        resto_times = qtd_nomes_informados % n_times

        if resto_times == 0:
            st.caption(
                f"Distribuição estimada: {n_times} time(s) · {base_por_time} por time."
            )
        else:
            qtd_times_menores = n_times - resto_times
            qtd_times_maiores = resto_times
            st.caption(
                f"Prévia: {qtd_nomes_informados} nome(s) lido(s) da lista · {n_times} time(s) · "
                f"{qtd_times_menores} time(s) com {base_por_time} e {qtd_times_maiores} time(s) com {base_por_time + 1}."
            )

        if qtd_nomes_informados < n_times * 2:
            st.warning(
                "Atenção: há poucos nomes na lista para a quantidade de times escolhida. O sorteio pode ficar pouco equilibrado."
            )

    lista_alterada_pos_revisao = (
        bool(st.session_state.lista_texto_revisado)
        and lista_texto != st.session_state.lista_texto_revisado
    )
    if lista_alterada_pos_revisao and (
        st.session_state.diagnostico_lista is not None
        or st.session_state.lista_revisada_confirmada
    ):
        limpar_estado_revisao_lista()
        st.info("A lista foi alterada após a última revisão. Revise novamente antes de sortear.")

    review_role = "primary"
    if st.session_state.diagnostico_lista is not None or st.session_state.lista_revisada_confirmada:
        review_role = "secondary"

    revisar_lista = render_action_button(
        "🔎 Revisar lista",
        key="acao_revisar_lista",
        role=review_role,
        use_primary_type=(review_role == "primary"),
    )

    if revisar_lista:
        diagnostico = diagnosticar_lista_no_estado(logic, lista_texto)
        if diagnostico is None:
            st.warning("Cole uma lista de jogadores para revisar antes do sorteio.")

    render_revisao_lista(logic, lista_texto)

    render_section_header(
        "5. Critérios do sorteio",
        "Escolha quais características devem ser equilibradas entre os times."
    )
    with st.expander(resumo_expander_criterios(), expanded=False):
        st.checkbox("Equilibrar Posição", value=True, key="criterio_posicao")
        st.checkbox("Equilibrar Nota", value=True, key="criterio_nota")
        st.checkbox("Equilibrar Velocidade", value=True, key="criterio_velocidade")
        st.checkbox("Equilibrar Movimentação", value=True, key="criterio_movimentacao")

        criterios_ativos = obter_criterios_ativos()
        qtd_ativos = sum(criterios_ativos.values())

        st.caption(f"Configuração ativa: {resumo_criterios_ativos()}")

        if qtd_ativos == 4:
            st.caption(
                "Modo padrão: o sorteio tentará equilibrar posição, nota, velocidade e movimentação."
            )
        elif qtd_ativos == 0:
            st.warning(
                "Nenhum critério está ativo. O sorteio ficará mais próximo de uma divisão aleatória simples."
            )
        else:
            st.caption(
                "Modo personalizado: o sorteio equilibrará apenas os critérios selecionados."
            )

    lista_revisada_ok = bool(st.session_state.diagnostico_lista is not None)
    lista_confirmada_ok = bool(
        st.session_state.lista_revisada_confirmada and st.session_state.lista_revisada
    )
    base_pronta_ok = bool(
        not st.session_state.df_base.empty or st.session_state.novos_jogadores
    )

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.42); border: 1px solid #334155; border-radius: 12px; padding: 12px 14px; margin: 0.35rem 0 0.8rem 0;">
            <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 8px;">Pronto para sortear?</div>
            <div style="color: #E2E8F0; margin-bottom: 4px;">{"✅" if lista_revisada_ok else "❌"} Lista revisada</div>
            <div style="color: #E2E8F0; margin-bottom: 4px;">{"✅" if lista_confirmada_ok else "❌"} Lista confirmada</div>
            <div style="color: #E2E8F0;">{"✅" if base_pronta_ok else "❌"} Base pronta</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pode_sortear_agora = bool(
        st.session_state.lista_revisada_confirmada
        and st.session_state.lista_revisada
        and not st.session_state.cadastro_guiado_ativo
        and st.session_state.diagnostico_lista
        and not st.session_state.diagnostico_lista.get("tem_nao_encontrados", False)
        and not st.session_state.diagnostico_lista.get("tem_bloqueio_base", False)
    )
    diagnostico_atual = st.session_state.diagnostico_lista or {}

    if st.session_state.get("resultado_invalidado_msg", False):
        st.info("O resultado anterior foi invalidado porque os dados de entrada mudaram. Faça um novo sorteio.")
        st.session_state.resultado_invalidado_msg = False

    if st.session_state.cadastro_guiado_ativo:
        st.caption('Próximo passo: conclua o cadastro guiado dos jogadores faltantes e depois revise a lista novamente.')
    elif not lista_revisada_ok:
        st.caption('Próximo passo: clique em "🔎 Revisar lista" para verificar nomes e pendências.')
    elif diagnostico_atual.get("tem_nao_encontrados", False):
        st.caption('Próximo passo: clique em "➕ Cadastrar faltantes agora", conclua o cadastro e depois revise a lista novamente.')
    elif diagnostico_atual.get("tem_bloqueio_base", False):
        st.caption('Próximo passo: corrija os registros duplicados ou inconsistentes da base atual e depois revise a lista novamente.')
    elif not lista_confirmada_ok:
        st.caption('Próximo passo: em "🔎 Revisão da lista", clique em "✅ Confirmar lista final".')
    elif not base_pronta_ok:
        st.caption("Próximo passo: carregue uma base na etapa 1 ou complete os jogadores na etapa 3.")
    else:
        st.markdown(
            "<div class='action-hint'>Tudo pronto. Ajuste os critérios, se quiser, e clique em “🎲 SORTEAR TIMES”.</div>",
            unsafe_allow_html=True,
        )

    sortear_times = render_action_button(
        "🎲 SORTEAR TIMES",
        key="acao_sortear_times",
        role="primary",
        disabled=not pode_sortear_agora,
        use_primary_type=True,
    )

    if sortear_times:
        revisao_atual_valida = (
            st.session_state.lista_revisada_confirmada
            and st.session_state.lista_revisada is not None
            and lista_texto == st.session_state.lista_texto_revisado
        )

        if not revisao_atual_valida:
            diagnostico = diagnosticar_lista_no_estado(logic, lista_texto)
            if diagnostico is None:
                st.warning("Cole uma lista de jogadores para revisar antes do sorteio.")
            elif diagnostico["tem_nao_encontrados"]:
                st.warning("Existem nomes não encontrados. Cadastre esses jogadores na etapa 3 e confirme a revisão antes de sortear.")
            elif diagnostico.get("tem_bloqueio_base", False):
                st.error("Existem nomes da lista com duplicidade ou inconsistência na base atual. Corrija a base e revise a lista novamente antes de sortear.")
            else:
                st.info("Revise a lista e clique em **✅ Confirmar lista final** antes de sortear.")
        else:
            nomes_corrigidos = st.session_state.lista_revisada

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

                df_jogar, nomes_bloqueados_base = preparar_df_sorteio(df_final, nomes_corrigidos)

                if nomes_bloqueados_base:
                    detalhes = " | ".join(
                        [f"{item['nome']}: {'; '.join(item['motivos'])}" for item in nomes_bloqueados_base]
                    )
                    st.error(
                        "Não é possível realizar o sorteio porque há nomes com duplicidade ou inconsistência na base atual. "
                        f"Corrija a base primeiro. Detalhes: {detalhes}"
                    )
                else:
                    try:
                        with st.spinner('Sorteando...'):
                            criterios_ativos = obter_criterios_ativos()
                            times = logic.otimizar(df_jogar, n_times, criterios_ativos)
                            st.session_state.resultado = times
                            st.session_state.resultado_contexto = {
                                'qtd_jogadores': len(nomes_corrigidos),
                                'qtd_times': len([time for time in times if time]),
                                'criterios': criterios_ativos.copy(),
                            }
                            st.session_state.resultado_assinatura = construir_assinatura_entrada_sorteio(lista_texto, n_times)
                            st.session_state.resultado_invalidado_msg = False
                            st.session_state.scroll_para_resultado = True
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
        st.markdown('<div id="resultado-anchor"></div>', unsafe_allow_html=True)
        render_section_header(
            "6. Resultado",
            "Veja os times sorteados e copie o resultado para compartilhar."
        )

        if st.session_state.get("scroll_para_resultado", False):
            components.html(
                '''
                <script>
                const parentDoc = window.parent.document;
                const anchor = parentDoc.getElementById("resultado-anchor");
                if (anchor) {
                    anchor.scrollIntoView({ behavior: "smooth", block: "start" });
                }
                </script>
                ''',
                height=0,
            )
            st.session_state.scroll_para_resultado = False
        times = st.session_state.resultado
        odds = logic.calcular_odds(times)
        contexto_resultado = st.session_state.get('resultado_contexto', {})
        criterios_resultado = contexto_resultado.get('criterios', obter_criterios_ativos())

        criterios_ativos_legiveis = []
        if criterios_resultado.get('pos', False):
            criterios_ativos_legiveis.append('Posição')
        if criterios_resultado.get('nota', False):
            criterios_ativos_legiveis.append('Nota')
        if criterios_resultado.get('vel', False):
            criterios_ativos_legiveis.append('Velocidade')
        if criterios_resultado.get('mov', False):
            criterios_ativos_legiveis.append('Movimentação')

        modo_criterios = 'Padrão' if all(criterios_resultado.values()) else 'Personalizado'
        if modo_criterios == 'Padrão':
            criterios_ativos_texto = 'Todos os 4 critérios'
        else:
            criterios_ativos_texto = ', '.join(criterios_ativos_legiveis) if criterios_ativos_legiveis else 'Nenhum'
        qtd_jogadores_resultado = contexto_resultado.get('qtd_jogadores', len(st.session_state.get('lista_revisada', [])))
        qtd_times_resultado = contexto_resultado.get('qtd_times', len([time for time in times if time]))

        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.55); border: 1px solid #3b4a63; border-radius: 12px; padding: 10px 14px; margin: 0.35rem 0 0.75rem 0;">
                <div style="font-size: 0.98rem; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">Resumo do sorteio</div>
                <div style="color: #CBD5E1; margin-bottom: 3px;">👥 <span style="font-weight: 600;">Jogadores:</span> <span style="color: #F8FAFC; font-weight: 700;">{qtd_jogadores_resultado}</span></div>
                <div style="color: #CBD5E1; margin-bottom: 3px;">🧩 <span style="font-weight: 600;">Times:</span> <span style="color: #F8FAFC; font-weight: 700;">{qtd_times_resultado}</span></div>
                <div style="color: #CBD5E1; margin-bottom: 3px;">⚙️ <span style="font-weight: 600;">Critérios:</span> <span style="color: #F8FAFC; font-weight: 700;">{modo_criterios}</span></div>
                <div style="color: #CBD5E1;">✅ <span style="font-weight: 600;">Ativos:</span> <span style="color: #F8FAFC; font-weight: 700;">{criterios_ativos_texto}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
