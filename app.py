
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
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


def render_section_header(titulo: str, subtitulo: str | None = None):
    st.markdown(f"<div class='section-title'>{titulo}</div>", unsafe_allow_html=True)
    if subtitulo:
        st.markdown(f"<div class='section-subtitle'>{subtitulo}</div>", unsafe_allow_html=True)


def ensure_local_session_state():
    if "base_admin_carregada" not in st.session_state:
        st.session_state.base_admin_carregada = False
    if "senha_admin_confirmada" not in st.session_state:
        st.session_state.senha_admin_confirmada = False
    if "ultima_senha_digitada" not in st.session_state:
        st.session_state.ultima_senha_digitada = ""
    if "qtd_jogadores_adicionados_manualmente" not in st.session_state:
        st.session_state.qtd_jogadores_adicionados_manualmente = 0
    if "criterio_posicao" not in st.session_state:
        st.session_state.criterio_posicao = True
    if "criterio_nota" not in st.session_state:
        st.session_state.criterio_nota = True
    if "criterio_velocidade" not in st.session_state:
        st.session_state.criterio_velocidade = True
    if "criterio_movimentacao" not in st.session_state:
        st.session_state.criterio_movimentacao = True

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


def render_copy_button_with_feedback(texto: str, *, key: str = "copiar_whatsapp"):
    payload = json.dumps(texto)
    button_id = f"copy-button-{key}"
    label_id = f"copy-label-{key}"
    status_id = f"copy-status-{key}"

    components.html(
        f"""
        <div style="width: 100%;">
            <button
                id="{button_id}"
                style="
                    width: 100%;
                    min-height: 3.4rem;
                    border: 1px solid #4ADE80;
                    border-radius: 14px;
                    background: #57CF5F;
                    color: #F8FAFC;
                    font-weight: 800;
                    font-size: 0.98rem;
                    cursor: pointer;
                    box-shadow: 0 6px 16px rgba(34, 197, 94, 0.20);
                    transition: background 0.18s ease, transform 0.18s ease;
                    padding: 0.85rem 1rem;
                "
            >
                <span id="{label_id}">📋 COPIAR PARA WHATSAPP</span>
            </button>
            <div
                id="{status_id}"
                style="
                    display: none;
                    margin-top: 0.4rem;
                    text-align: center;
                    color: #CFEFD3;
                    font-size: 0.88rem;
                    font-weight: 600;
                "
            >
                Resultado copiado. Agora é só colar no WhatsApp.
            </div>
        </div>
        <script>
            const button = document.getElementById({json.dumps(button_id)});
            const label = document.getElementById({json.dumps(label_id)});
            const status = document.getElementById({json.dumps(status_id)});
            const defaultLabel = "📋 COPIAR PARA WHATSAPP";
            const successLabel = "✅ COPIADO";
            const errorLabel = "⚠️ TENTE NOVAMENTE";
            const defaultBg = "#57CF5F";
            const successBg = "#16A34A";
            const errorBg = "#EF4444";
            let resetTimer = null;

            function setState(state) {{
                if (resetTimer) {{
                    clearTimeout(resetTimer);
                }}

                if (state === "success") {{
                    label.textContent = successLabel;
                    button.style.background = successBg;
                    status.style.display = "block";
                }} else if (state === "error") {{
                    label.textContent = errorLabel;
                    button.style.background = errorBg;
                    status.style.display = "none";
                }} else {{
                    label.textContent = defaultLabel;
                    button.style.background = defaultBg;
                    status.style.display = "none";
                }}

                if (state !== "default") {{
                    resetTimer = setTimeout(() => setState("default"), 1800);
                }}
            }}

            button.addEventListener("mouseenter", () => {{
                if (label.textContent === defaultLabel) {{
                    button.style.background = "#47BF55";
                }}
            }});

            button.addEventListener("mouseleave", () => {{
                if (label.textContent === defaultLabel) {{
                    button.style.background = defaultBg;
                }}
            }});

            button.addEventListener("click", async () => {{
                try {{
                    await navigator.clipboard.writeText({payload});
                    setState("success");
                }} catch (error) {{
                    try {{
                        const textArea = document.createElement("textarea");
                        textArea.value = {payload};
                        textArea.style.position = "fixed";
                        textArea.style.opacity = "0";
                        document.body.appendChild(textArea);
                        textArea.focus();
                        textArea.select();
                        const copied = document.execCommand("copy");
                        document.body.removeChild(textArea);
                        if (copied) {{
                            setState("success");
                        }} else {{
                            setState("error");
                        }}
                    }} catch (fallbackError) {{
                        setState("error");
                    }}
                }}
            }});
        </script>
        """,
        height=86,
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
            or st.session_state.cadastro_guiado_ativo
            or pos_cadastro_pendente
        )

        if st.session_state.lista_revisada_confirmada:
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
                n_m = st.slider("Nota", 1.0, 10.0, 6.0, 0.5, key="guiado_inline_nota")
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
                    "🔐 Confirmar senha",
                    key="grupo_confirmar_senha",
                    on_click=abrir_expander_grupo,
                ):
                    if senha == str(senha_adm):
                        st.session_state.senha_admin_confirmada = True
                        st.session_state.ultima_senha_digitada = senha
                    else:
                        st.session_state.senha_admin_confirmada = False
                        st.session_state.ultima_senha_digitada = senha
                        st.error("Senha incorreta")
                if st.session_state.senha_admin_confirmada:
                    st.success("Senha confirmada. Agora clique em Carregar base de dados.")
                else:
                    st.caption("Depois de informar a senha, toque em **Confirmar senha** ou clique direto em **Carregar base de dados**.")
            else:
                st.write("Já tem uma planilha? Envie o arquivo abaixo e depois clique em **Carregar base de dados**.")
                uploaded_file = st.file_uploader(
                    "Enviar planilha Excel",
                    type=["xlsx"],
                    label_visibility="collapsed",
                    key="grupo_upload_planilha",
                )

        if not admin_base_carregada and st.button(
            "📥 Carregar base de dados",
            key="grupo_carregar_base",
            on_click=abrir_expander_grupo,
        ):
            if origem_base == "Base original (Admin)":
                if not grupo_admin:
                    st.session_state.base_admin_carregada = False
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
                    st.session_state.base_admin_carregada = False
                    st.session_state.senha_admin_confirmada = False
                    st.error("Senha incorreta")
                else:
                    st.session_state.df_base = logic.carregar_dados_originais()
                    st.session_state.novos_jogadores = []
                    st.session_state.is_admin = True
                    st.session_state.base_admin_carregada = True
                    st.session_state.ultimo_arquivo = None
                    st.session_state.qtd_jogadores_adicionados_manualmente = 0
                    st.session_state.senha_admin_confirmada = True
                    st.session_state.grupo_config_expanded = False
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
                        st.session_state.base_admin_carregada = False
                        st.session_state.ultimo_arquivo = uploaded_file.name
                        st.session_state.qtd_jogadores_adicionados_manualmente = 0
                        st.session_state.senha_admin_confirmada = False
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
                    st.session_state.grupo_config_expanded = True
                    st.rerun()

    return nome_pelada


def render_manual_card(logic, nome_pelada: str):
    with st.expander(resumo_expander_cadastro_manual(), expanded=False):
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
                    st.session_state.qtd_jogadores_adicionados_manualmente += 1
                    st.success(f"{novo_nome} salvo!")
                else:
                    st.error("Digite um nome.")

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
    )
    diagnostico_atual = st.session_state.diagnostico_lista or {}

    if st.session_state.cadastro_guiado_ativo:
        st.caption('Próximo passo: conclua o cadastro guiado dos jogadores faltantes e depois revise a lista novamente.')
    elif not lista_revisada_ok:
        st.caption('Próximo passo: clique em "🔎 Revisar lista" para verificar nomes e pendências.')
    elif diagnostico_atual.get("tem_nao_encontrados", False):
        st.caption('Próximo passo: clique em "➕ Cadastrar faltantes agora", conclua o cadastro e depois revise a lista novamente.')
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

                df_jogar = df_final[df_final['Nome'].isin(nomes_corrigidos)].drop_duplicates(subset=['Nome'], keep='last')

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
            "6. Resultado",
            "Veja os times sorteados e copie o resultado para compartilhar."
        )
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

        render_copy_button_with_feedback(texto_copiar)

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
