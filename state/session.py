import pandas as pd
import streamlit as st

import state.keys as K

from core.validators import diagnosticar_nomes_bloqueados_para_sorteio

import state.keys as K
def init_session_state(logic):
    if K.DF_BASE not in st.session_state:
        st.session_state[K.DF_BASE] = logic.criar_base_vazia()

    if K.NOVOS_JOGADORES not in st.session_state:
        st.session_state[K.NOVOS_JOGADORES] = []

    if K.IS_ADMIN not in st.session_state:
        st.session_state[K.IS_ADMIN] = False

    if K.AVISO_SEM_PLANILHA not in st.session_state:
        st.session_state[K.AVISO_SEM_PLANILHA] = False

    if K.DIAGNOSTICO_LISTA not in st.session_state:
        st.session_state[K.DIAGNOSTICO_LISTA] = None

    if K.LISTA_REVISADA not in st.session_state:
        st.session_state[K.LISTA_REVISADA] = None

    if K.LISTA_REVISADA_CONFIRMADA not in st.session_state:
        st.session_state[K.LISTA_REVISADA_CONFIRMADA] = False

    if K.LISTA_TEXTO_REVISADO not in st.session_state:
        st.session_state[K.LISTA_TEXTO_REVISADO] = ""

    if K.REVISAO_LISTA_EXPANDIDA not in st.session_state:
        st.session_state[K.REVISAO_LISTA_EXPANDIDA] = False

    if K.FALTANTES_REVISAO not in st.session_state:
        st.session_state[K.FALTANTES_REVISAO] = []

    if K.CADASTRO_GUIADO_ATIVO not in st.session_state:
        st.session_state[K.CADASTRO_GUIADO_ATIVO] = False

    if K.FALTANTES_CADASTRADOS_NA_RODADA not in st.session_state:
        st.session_state[K.FALTANTES_CADASTRADOS_NA_RODADA] = []

    if K.REVISAO_PENDENTE_POS_CADASTRO not in st.session_state:
        st.session_state[K.REVISAO_PENDENTE_POS_CADASTRO] = False


def registrar_base_carregada_no_estado(logic, df_base: pd.DataFrame, *, is_admin: bool, ultimo_arquivo: str | None):
    st.session_state[K.DF_BASE] = df_base
    st.session_state[K.NOVOS_JOGADORES] = []
    st.session_state[K.IS_ADMIN] = is_admin
    st.session_state[K.BASE_ADMIN_CARREGADA] = is_admin
    st.session_state[K.ULTIMO_ARQUIVO] = ultimo_arquivo
    st.session_state[K.QTD_JOGADORES_ADICIONADOS_MANUALMENTE] = 0
    st.session_state[K.SCROLL_PARA_LISTA] = True
    atualizar_integridade_base_no_estado(logic)



def atualizar_integridade_base_no_estado(logic):
    if hasattr(logic, "diagnosticar_inconsistencias_base"):
        st.session_state[K.BASE_INCONSISTENCIAS_CARREGAMENTO] = logic.diagnosticar_inconsistencias_base(
            st.session_state[K.DF_BASE]
        )
    if hasattr(logic, "listar_registros_inconsistentes"):
        st.session_state[K.BASE_REGISTROS_INCONSISTENTES_CARREGAMENTO] = (
            logic.listar_registros_inconsistentes(st.session_state[K.DF_BASE]).to_dict("records")
        )
    else:
        st.session_state[K.BASE_REGISTROS_INCONSISTENTES_CARREGAMENTO] = []


def limpar_estado_revisao_lista():
    st.session_state[K.DIAGNOSTICO_LISTA] = None
    st.session_state[K.LISTA_REVISADA] = None
    st.session_state[K.LISTA_REVISADA_CONFIRMADA] = False
    st.session_state[K.LISTA_TEXTO_REVISADO] = ""
    st.session_state[K.REVISAO_LISTA_EXPANDIDA] = False


def diagnosticar_lista_no_estado(logic, lista_texto: str):
    processamento = logic.processar_lista(
        lista_texto,
        return_metadata=True,
        emit_warning=False,
    )

    if not processamento["jogadores"]:
        limpar_estado_revisao_lista()
        return None

    base_pronta = bool(
        not st.session_state[K.DF_BASE].empty or st.session_state[K.NOVOS_JOGADORES]
    )

    if not base_pronta:
        nomes_brutos = list(processamento["jogadores"])
        ignorados = list(processamento["ignorados"])

        duplicados = []
        nomes_unicos = []
        vistos = set()
        for nome in nomes_brutos:
            if nome in vistos:
                if nome not in duplicados:
                    duplicados.append(nome)
            else:
                vistos.add(nome)
                nomes_unicos.append(nome)

        diagnostico = {
            "nomes_brutos": nomes_brutos,
            "ignorados": ignorados,
            "nomes_corrigidos": nomes_brutos,
            "correcoes_aplicadas": [],
            "duplicados": duplicados,
            "nao_encontrados": [],
            "lista_final_sugerida": nomes_unicos,
            "total_brutos": len(nomes_brutos),
            "total_validos": len(nomes_unicos),
            "tem_nao_encontrados": False,
            "tem_duplicados": len(duplicados) > 0,
            "tem_correcoes": False,
            "tem_problemas": len(ignorados) > 0 or len(duplicados) > 0,
            "nomes_bloqueados_base": [],
            "tem_bloqueio_base": False,
            "modo_revisao": "aleatorio_lista",
        }
    else:
        diagnostico = logic.diagnosticar_lista_para_sorteio(
            lista_texto,
            st.session_state[K.DF_BASE],
            st.session_state[K.NOVOS_JOGADORES],
        )

        df_final = st.session_state[K.DF_BASE].copy()
        if st.session_state[K.NOVOS_JOGADORES]:
            df_final = pd.concat([df_final, pd.DataFrame(st.session_state[K.NOVOS_JOGADORES])], ignore_index=True)

        nomes_bloqueados_base = diagnosticar_nomes_bloqueados_para_sorteio(
            df_final,
            diagnostico.get("lista_final_sugerida", []),
        )
        diagnostico["nomes_bloqueados_base"] = nomes_bloqueados_base
        diagnostico["tem_bloqueio_base"] = len(nomes_bloqueados_base) > 0
        diagnostico["modo_revisao"] = "balanceado"

    st.session_state[K.DIAGNOSTICO_LISTA] = diagnostico
    st.session_state[K.LISTA_REVISADA] = None
    st.session_state[K.LISTA_REVISADA_CONFIRMADA] = False
    st.session_state[K.LISTA_TEXTO_REVISADO] = lista_texto
    st.session_state[K.REVISAO_LISTA_EXPANDIDA] = True
    return diagnostico