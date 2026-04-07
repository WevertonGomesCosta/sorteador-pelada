import pandas as pd
import streamlit as st

from core.validators import diagnosticar_nomes_bloqueados_para_sorteio



def init_session_state(logic):
    if 'df_base' not in st.session_state:
        st.session_state.df_base = logic.criar_base_vazia()

    if 'novos_jogadores' not in st.session_state:
        st.session_state.novos_jogadores = []

    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False

    if 'aviso_sem_planilha' not in st.session_state:
        st.session_state.aviso_sem_planilha = False

    if 'diagnostico_lista' not in st.session_state:
        st.session_state.diagnostico_lista = None

    if 'lista_revisada' not in st.session_state:
        st.session_state.lista_revisada = None

    if 'lista_revisada_confirmada' not in st.session_state:
        st.session_state.lista_revisada_confirmada = False

    if 'lista_texto_revisado' not in st.session_state:
        st.session_state.lista_texto_revisado = ""

    if 'revisao_lista_expandida' not in st.session_state:
        st.session_state.revisao_lista_expandida = False

    if "faltantes_revisao" not in st.session_state:
        st.session_state.faltantes_revisao = []

    if "cadastro_guiado_ativo" not in st.session_state:
        st.session_state.cadastro_guiado_ativo = False

    if "faltantes_cadastrados_na_rodada" not in st.session_state:
        st.session_state.faltantes_cadastrados_na_rodada = []

    if "revisao_pendente_pos_cadastro" not in st.session_state:
        st.session_state.revisao_pendente_pos_cadastro = False


def registrar_base_carregada_no_estado(logic, df_base: pd.DataFrame, *, is_admin: bool, ultimo_arquivo: str | None):
    st.session_state.df_base = df_base
    st.session_state.novos_jogadores = []
    st.session_state.is_admin = is_admin
    st.session_state.base_admin_carregada = is_admin
    st.session_state.ultimo_arquivo = ultimo_arquivo
    st.session_state.qtd_jogadores_adicionados_manualmente = 0
    atualizar_integridade_base_no_estado(logic)



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

    base_pronta = bool(
        not st.session_state.df_base.empty or st.session_state.novos_jogadores
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
        diagnostico["modo_revisao"] = "balanceado"

    st.session_state.diagnostico_lista = diagnostico
    st.session_state.lista_revisada = None
    st.session_state.lista_revisada_confirmada = False
    st.session_state.lista_texto_revisado = lista_texto
    st.session_state.revisao_lista_expandida = True
    return diagnostico
