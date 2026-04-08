#!/usr/bin/env python3
"""Checagem estrutural mínima da base do Sorteador Pelada PRO.

Objetivo:
- validar a existência dos módulos centrais após a reorganização;
- validar funções críticas por AST;
- validar a presença das chaves centrais do session_state;
- compilar o projeto para detectar regressões sintáticas.

Uso:
    python scripts/check_base.py
"""

from __future__ import annotations

import ast
import compileall
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "app.py",
    "core/flow_guard.py",
    "core/logic.py",
    "core/optimizer.py",
    "core/validators.py",
    "data/repository.py",
    "state/keys.py",
    "state/session.py",
    "state/ui_state.py",
    "state/view_models.py",
    "ui/actions.py",
    "ui/base_view.py",
    "ui/components.py",
    "ui/group_config_view.py",
    "ui/manual_card.py",
    "ui/panels.py",
    "ui/pre_sort_view.py",
    "ui/primitives.py",
    "ui/result_view.py",
    "ui/review_view.py",
    "ui/styles.py",
    "ui/summary_strings.py",
]

EXPECTED_FUNCTIONS = {
    "app.py": {"main"},
    "core/flow_guard.py": {
        "construir_assinatura_entrada_sorteio",
        "extrair_nomes_unicos_da_lista",
        "sortear_times_aleatorios_por_lista",
        "invalidar_resultado_se_entrada_mudou",
        "contar_duplicados_base_atual",
        "construir_gate_pre_sorteio",
    },
    "state/session.py": {
        "init_session_state",
        "registrar_base_carregada_no_estado",
        "atualizar_integridade_base_no_estado",
        "limpar_estado_revisao_lista",
        "diagnosticar_lista_no_estado",
    },
    "state/ui_state.py": {"ensure_local_session_state", "abrir_expander_cadastro_manual"},
    "state/view_models.py": {
        "determinar_visibilidade_revisao",
        "determinar_etapa_visual_ativa",
        "construir_status_sessao_visual",
        "construir_estado_blocos_visuais",
    },
    "ui/actions.py": {"render_action_button"},
    "ui/base_view.py": {
        "render_base_summary",
        "render_base_inconsistencias_expander",
        "render_base_integrity_alert",
        "render_base_preview",
        "total_inconsistencias_base",
        "resumo_inconsistencias_base",
    },
    "ui/group_config_view.py": {
        "abrir_expander_grupo",
        "grupo_config_deve_abrir",
        "ativar_fluxo_somente_lista",
        "render_group_config_expander",
    },
    "ui/panels.py": {"render_step_cta_panel", "render_session_status_panel"},
    "ui/pre_sort_view.py": {"render_resumo_operacional_pre_sorteio"},
    "ui/primitives.py": {"render_section_header", "render_inline_status_note", "render_app_meta_footer"},
    "ui/result_view.py": {
        "formatar_timestamp_sorteio_para_exibicao",
        "formatar_timestamp_sorteio_para_arquivo",
        "construir_cabecalho_padronizado_sorteio",
        "construir_texto_compartilhamento_resultado",
        "render_acoes_resultado",
        "render_result_summary_panel",
        "render_sort_ready_panel",
        "render_team_cards",
    },
    "ui/review_view.py": {
        "render_revisao_pendencias_panel",
        "render_correcao_inline_bloqueios_base",
        "render_correcao_inline_etapa2",
        "render_revisao_lista",
    },
    "ui/summary_strings.py": {
        "resumo_expander_configuracao",
        "resumo_expander_cadastro_manual",
        "obter_criterios_ativos",
        "resumo_criterios_ativos",
        "resumo_expander_criterios",
    },
}

EXPECTED_KEYS = {
    "DF_BASE",
    "DIAGNOSTICO_LISTA",
    "LISTA_REVISADA",
    "LISTA_REVISADA_CONFIRMADA",
    "LISTA_TEXTO_INPUT",
    "LISTA_TEXTO_INPUT_DRAFT",
    "LISTA_TEXTO_INPUT_PENDING",
    "LISTA_TEXTO_REVISADO",
    "CADASTRO_GUIADO_ATIVO",
    "FALTANTES_REVISAO",
    "CADASTRO_MANUAL_EXPANDED",
    "MANUAL_SECTION_VISIBLE",
    "GRUPO_CONFIG_EXPANDED",
    "GRUPO_ORIGEM_FLUXO",
    "SCROLL_PARA_REVISAO",
    "SCROLL_DESTINO_REVISAO",
    "SCROLL_PARA_RESULTADO",
    "RESULTADO",
    "RESULTADO_CONTEXTO",
    "RESULTADO_ASSINATURA",
}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def function_names(tree: ast.AST) -> set[str]:
    return {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def assigned_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []

    for rel_path in REQUIRED_FILES:
        full_path = ROOT / rel_path
        if not full_path.exists():
            errors.append(f"Arquivo ausente: {rel_path}")

    for rel_path, expected in EXPECTED_FUNCTIONS.items():
        full_path = ROOT / rel_path
        if not full_path.exists():
            continue
        tree = read_ast(full_path)
        found = function_names(tree)
        missing = sorted(expected - found)
        if missing:
            errors.append(
                f"Funções ausentes em {rel_path}: {', '.join(missing)}"
            )
        else:
            notes.append(f"OK funções: {rel_path}")

    keys_path = ROOT / "state/keys.py"
    if keys_path.exists():
        keys_tree = read_ast(keys_path)
        found_keys = assigned_names(keys_tree)
        missing_keys = sorted(EXPECTED_KEYS - found_keys)
        if missing_keys:
            errors.append(
                "Chaves ausentes em state/keys.py: " + ", ".join(missing_keys)
            )
        else:
            notes.append("OK chaves centrais: state/keys.py")

    compiled = compileall.compile_dir(str(ROOT), force=True, quiet=1)
    if not compiled:
        errors.append("Falha na compilação sintática do projeto (compileall).")
    else:
        notes.append("OK compilação: python -m compileall .")

    print("=== CHECK BASE | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")

    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nBase estrutural íntegra.")
    print("Sugestão: também execute o CHECKLIST_REGRESSAO.md após mudanças funcionais.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
