#!/usr/bin/env python3
"""Checagem estrutural da base do Sorteador Pelada PRO.

Objetivo:
- validar a existência dos módulos centrais e artefatos de governança;
- validar funções críticas por AST e sua localização correta;
- validar chaves centrais do session_state;
- validar contratos simples da arquitetura documentada;
- compilar o projeto para detectar regressões sintáticas.

Uso:
    python scripts/quality/check_base.py
"""

from __future__ import annotations

import ast
import compileall
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = [
    "app.py",
    "README.md",
    "CHANGELOG.md",
    "CHECKLIST_REGRESSAO.md",
    "docs/README.md",
    "docs/ARQUITETURA_BASE.md",
    "docs/MANUTENCAO_OPERACIONAL.md",
    "docs/RELEASE_OPERACIONAL.md",
    "docs/BASELINE_OFICIAL.md",
    "docs/PLANO_SMOKE_TEST_MINIMO.md",
    "docs/VALIDACAO_MANUAL_GUIA.md",
    "docs/OPERACAO_LOCAL.md",
    "docs/POLITICA_COMPATIBILIDADE_TEMPORARIA.md",
    "docs/VALIDACAO_UX_MOBILE_2026-04-09.md",
    "docs/architecture/ARQUITETURA_BASE.md",
    "docs/operations/MANUTENCAO_OPERACIONAL.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md",
    "docs/releases/BASELINE_OFICIAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
    "docs/validation/PLANO_SMOKE_TEST_MINIMO.md",
    "docs/validation/VALIDACAO_MANUAL_GUIA.md",
    "docs/validation/VALIDACAO_UX_MOBILE_2026-04-09.md",
    "core/base_summary.py",
    "core/flow_guard.py",
    "core/logic.py",
    "core/optimizer.py",
    "core/validators.py",
    "data/repository.py",
    "state/keys.py",
    "state/session.py",
    "state/ui_state.py",
    "state/view_models.py",
    "state/criteria_state.py",
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
    "scripts/__init__.py",
    "scripts/check_base.py",
    "scripts/release_guard.py",
    "scripts/smoke_test_base.py",
    "scripts/quality_gate.py",
    "scripts/runtime_preflight.py",
    "scripts/release_metadata_guard.py",
    "scripts/compatibility_contract_guard.py",
    "scripts/operational_checks_contract_guard.py",
    "scripts/canonical_paths_reference_guard.py",
    "scripts/script_cli_contract_guard.py",
    "scripts/release_artifacts_hygiene_guard.py",
    "scripts/runtime_dependencies_contract_guard.py",
    "scripts/documentation_commands_examples_guard.py",
    "scripts/manual_validation_pack.py",
    "scripts/release_health_report.py",
    "scripts/quality/__init__.py",
    "scripts/quality/check_base.py",
    "scripts/quality/release_guard.py",
    "scripts/quality/quality_gate.py",
    "scripts/quality/runtime_preflight.py",
    "scripts/quality/release_metadata_guard.py",
    "scripts/quality/compatibility_contract_guard.py",
    "scripts/quality/operational_checks_contract_guard.py",
    "scripts/quality/canonical_paths_reference_guard.py",
    "scripts/quality/script_cli_contract_guard.py",
    "scripts/quality/release_artifacts_hygiene_guard.py",
    "scripts/quality/runtime_dependencies_contract_guard.py",
    "scripts/quality/documentation_commands_examples_guard.py",
    "scripts/validation/__init__.py",
    "scripts/validation/smoke_test_base.py",
    "scripts/reports/__init__.py",
    "scripts/reports/manual_validation_pack.py",
    "scripts/reports/release_health_report.py",
    "tests/__init__.py",
    "tests/_smoke_shared.py",
    "tests/test_smoke_base.py",
    "tests/test_core_smoke.py",
    "tests/test_state_smoke.py",
    "tests/test_ui_safe_smoke.py",
    "tests/test_scripts_smoke.py",
    "reports/.gitkeep",
]

FORBIDDEN_FILES = [
    "ui/sections.py",
]

EXPECTED_TOP_LEVEL_FUNCTIONS = {
    "app.py": {"main"},
    "core/base_summary.py": {
        "total_inconsistencias_base",
        "resumo_inconsistencias_base",
    },
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
        "resumo_expander_criterios",
    },
}

CRITICAL_FUNCTION_OWNERSHIP = {
    "render_base_summary": "ui/base_view.py",
    "render_base_inconsistencias_expander": "ui/base_view.py",
    "render_base_integrity_alert": "ui/base_view.py",
    "render_base_preview": "ui/base_view.py",
    "total_inconsistencias_base": "core/base_summary.py",
    "resumo_inconsistencias_base": "core/base_summary.py",
    "render_group_config_expander": "ui/group_config_view.py",
    "render_revisao_lista": "ui/review_view.py",
    "render_revisao_pendencias_panel": "ui/review_view.py",
    "render_correcao_inline_bloqueios_base": "ui/review_view.py",
    "render_correcao_inline_etapa2": "ui/review_view.py",
    "construir_status_sessao_visual": "state/view_models.py",
    "obter_criterios_ativos": "state/criteria_state.py",
    "resumo_criterios_ativos": "state/criteria_state.py",
    "construir_estado_blocos_visuais": "state/view_models.py",
    "determinar_etapa_visual_ativa": "state/view_models.py",
    "determinar_visibilidade_revisao": "state/view_models.py",
    "render_resumo_operacional_pre_sorteio": "ui/pre_sort_view.py",
    "construir_assinatura_entrada_sorteio": "core/flow_guard.py",
    "construir_gate_pre_sorteio": "core/flow_guard.py",
    "invalidar_resultado_se_entrada_mudou": "core/flow_guard.py",
    "formatar_timestamp_sorteio_para_exibicao": "ui/result_view.py",
    "formatar_timestamp_sorteio_para_arquivo": "ui/result_view.py",
    "construir_cabecalho_padronizado_sorteio": "ui/result_view.py",
    "construir_texto_compartilhamento_resultado": "ui/result_view.py",
    "render_acoes_resultado": "ui/result_view.py",
}

ARCHITECTURE_IMPORT_CONTRACTS = {
    "app.py": {
        "required": {
            "core.flow_guard",
            "state.keys",
            "state.ui_state",
            "state.view_models",
            "state.criteria_state",
            "ui.base_view",
            "ui.group_config_view",
            "ui.pre_sort_view",
            "ui.result_view",
            "ui.review_view",
        },
        "forbidden": {"ui.sections"},
    },
    "state/view_models.py": {
        "forbidden": {"streamlit", "core.logic", "ui.base_view", "ui.review_view", "ui.result_view"},
    },
    "ui/base_view.py": {
        "required": {"core.base_summary"},
        "forbidden": {"app", "ui.sections"},
    },
    "core/flow_guard.py": {
        "required": {"core.base_summary", "state.criteria_state"},
        "forbidden": {"ui.base_view", "ui.summary_strings", "ui.sections"},
    },
    "ui/summary_strings.py": {
        "required": {"state.criteria_state"},
        "forbidden": {"app", "ui.sections"},
    },
    "ui/review_view.py": {
        "forbidden": {"app", "ui.sections"},
    },
    "ui/group_config_view.py": {
        "forbidden": {"app", "ui.sections"},
    },
    "ui/result_view.py": {
        "forbidden": {"app", "ui.sections"},
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


class ImportCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.modules: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.modules.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.modules.add(node.module)


def read_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def top_level_function_names(tree: ast.Module) -> set[str]:
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def all_function_names(tree: ast.AST) -> set[str]:
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


def imported_modules(tree: ast.Module) -> set[str]:
    collector = ImportCollector()
    collector.visit(tree)
    return collector.modules


def all_python_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.py")
        if ".venv" not in path.parts and "__pycache__" not in path.parts
    )


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []

    py_files = all_python_files(ROOT)
    ast_by_file: dict[str, ast.Module] = {}
    funcs_by_file: dict[str, set[str]] = {}

    for rel_path in REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            errors.append(f"Arquivo ausente: {rel_path}")
        else:
            notes.append(f"OK arquivo obrigatório: {rel_path}")

    for rel_path in FORBIDDEN_FILES:
        if (ROOT / rel_path).exists():
            errors.append(f"Arquivo legado não deve existir mais: {rel_path}")
        else:
            notes.append(f"OK arquivo legado ausente: {rel_path}")

    for path in py_files:
        rel = relative(path)
        tree = read_ast(path)
        ast_by_file[rel] = tree
        funcs_by_file[rel] = all_function_names(tree)

    for rel_path, expected in EXPECTED_TOP_LEVEL_FUNCTIONS.items():
        if rel_path not in ast_by_file:
            continue
        found = top_level_function_names(ast_by_file[rel_path])
        missing = sorted(expected - found)
        if missing:
            errors.append(f"Funções de topo ausentes em {rel_path}: {', '.join(missing)}")
        else:
            notes.append(f"OK funções de topo: {rel_path}")

    app_found = top_level_function_names(ast_by_file.get("app.py", ast.parse("")))
    app_extra = sorted(app_found - {"main"})
    if app_extra:
        errors.append("app.py deve permanecer como orquestrador com apenas 'main' no topo. Extras encontrados: " + ", ".join(app_extra))
    else:
        notes.append("OK app.py enxuto: apenas main() no topo")

    for function_name, owner in CRITICAL_FUNCTION_OWNERSHIP.items():
        found_in = sorted(rel for rel, funcs in funcs_by_file.items() if function_name in funcs)
        if found_in != [owner]:
            errors.append(
                f"Função crítica '{function_name}' deve existir apenas em {owner}. Encontrado em: {', '.join(found_in) if found_in else 'nenhum arquivo'}"
            )
        else:
            notes.append(f"OK ownership: {function_name} -> {owner}")

    keys_path = ROOT / "state/keys.py"
    if keys_path.exists():
        keys_tree = ast_by_file.get("state/keys.py") or read_ast(keys_path)
        found_keys = assigned_names(keys_tree)
        missing_keys = sorted(EXPECTED_KEYS - found_keys)
        if missing_keys:
            errors.append("Chaves ausentes em state/keys.py: " + ", ".join(missing_keys))
        else:
            notes.append("OK chaves centrais: state/keys.py")

    for rel_path, contract in ARCHITECTURE_IMPORT_CONTRACTS.items():
        if rel_path not in ast_by_file:
            continue
        mods = imported_modules(ast_by_file[rel_path])
        missing = sorted(contract.get("required", set()) - mods)
        forbidden = sorted(mod for mod in mods if mod in contract.get("forbidden", set()))
        if missing:
            errors.append(f"Imports obrigatórios ausentes em {rel_path}: {', '.join(missing)}")
        else:
            if contract.get("required"):
                notes.append(f"OK imports obrigatórios: {rel_path}")
        if forbidden:
            errors.append(f"Imports proibidos em {rel_path}: {', '.join(forbidden)}")
        else:
            if contract.get("forbidden"):
                notes.append(f"OK sem imports proibidos: {rel_path}")

    doc_path = ROOT / "docs/architecture/ARQUITETURA_BASE.md"
    if doc_path.exists():
        doc_text = doc_path.read_text(encoding="utf-8")
        required_doc_terms = [
            "app.py",
            "core/base_summary.py",
            "core/flow_guard.py",
            "state/criteria_state.py",
            "state/view_models.py",
    "state/criteria_state.py",
            "ui/review_view.py",
            "ui/result_view.py",
            "session_state",
        ]
        missing_terms = [term for term in required_doc_terms if term not in doc_text]
        if missing_terms:
            errors.append("Termos esperados ausentes em docs/architecture/ARQUITETURA_BASE.md: " + ", ".join(missing_terms))
        else:
            notes.append("OK documento de arquitetura com módulos centrais esperados")

    changelog_path = ROOT / "CHANGELOG.md"
    if changelog_path.exists():
        changelog_text = changelog_path.read_text(encoding="utf-8")
        required_changelog_terms = [
            "## Padrão oficial para novas entradas",
            "## Histórico técnico consolidado",
            "## Observações de congelamento vigentes",
            "confirmação/sorteio",
        ]
        missing_terms = [term for term in required_changelog_terms if term not in changelog_text]
        if missing_terms:
            errors.append("Termos esperados ausentes em CHANGELOG.md: " + ", ".join(missing_terms))
        else:
            notes.append("OK changelog com padrão oficial e congelamentos vigentes")

    maintenance_doc_path = ROOT / "docs/operations/MANUTENCAO_OPERACIONAL.md"
    if maintenance_doc_path.exists():
        maintenance_text = maintenance_doc_path.read_text(encoding="utf-8")
        required_maintenance_terms = [
            "Regra de ouro da base",
            "Rituais obrigatórios antes de editar",
            "Rituais obrigatórios depois de editar",
            "Módulos oficiais por tipo de problema",
            "Tipos de mudança permitidos",
            "scripts/quality/check_base.py",
            "CHECKLIST_REGRESSAO.md",
            "state/keys.py",
        ]
        missing_terms = [term for term in required_maintenance_terms if term not in maintenance_text]
        if missing_terms:
            errors.append("Termos esperados ausentes em docs/operations/MANUTENCAO_OPERACIONAL.md: " + ", ".join(missing_terms))
        else:
            notes.append("OK documento de manutenção operacional com protocolo mínimo esperado")

    release_doc_path = ROOT / "docs/releases/RELEASE_OPERACIONAL.md"
    if release_doc_path.exists():
        release_text = release_doc_path.read_text(encoding="utf-8")
        required_release_terms = [
            "Regra de ouro",
            "Ritual obrigatório antes de editar",
            "Ritual obrigatório depois de editar",
            "Fechamento oficial da release",
            "CHANGELOG.md",
            "scripts/quality/check_base.py",
            "scripts/quality/release_guard.py",
        ]
        missing_terms = [term for term in required_release_terms if term not in release_text]
        if missing_terms:
            errors.append("Termos esperados ausentes em docs/releases/RELEASE_OPERACIONAL.md: " + ", ".join(missing_terms))
        else:
            notes.append("OK documento de release operacional com protocolo mínimo esperado")

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
