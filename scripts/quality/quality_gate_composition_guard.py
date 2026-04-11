#!/usr/bin/env python3
"""Guard leve da composição determinística do quality gate.

Valida, sem tocar no núcleo funcional do app, que ``scripts/quality/quality_gate.py``:
- compõe ``CHECKS`` diretamente a partir de ``scripts/quality/checks_registry.py``;
- preserva ordem determinística, cobertura completa e ausência de duplicidades;
- reutiliza o contrato canônico de timeout do registro;
- mantém a documentação operacional oficial sincronizada com esse contrato.

Uso:
    python scripts/quality/quality_gate_composition_guard.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality import quality_gate
from scripts.quality.checks_registry import expected_official_checks

QUALITY_GATE_PATH = ROOT / "scripts" / "quality" / "quality_gate.py"
DOCS_TO_VALIDATE = [
    "README.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
    "docs/releases/BASELINE_OFICIAL.md",
    "docs/validation/VALIDACAO_MANUAL_GUIA.md",
]
DOC_MARKERS = [
    "python scripts/quality/quality_gate_composition_guard.py",
    "scripts/quality/checks_registry.py",
    "composição determinística do quality_gate",
]
MUTATING_METHODS = {"append", "extend", "insert", "sort", "reverse", "clear", "pop", "remove"}


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def normalize_command(command: list[str]) -> str:
    normalized: list[str] = []
    for index, token in enumerate(command):
        if index == 0:
            normalized.append("python")
            continue
        token_path = Path(token)
        if token_path.is_absolute():
            try:
                normalized.append(token_path.relative_to(ROOT).as_posix())
                continue
            except ValueError:
                pass
        normalized.append(token)
    return " ".join(normalized)


def normalize_checks(checks: list[tuple[str, list[str]]]) -> list[tuple[str, str]]:
    return [(name, normalize_command(command)) for name, command in checks]


def is_quality_gate_checks_assignment(node: ast.AST) -> bool:
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name) or node.targets[0].id != "CHECKS":
            return False
        call = node.value
    elif isinstance(node, ast.AnnAssign):
        if not isinstance(node.target, ast.Name) or node.target.id != "CHECKS":
            return False
        call = node.value
    else:
        return False
    if not isinstance(call, ast.Call):
        return False
    if not isinstance(call.func, ast.Name) or call.func.id != "build_check_commands":
        return False
    if len(call.args) != 2:
        return False
    if not isinstance(call.args[0], ast.Name) or call.args[0].id != "ROOT":
        return False
    second = call.args[1]
    if not isinstance(second, ast.Attribute) or second.attr != "executable":
        return False
    if not isinstance(second.value, ast.Name) or second.value.id != "sys":
        return False
    keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
    target = keywords.get("target")
    return isinstance(target, ast.Constant) and target.value == "quality_gate"


def extract_main_function(module: ast.Module) -> ast.FunctionDef | None:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    return None


def has_main_iteration_over_checks(function: ast.FunctionDef) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.For):
            continue
        if not isinstance(node.iter, ast.Name) or node.iter.id != "CHECKS":
            continue
        target = node.target
        if isinstance(target, ast.Tuple) and len(target.elts) == 2:
            names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
            if names == ["name", "command"]:
                return True
    return False


def has_mutation_of_checks(module: ast.Module) -> bool:
    for node in ast.walk(module):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "CHECKS":
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Name) and owner.id == "CHECKS" and node.func.attr in MUTATING_METHODS:
                return True
    return False


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(__doc__.strip())
        return 0

    errors: list[str] = []
    notes: list[str] = []

    source = QUALITY_GATE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)

    required_import_markers = [
        "CHECK_TIMEOUT_OVERRIDES",
        "DEFAULT_CHECK_TIMEOUT_SECONDS",
        "build_check_commands",
    ]
    for marker in required_import_markers:
        if marker not in source:
            errors.append(f"scripts/quality/quality_gate.py deve importar `{marker}` a partir do checks_registry canônico")
        else:
            notes.append(f"OK quality_gate referencia {marker} a partir do checks_registry")

    if "CHECK_TIMEOUT_OVERRIDES.get(name, DEFAULT_CHECK_TIMEOUT_SECONDS)" not in source:
        errors.append("scripts/quality/quality_gate.py deve usar o contrato canônico de timeout do checks_registry")
    else:
        notes.append("OK quality_gate reutiliza o contrato canônico de timeout")

    checks_assignments = [node for node in module.body if is_quality_gate_checks_assignment(node)]
    if len(checks_assignments) != 1:
        errors.append(
            "scripts/quality/quality_gate.py deve definir CHECKS exatamente uma vez via build_check_commands(ROOT, sys.executable, target=\"quality_gate\")"
        )
    else:
        notes.append("OK CHECKS é composto diretamente via build_check_commands(..., target=\"quality_gate\")")

    if has_mutation_of_checks(module):
        errors.append("scripts/quality/quality_gate.py não deve mutar CHECKS após a composição canônica")
    else:
        notes.append("OK quality_gate não muta CHECKS após a composição canônica")

    main_function = extract_main_function(module)
    if main_function is None:
        errors.append("scripts/quality/quality_gate.py deve expor main()")
    elif not has_main_iteration_over_checks(main_function):
        errors.append("scripts/quality/quality_gate.py deve iterar diretamente sobre CHECKS em main()")
    else:
        notes.append("OK main() itera diretamente sobre CHECKS")

    expected = expected_official_checks(target="quality_gate")
    actual = normalize_checks(quality_gate.CHECKS)
    if actual != expected:
        errors.append("scripts/quality/quality_gate.py deve preservar ordem e cobertura exatas do checks_registry canônico")
    else:
        notes.append("OK quality_gate preserva ordem determinística e cobertura completa do checks_registry")

    actual_names = [name for name, _ in actual]
    actual_commands = [command for _, command in actual]
    if len(actual_names) != len(set(actual_names)):
        errors.append("scripts/quality/quality_gate.py não deve compor checks com nomes duplicados")
    else:
        notes.append("OK quality_gate sem nomes duplicados")
    if len(actual_commands) != len(set(actual_commands)):
        errors.append("scripts/quality/quality_gate.py não deve compor checks com comandos duplicados")
    else:
        notes.append("OK quality_gate sem comandos duplicados")

    for rel_path in DOCS_TO_VALIDATE:
        text = read_text(rel_path)
        lowered = text.lower()
        for marker in DOC_MARKERS:
            haystack = lowered if marker == "composição determinística do quality_gate" else text
            needle = marker.lower() if marker == "composição determinística do quality_gate" else marker
            if needle not in haystack:
                errors.append(f"{rel_path} deve citar `{marker}`")
            else:
                notes.append(f"OK marcador documental presente: {rel_path} -> {marker}")

    print("=== QUALITY GATE COMPOSITION GUARD | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")

    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nComposição determinística do quality_gate íntegra.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
