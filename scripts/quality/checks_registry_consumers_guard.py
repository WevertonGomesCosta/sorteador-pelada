#!/usr/bin/env python3
"""Guard leve do consumo exclusivo do checks_registry canônico.

Valida que os consumidores oficiais da rotina operacional continuem usando
``scripts/quality/checks_registry.py`` como fonte única de verdade dos checks,
sem reintroduzir listas locais paralelas em ``quality_gate.py``,
``release_health_report.py`` e consumidores auxiliares.

Uso:
    python scripts/quality/checks_registry_consumers_guard.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality import checks_registry, quality_gate, operational_checks_contract_guard
from scripts.reports import release_health_report

DOCS_TO_VALIDATE = [
    "README.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
    "docs/releases/BASELINE_OFICIAL.md",
    "docs/validation/VALIDACAO_MANUAL_GUIA.md",
]
DOC_MARKERS = [
    "python scripts/quality/checks_registry_consumers_guard.py",
    "consumo exclusivo do checks_registry canônico",
    "scripts/quality/checks_registry.py",
]
TARGET_FILES = {
    "scripts/quality/quality_gate.py": {
        "assignment": "CHECKS",
        "call_name": "build_check_commands",
    },
    "scripts/reports/release_health_report.py": {
        "assignment": "CHECKS",
        "call_name": "build_check_commands",
    },
    "scripts/quality/operational_checks_contract_guard.py": {
        "assignment": "EXPECTED_OFFICIAL_CHECKS",
        "call_name": "checks_registry.expected_official_checks",
    },
}
LIST_LITERAL_FORBIDDEN_NAMES = {"CHECKS", "EXPECTED_OFFICIAL_CHECKS", "OFFICIAL_CHECKS", "REGISTRY_CHECKS"}


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def parse_module(rel_path: str) -> ast.Module:
    return ast.parse(read_text(rel_path), filename=rel_path)


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        if prefix:
            return f"{prefix}.{node.attr}"
    return None


def validate_assignment_contract(rel_path: str, assignment_name: str, call_name: str, notes: list[str], errors: list[str]) -> None:
    module = parse_module(rel_path)
    matching_nodes: list[ast.Assign | ast.AnnAssign] = []
    forbidden_literals: list[int] = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if assignment_name in targets:
                matching_nodes.append(node)
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    forbidden_literals.append(node.lineno)
            elif any(name in LIST_LITERAL_FORBIDDEN_NAMES for name in targets) and isinstance(node.value, (ast.List, ast.Tuple)):
                forbidden_literals.append(node.lineno)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == assignment_name:
                matching_nodes.append(node)
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    forbidden_literals.append(node.lineno)
            elif node.target.id in LIST_LITERAL_FORBIDDEN_NAMES and isinstance(node.value, (ast.List, ast.Tuple)):
                forbidden_literals.append(node.lineno)

    if len(matching_nodes) != 1:
        errors.append(f"{rel_path} deve definir exatamente uma atribuição para `{assignment_name}`")
        return

    node = matching_nodes[0]
    value = node.value
    if not isinstance(value, ast.Call):
        errors.append(f"{rel_path} deve definir `{assignment_name}` por chamada ao registro canônico")
        return

    called = dotted_name(value.func)
    if called != call_name:
        errors.append(f"{rel_path} deve definir `{assignment_name}` via `{call_name}`; recebeu `{called}`")
    else:
        notes.append(f"OK {rel_path} consome o checks_registry via `{call_name}`")

    if forbidden_literals:
        lines = ", ".join(str(line) for line in sorted(set(forbidden_literals)))
        errors.append(f"{rel_path} não deve manter listas literais paralelas de checks oficiais (linhas {lines})")
    else:
        notes.append(f"OK {rel_path} sem listas literais paralelas de checks oficiais")


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(__doc__.strip())
        return 0

    errors: list[str] = []
    notes: list[str] = []

    expected_quality = checks_registry.expected_official_checks(target="quality_gate")
    expected_report = checks_registry.expected_official_checks(target="release_health_report")

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

    quality_checks = [(name, normalize_command(command)) for name, command in quality_gate.CHECKS]
    report_checks = [(name, normalize_command(command)) for name, command in release_health_report.CHECKS]

    if quality_checks != expected_quality:
        errors.append("scripts/quality/quality_gate.py deve consumir exclusivamente o checks_registry canônico")
    else:
        notes.append("OK quality_gate reflete exclusivamente o checks_registry canônico")

    if report_checks != expected_report:
        errors.append("scripts/reports/release_health_report.py deve consumir exclusivamente o checks_registry canônico")
    else:
        notes.append("OK release_health_report reflete exclusivamente o checks_registry canônico")

    if operational_checks_contract_guard.EXPECTED_OFFICIAL_CHECKS != expected_quality:
        errors.append("scripts/quality/operational_checks_contract_guard.py deve derivar EXPECTED_OFFICIAL_CHECKS diretamente do checks_registry")
    else:
        notes.append("OK operational_checks_contract_guard deriva EXPECTED_OFFICIAL_CHECKS do checks_registry")

    for rel_path, config in TARGET_FILES.items():
        validate_assignment_contract(rel_path, config["assignment"], config["call_name"], notes, errors)
        text = read_text(rel_path)
        if "scripts/quality/checks_registry.py" not in text and "from scripts.quality import checks_registry" not in text and "from scripts.quality.checks_registry import" not in text:
            errors.append(f"{rel_path} deve importar ou referenciar explicitamente scripts/quality/checks_registry.py")
        else:
            notes.append(f"OK {rel_path} referencia explicitamente o checks_registry")

    for rel_path in DOCS_TO_VALIDATE:
        text = read_text(rel_path)
        text_lower = text.lower()
        for marker in DOC_MARKERS:
            haystack = text_lower if marker == "consumo exclusivo do checks_registry canônico" else text
            needle = marker.lower() if marker == "consumo exclusivo do checks_registry canônico" else marker
            if needle not in haystack:
                errors.append(f"{rel_path} deve citar `{marker}`")
            else:
                notes.append(f"OK marcador documental presente: {rel_path} -> {marker}")

    print("=== CHECKS REGISTRY CONSUMERS GUARD | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")

    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nConsumidores oficiais do checks_registry íntegros.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
