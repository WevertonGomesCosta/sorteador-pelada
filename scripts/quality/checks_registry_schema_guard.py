#!/usr/bin/env python3
"""Guard leve do schema do registro canônico dos checks.

Valida a integridade estrutural e semântica de ``scripts/quality/checks_registry.py``:
- campos obrigatórios e tipos mínimos por entrada;
- unicidade de ids, nomes, caminhos e comandos canônicos;
- coerência entre categoria, timeout e flags de inclusão;
- existência real dos caminhos apontados no registro;
- presença do contrato canônico na documentação operacional oficial.

Uso:
    python scripts/quality/checks_registry_schema_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality import checks_registry

ALLOWED_CATEGORIES = {
    "structural",
    "validation",
    "release",
    "compatibility",
    "operations",
    "documentation",
    "runtime",
    "hygiene",
    "governance",
}
ALLOWED_KINDS = {"script", "module"}
DOCS_TO_VALIDATE = [
    "README.md",
    "docs/operations/OPERACAO_LOCAL.md",
    "docs/releases/RELEASE_OPERACIONAL.md",
    "docs/releases/BASELINE_OFICIAL.md",
]
DOC_MARKERS = [
    "python scripts/quality/checks_registry_schema_guard.py",
    "scripts/quality/checks_registry.py",
    "Schema canônico",
]


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def normalized_command(spec: dict[str, object]) -> str:
    kind = str(spec["kind"])
    if kind == "module":
        args = " ".join(str(x) for x in spec["args"])
        return f"python {args}"
    return f"python {spec['rel_path']}"


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(__doc__.strip())
        return 0

    errors: list[str] = []
    notes: list[str] = []
    entries = checks_registry.registry_entries()
    ids: list[str] = []
    names: list[str] = []
    commands: list[str] = []
    rel_paths: list[str] = []

    for index, spec in enumerate(entries, start=1):
        label = f"entrada #{index}"
        required_fields = (
            "id",
            "name",
            "category",
            "kind",
            "timeout_seconds",
            "enabled_in_quality_gate",
            "enabled_in_release_health_report",
        )
        missing = [field for field in required_fields if field not in spec]
        if missing:
            errors.append(f"{label} deve conter os campos obrigatórios: {', '.join(missing)}")
            continue

        check_id = spec["id"]
        check_name = spec["name"]
        category = spec["category"]
        kind = spec["kind"]
        timeout_seconds = spec["timeout_seconds"]
        gate_flag = spec["enabled_in_quality_gate"]
        report_flag = spec["enabled_in_release_health_report"]

        if not isinstance(check_id, str) or not check_id.strip():
            errors.append(f"{label} deve ter `id` não vazio")
        else:
            ids.append(check_id)
            notes.append(f"OK id presente: {check_id}")

        if not isinstance(check_name, str) or not check_name.strip():
            errors.append(f"{label} deve ter `name` não vazio")
        else:
            names.append(check_name)
            notes.append(f"OK name presente: {check_name}")

        if isinstance(check_id, str) and isinstance(check_name, str) and check_id != check_name:
            errors.append(f"{label} deve manter `id` e `name` sincronizados")

        if category not in ALLOWED_CATEGORIES:
            errors.append(f"{label} deve usar categoria válida; recebeu `{category}`")
        else:
            notes.append(f"OK categoria válida em {check_name}: {category}")

        if kind not in ALLOWED_KINDS:
            errors.append(f"{label} deve usar kind válido; recebeu `{kind}`")
        else:
            notes.append(f"OK kind válido em {check_name}: {kind}")

        if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
            errors.append(f"{label} deve definir `timeout_seconds` como inteiro positivo")
        else:
            notes.append(f"OK timeout válido em {check_name}: {timeout_seconds}s")
            if category == "validation" and timeout_seconds < checks_registry.DEFAULT_CHECK_TIMEOUT_SECONDS:
                errors.append(f"{label} com categoria `validation` deve ter timeout >= {checks_registry.DEFAULT_CHECK_TIMEOUT_SECONDS}s")

        if not isinstance(gate_flag, bool) or not isinstance(report_flag, bool):
            errors.append(f"{label} deve definir flags booleanas de inclusão")
        elif not gate_flag and not report_flag:
            errors.append(f"{label} não pode ficar fora de quality_gate e release_health_report ao mesmo tempo")
        else:
            notes.append(f"OK flags de inclusão válidas em {check_name}")

        if kind == "script":
            rel_path = spec.get("rel_path")
            if not isinstance(rel_path, str) or not rel_path.strip():
                errors.append(f"{label} do tipo script deve definir `rel_path` não vazio")
            else:
                rel_paths.append(rel_path)
                script_path = ROOT / rel_path
                if not script_path.exists():
                    errors.append(f"{label} aponta para script inexistente: {rel_path}")
                else:
                    notes.append(f"OK script existente no registro: {rel_path}")
        elif kind == "module":
            args = spec.get("args")
            if not isinstance(args, list) or not args or not all(isinstance(x, str) and x.strip() for x in args):
                errors.append(f"{label} do tipo module deve definir `args` como lista não vazia de strings")
            else:
                notes.append(f"OK args válidos em {check_name}: {' '.join(args)}")

        commands.append(normalized_command(spec))

    if len(ids) != len(set(ids)):
        errors.append("scripts/quality/checks_registry.py não deve conter ids duplicados")
    else:
        notes.append("OK sem ids duplicados no checks_registry")
    if len(names) != len(set(names)):
        errors.append("scripts/quality/checks_registry.py não deve conter nomes duplicados")
    else:
        notes.append("OK sem nomes duplicados no checks_registry")
    if len(commands) != len(set(commands)):
        errors.append("scripts/quality/checks_registry.py não deve conter comandos canônicos duplicados")
    else:
        notes.append("OK sem comandos canônicos duplicados no checks_registry")
    if len(rel_paths) != len(set(rel_paths)):
        errors.append("scripts/quality/checks_registry.py não deve conter caminhos canônicos duplicados")
    else:
        notes.append("OK sem caminhos canônicos duplicados no checks_registry")

    for rel_path in DOCS_TO_VALIDATE:
        text = read_text(rel_path)
        for marker in DOC_MARKERS:
            if marker not in text:
                errors.append(f"{rel_path} deve citar `{marker}`")
            else:
                notes.append(f"OK marcador documental presente: {rel_path} -> {marker}")

    print("=== CHECKS REGISTRY SCHEMA GUARD | Sorteador Pelada PRO ===")
    for note in notes:
        print(f"[OK] {note}")

    if errors:
        print("\nErros encontrados:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nSchema canônico do checks_registry íntegro.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
