# BASELINE_OFICIAL

> Nesta baseline, os caminhos canônicos reorganizados são o padrão oficial de uso. Wrappers e arquivos-ponte históricos continuam disponíveis apenas como compatibilidade temporária.

## Versão oficial vigente

A baseline oficial vigente desta base é **v104**.

A **v96** segue registrada como marco de manutenção sob demanda da frente estrutural em `docs/releases/MAINTENANCE_MODE.md`.

A **v97** inaugurou a iteração funcional controlada de UX na revisão/preparo da lista, a **v98** consolidou a segunda micro-melhoria funcional dessa frente, a **v99** consolidou a terceira micro-melhoria funcional com foco no resumo pré-sorteio, a **v100** fechou a frente de pós-resultado com histórico recente da sessão e reabertura fiel do resultado, e a **v101** consolida uma simplificação funcional da revisão de inconsistências, preservando o congelamento estrutural e a lógica central do app, enquanto a **v102** corrige o fluxo de múltiplos faltantes no cadastro guiado e antecipa o alerta de duplicados da base logo após o carregamento, a **v103** preserva qualificadores distintivos na detecção de duplicados da lista para evitar falso positivo em casos como `Douglas` vs `Douglas (pimpim)` e `Joel` vs `Joel (convidado)`, e a **v104** prioriza o scroll de revisão para o bloco de cadastro guiado enquanto `CADASTRO_GUIADO_ATIVO` estiver verdadeiro.

## Princípios de preservação

Esta baseline deve ser tratada como a referência única, fixa e estável do projeto para manutenção incremental.

Não reabrir sem necessidade concreta:
- arquitetura ampla
- lógica do sorteio
- regras da revisão
- fluxo estabilizado de scroll
- confirmação/sorteio
- áreas sensíveis já congeladas

## Estado consolidado da base

A base atual preserva:
- arquitetura modular por domínio
- governança documental e operacional
- gates estruturais e de release
- desacoplamento entre `core/flow_guard.py` e a camada de UI
- camada leve ampliada de smoke test funcional para módulos neutros e auxiliares seguros
- runner único de quality gate técnico e pré-checagem de runtime local
- gerador padronizado de registro para validação manual no navegador
- reorganização operacional leve da documentação, scripts e testes com compatibilidade preservada

## Área sensível congelada

`ui/review_view.py` permanece como área sensível.

Mudanças nesse módulo só devem ocorrer se houver necessidade concreta, localizada e justificada por regressão ou bloqueio real de uso.

## Validação mínima obrigatória

Antes de fechar uma nova iteração ou release oficial, executar:

```bash
python scripts/quality/runtime_preflight.py
python scripts/quality/check_base.py
python scripts/validation/smoke_test_base.py
python scripts/quality/release_metadata_guard.py
python scripts/quality/compatibility_contract_guard.py
python scripts/quality/operational_checks_contract_guard.py
python scripts/quality/canonical_paths_reference_guard.py
python scripts/quality/script_cli_contract_guard.py
python scripts/quality/release_artifacts_hygiene_guard.py
python scripts/quality/runtime_dependencies_contract_guard.py
python scripts/quality/documentation_commands_examples_guard.py
python scripts/quality/release_manifest_guard.py
python scripts/quality/quality_runtime_budget_guard.py
python scripts/quality/protected_scope_hash_guard.py
python scripts/quality/quality_gate_composition_guard.py
python scripts/quality/release_guard.py
python scripts/quality/quality_gate.py
python scripts/reports/manual_validation_pack.py
python scripts/reports/release_health_report.py
python scripts/reports/maintenance_snapshot_report.py
python scripts/reports/maintenance_handoff_pack.py
python scripts/reports/maintenance_resume_brief.py
python scripts/reports/maintenance_command_journal.py
python scripts/reports/maintenance_reports_cleanup.py
python scripts/reports/maintenance_refresh_bundle.py
python scripts/reports/maintenance_reports_index.py
```

Wrappers e arquivos-ponte históricos continuam válidos apenas como compatibilidade temporária.

A política oficial dessa compatibilidade está formalizada em `docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md`.

O manifesto oficial do escopo protegido está registrado em `docs/releases/PROTECTED_SCOPE_HASHES.json`.

Para triagem operacional, handoff técnico somente leitura, regeneração canônica dos artefatos de manutenção e higiene segura de artefatos transitórios, esta baseline também passa a fornecer:
- `scripts/reports/maintenance_snapshot_report.py`
- `scripts/reports/maintenance_handoff_pack.py`
- `scripts/reports/maintenance_resume_brief.py`
- `scripts/reports/maintenance_command_journal.py`
- `scripts/reports/maintenance_reports_cleanup.py`
- `scripts/reports/maintenance_refresh_bundle.py`

## Crosslinks canônicos de governança

Este documento deve permanecer navegável em conjunto com:
- `docs/releases/RELEASE_OPERACIONAL.md`
- `docs/releases/MAINTENANCE_MODE.md`
- `docs/operations/OPERACAO_LOCAL.md`
- `docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md`
- `docs/validation/VALIDACAO_MANUAL_GUIA.md`

Guard leve desta coerência documental:

```bash
python scripts/quality/governance_docs_crosslinks_guard.py
```


## Registro canônico dos checks

A rotina oficial de checks passa a ter uma fonte única de verdade em `scripts/quality/checks_registry.py`

Schema canônico do registro: `scripts/quality/checks_registry_schema_guard.py`

Consumo exclusivo do checks_registry canônico: `scripts/quality/checks_registry_consumers_guard.py`

Composição determinística do quality_gate: `scripts/quality/quality_gate_composition_guard.py`.

Validação do contrato do registro:

python scripts/quality/checks_registry_contract_guard.py
python scripts/quality/checks_registry_schema_guard.py
python scripts/quality/checks_registry_consumers_guard.py
python scripts/quality/quality_gate_composition_guard.py
