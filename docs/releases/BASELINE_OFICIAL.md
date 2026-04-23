# BASELINE_OFICIAL

> Nesta baseline, os caminhos canônicos reorganizados são o padrão oficial de uso. Wrappers e arquivos-ponte históricos continuam disponíveis apenas como compatibilidade temporária.

## Versão oficial vigente

A baseline oficial vigente desta base é **v126**.

A **v96** segue registrada como marco de manutenção sob demanda da frente estrutural em `docs/releases/MAINTENANCE_MODE.md`.

A **v97** inaugurou a iteração funcional controlada de UX na revisão/preparo da lista, a **v98** consolidou a segunda micro-melhoria funcional dessa frente, a **v99** consolidou a terceira micro-melhoria funcional com foco no resumo pré-sorteio, a **v100** fechou a frente de pós-resultado com histórico recente da sessão e reabertura fiel do resultado, e a **v101** consolida uma simplificação funcional da revisão de inconsistências, preservando o congelamento estrutural e a lógica central do app, enquanto a **v102** corrige o fluxo de múltiplos faltantes no cadastro guiado e antecipa o alerta de duplicados da base logo após o carregamento, a **v103** preserva qualificadores distintivos na detecção de duplicados da lista para evitar falso positivo em casos como `Douglas` vs `Douglas (pimpim)` e `Joel` vs `Joel (convidado)`, a **v104** prioriza o scroll de revisão para o bloco de cadastro guiado enquanto `CADASTRO_GUIADO_ATIVO` estiver verdadeiro, e a **v115** consolida a revisão dos nomes fora da base em uma lista totalmente visível, trazendo o cadastro guiado para o próprio item ativo da revisão, enquanto a **v116** remove a subseção separada `Fora da base` e coloca o cadastro guiado diretamente abaixo do resumo pré-sorteio, com edição do nome e ação de remover dentro do próprio fluxo do atleta atual; a **v117** remove o scroll direcionado entre faltantes intermediários; a **v118** estabiliza a identidade do formulário guiado; a **v119** neutraliza qualquer destino explícito de scroll do cadastro guiado; a **v120** preserva a posição visual entre faltantes intermediários, mantendo o redirecionamento apenas no último passo para o botão de confirmação; e a **v121** saneia a arquitetura do scroll da revisão, removendo a lógica intermediária de destino do cadastro guiado, eliminando âncoras duplicadas e preservando somente o redirecionamento final para `Confirmar lista final`. A **v122** move todo o cadastro do atleta atual para dentro de um único `st.form` e extrai o bloco `Cadastro guiado dos faltantes` para fora de `render_revisao_pendencias_panel(...)`, preservando apenas o redirecionamento final para `Confirmar lista final`. A **v123** reposiciona `Lista final sugerida` abaixo do fluxo de faltantes, exibe a lista explícita dos atletas faltantes abaixo do aviso principal e faz o botão `Cadastrar faltantes agora` rolar diretamente para o bloco `Cadastro guiado dos faltantes`.

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


A **v124** inaugura a Fase 1 de reorganização sem alterar o motor do app: limpa resíduos de release, reforça o `.gitignore` e isola wrappers/documentos-ponte em `scripts/compat/` e `docs/compat/`, mantendo apenas os caminhos canônicos como referência oficial.


A **v126** é uma release apenas de preparação da reorganização: não move lógica entre arquivos nem altera o motor do app; ela registra o contrato das fronteiras de renderização, o mapa dos estados de revisão/cadastro/confirmação/sorteio e o plano faseado das próximas microetapas seguras a partir da baseline funcional v124.
