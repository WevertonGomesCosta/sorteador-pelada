# OPERACAO_LOCAL

> Nesta baseline, os caminhos canônicos reorganizados são o padrão oficial de uso. Wrappers e arquivos-ponte históricos continuam disponíveis apenas como compatibilidade temporária.

## Objetivo

Padronizar a execução local mínima da base para:
- validar dependências do ambiente;
- abrir o app com segurança;
- executar os checks técnicos oficiais;
- registrar a validação manual no navegador.

## Fluxo recomendado

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Fazer a pré-checagem do ambiente

```bash
python scripts/quality/runtime_preflight.py
```

Essa etapa confirma, de forma leve, que:
- os arquivos essenciais da base estão presentes;
- as dependências declaradas estão importáveis;
- o ambiente está apto para abrir o app.

### 3. Rodar o quality gate técnico

```bash
python scripts/quality/quality_gate.py
```

Validação leve dos metadados de release, quando a rodada alterar versão, changelog, baseline ou rodapé:

```bash
python scripts/quality/release_metadata_guard.py
```

Verificação do contrato de compatibilidade temporária durante a fase de transição estável:

```bash
python scripts/quality/compatibility_contract_guard.py
```

Verificação de que README, documentos operacionais e wrappers continuam promovendo os caminhos canônicos como padrão oficial:

```bash
python scripts/quality/canonical_paths_reference_guard.py
```

Verificação do contrato mínimo de CLI dos scripts canônicos e wrappers temporários:

```bash
python scripts/quality/script_cli_contract_guard.py
```

Verificação de higiene do pacote de release, exigindo ausência de resíduos transitórios e mantendo `reports/` limpo no pacote oficial:

```bash
python scripts/quality/release_artifacts_hygiene_guard.py
```

Verificação do contrato mínimo de dependências de runtime local, exigindo sincronização entre `requirements.txt`, `runtime_preflight.py` e a documentação operacional oficial:

```bash
python scripts/quality/runtime_dependencies_contract_guard.py
```

Verificação de que os exemplos de comandos em README e documentos operacionais continuam válidos, canônicos e coerentes com os scripts reais:

```bash
python scripts/quality/documentation_commands_examples_guard.py
```

Verificação do inventário estrutural obrigatório da release:

```bash
python scripts/quality/release_manifest_guard.py
python scripts/quality/quality_runtime_budget_guard.py
python scripts/quality/protected_scope_hash_guard.py
```

Esse runner executa, em sequência:
- `python scripts/quality/check_base.py`
- `python scripts/validation/smoke_test_base.py`
- `python -m compileall .`
- `python scripts/quality/release_metadata_guard.py`
- `python scripts/quality/compatibility_contract_guard.py`
- `python scripts/quality/operational_checks_contract_guard.py`
- `python scripts/quality/canonical_paths_reference_guard.py`
- `python scripts/quality/script_cli_contract_guard.py`
- `python scripts/quality/release_artifacts_hygiene_guard.py`
- `python scripts/quality/runtime_dependencies_contract_guard.py`
- `python scripts/quality/documentation_commands_examples_guard.py`
- `python scripts/quality/release_manifest_guard.py`, `python scripts/quality/quality_runtime_budget_guard.py` e `python scripts/quality/protected_scope_hash_guard.py`
- `python scripts/quality/quality_gate_composition_guard.py`
- `python scripts/quality/release_guard.py`

### 4. Gerar o relatório-base da validação manual

```bash
python scripts/reports/manual_validation_pack.py
python scripts/reports/release_health_report.py
python scripts/reports/maintenance_snapshot_report.py
python scripts/reports/maintenance_handoff_pack.py
python scripts/reports/maintenance_resume_brief.py
python scripts/reports/maintenance_reports_cleanup.py
```

Esses comandos criam arquivos locais em `reports/` para apoio operacional, triagem e handoff. Antes de empacotar a baseline oficial, execute `python scripts/reports/maintenance_reports_cleanup.py` para higienizar `reports/` com segurança e voltar a conter apenas `.gitkeep`.

## Execução do app

Com o ambiente pronto:

```bash
streamlit run app.py
```

## Validação manual final

Depois da abertura do app, executar o checklist usando o relatório gerado em `reports/`.

Quando a necessidade for leitura rápida do estado da baseline, retomada operacional curta, empacotamento de referências para revisão/handoff ou higiene final de `reports/`, usar `python scripts/reports/maintenance_snapshot_report.py`, `python scripts/reports/maintenance_handoff_pack.py`, `python scripts/reports/maintenance_resume_brief.py` e `python scripts/reports/maintenance_reports_cleanup.py`.

Referências da rodada:
- `CHECKLIST_REGRESSAO.md`
- `docs/validation/VALIDACAO_MANUAL_GUIA.md`

Registrar apenas falhas reproduzíveis, com:
- item do checklist;
- passos executados;
- observado;
- esperado;
- frequência;
- ambiente;
- evidência visual, quando houver.

## Restrições operacionais

Durante a validação local:
- não reabrir arquitetura ampla;
- não mexer em `ui/review_view.py` sem necessidade concreta;
- não alterar confirmação/sorteio sem defeito comprovado;
- preservar a baseline oficial vigente.


## Nota sobre organização

Os caminhos canônicos em `scripts/quality/`, `scripts/validation/` e `scripts/reports/` são o padrão oficial.
Os comandos históricos em `scripts/` continuam válidos apenas como compatibilidade temporária.


Verificação do orçamento operacional da rotina oficial de checks:

```bash
python scripts/quality/quality_runtime_budget_guard.py
```

Esse guard confirma que a rotina oficial permanece dentro de um orçamento operacional razoável e com timeouts explícitos nos runners compostos.

Verificação do escopo protegido e do manifesto oficial de hashes:

```bash
python scripts/quality/protected_scope_hash_guard.py
```

Esse guard confirma que `app.py` e `ui/review_view.py` permanecem alinhados ao manifesto `docs/releases/PROTECTED_SCOPE_HASHES.json`.


python scripts/quality/script_exit_codes_contract_guard.py


Observação operacional: os scripts canônicos de governança devem manter **códigos de saída previsíveis** para sucesso e falha controlada.

## Crosslinks canônicos de governança

A operação local deve ser lida em conjunto com:
- `docs/releases/BASELINE_OFICIAL.md`
- `docs/releases/RELEASE_OPERACIONAL.md`
- `docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md`
- `docs/validation/VALIDACAO_MANUAL_GUIA.md`

Guard leve desta coerência documental:

```bash
python scripts/quality/governance_docs_crosslinks_guard.py
```


Registro canônico dos checks:

python scripts/quality/checks_registry_contract_guard.py
python scripts/quality/checks_registry_schema_guard.py
python scripts/quality/checks_registry_consumers_guard.py
python scripts/quality/quality_gate_composition_guard.py

fonte única de verdade: `scripts/quality/checks_registry.py`

Schema canônico do registro: `scripts/quality/checks_registry_schema_guard.py`

Consumo exclusivo do checks_registry canônico: `scripts/quality/checks_registry_consumers_guard.py`

Composição determinística do quality_gate: `scripts/quality/quality_gate_composition_guard.py`
