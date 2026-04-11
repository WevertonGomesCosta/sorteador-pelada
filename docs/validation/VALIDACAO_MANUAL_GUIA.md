# VALIDACAO_MANUAL_GUIA

> Nesta baseline, os caminhos canônicos reorganizados são o padrão oficial de uso. Wrappers e arquivos-ponte históricos continuam disponíveis apenas como compatibilidade temporária.

## Objetivo

Padronizar o registro da validação manual final no navegador real, sem reabrir a arquitetura nem tocar nas áreas congeladas.

## Fluxo recomendado

### 1. Preparar o ambiente

```bash
pip install -r requirements.txt
python scripts/quality/runtime_preflight.py
python scripts/quality/release_metadata_guard.py
python scripts/quality/compatibility_contract_guard.py
python scripts/quality/documentation_commands_examples_guard.py
python scripts/quality/release_manifest_guard.py
python scripts/quality/quality_runtime_budget_guard.py
python scripts/quality/protected_scope_hash_guard.py
python scripts/quality/quality_gate.py
```

### 2. Gerar o relatório-base da validação

```bash
python scripts/reports/manual_validation_pack.py
python scripts/reports/release_health_report.py
```

Esse comando canônico cria um arquivo em `reports/` com:
- metadados da rodada;
- checklist manual completo;
- bloco padronizado para registrar falhas reproduzidas;
- seção final de conclusão.

### 3. Abrir o app

```bash
streamlit run app.py
```

### 4. Executar a validação manual

Usar o arquivo gerado em `reports/` como registro oficial da rodada.

Preencher apenas:
- os itens do checklist percorridos;
- as falhas reproduzidas de forma objetiva;
- a conclusão final da rodada.

## Regras de registro

- registrar apenas falhas reproduzíveis;
- uma falha por bloco;
- não misturar problema visual com problema lógico no mesmo item;
- informar se a falha ocorreu com lista manual, base do grupo ou Excel próprio;
- anexar print quando houver.

## Restrições operacionais

Durante a validação manual:
- não reabrir arquitetura ampla;
- não mexer em `ui/review_view.py` sem defeito real;
- não alterar confirmação/sorteio sem evidência prática;
- preservar a baseline oficial vigente.


## Observação operacional adicional

Antes de fechar a rodada, a base pode validar a interface mínima dos scripts operacionais com `python scripts/quality/script_cli_contract_guard.py` e confirmar que os exemplos documentados continuam válidos com `python scripts/quality/documentation_commands_examples_guard.py` e validar o inventário estrutural da release com `python scripts/quality/release_manifest_guard.py`, `python scripts/quality/quality_runtime_budget_guard.py` e `python scripts/quality/protected_scope_hash_guard.py`, sem tocar no núcleo funcional.


python scripts/quality/script_exit_codes_contract_guard.py

## Crosslinks canônicos de governança

A validação manual deve permanecer conectada a:
- `docs/releases/BASELINE_OFICIAL.md`
- `docs/releases/RELEASE_OPERACIONAL.md`
- `docs/operations/OPERACAO_LOCAL.md`
- `docs/validation/PLANO_SMOKE_TEST_MINIMO.md`

Guard leve desta coerência documental:

```bash
python scripts/quality/governance_docs_crosslinks_guard.py
```


Registro canônico dos checks:

python scripts/quality/checks_registry_contract_guard.py
python scripts/quality/checks_registry_schema_guard.py
python scripts/quality/checks_registry_consumers_guard.py
python scripts/quality/quality_gate_composition_guard.py

Fonte única de verdade: `scripts/quality/checks_registry.py`

Schema canônico do registro: `scripts/quality/checks_registry_schema_guard.py`

Consumo exclusivo do checks_registry canônico: `scripts/quality/checks_registry_consumers_guard.py`

Composição determinística do quality_gate: `scripts/quality/quality_gate_composition_guard.py`
