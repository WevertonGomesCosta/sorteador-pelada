# PLANO_SMOKE_TEST_MINIMO

## Objetivo

Adicionar uma camada mínima de validação comportamental para complementar a validação estrutural já existente, sem transformar o projeto em uma suíte pesada e sem alterar o comportamento do app.

## Escopo aprovado

O smoke test deve permanecer leve, seguro e restrito a módulos neutros.

Prioridades:
- imports mínimos dos módulos principais
- integridade básica de funções puras e neutras
- cenários pequenos e sintéticos
- validação de guardas simples do pré-sorteio

## Módulos-alvo

Base mínima original:
- `core/base_summary.py`
- `state/criteria_state.py`
- `state/view_models.py`
- `core/flow_guard.py`

Ampliação incremental leve aprovada:
- `core/validators.py`
- `ui/summary_strings.py`
- `ui/panels.py` (apenas escape seguro de conteúdo dinâmico)

## O que validar

### 1. Imports mínimos
- importação dos módulos centrais sem erro

### 2. Resumos e critérios
- total e resumo de inconsistências da base
- leitura dos critérios ativos e seus resumos

### 3. Interpretação do estado visual
- visibilidade de revisão
- etapa visual ativa
- resumo de status da sessão
- estado de expansão dos blocos

### 4. Guardas leves do fluxo
- contagem de duplicados normalizada
- assinatura de entrada do sorteio
- invalidação do resultado quando a entrada muda
- gate simples em modo aleatório por lista
- gate simples bloqueado por revisão pendente

### 5. Validadores e resumos auxiliares
- registro válido para sorteio
- diagnóstico leve de bloqueios por duplicidade/inconsistência/ausência
- correção básica de valores numéricos para sliders
- rótulos/resumos mínimos dos expanders
- escape seguro de conteúdo dinâmico no painel de status da sessão

## Complemento operacional aprovado

Para facilitar a execução local sem ampliar o escopo do smoke test, a base pode manter artefatos auxiliares de operação, desde que não alterem a lógica do app:
- `scripts/quality/runtime_preflight.py`
- `scripts/quality/release_metadata_guard.py`
- `scripts/quality/compatibility_contract_guard.py`
- `scripts/quality/quality_gate.py`
- `scripts/reports/release_health_report.py`
- `docs/operations/OPERACAO_LOCAL.md`

Esses artefatos não substituem o smoke test; eles apenas organizam a validação local e a checagem do ambiente.

## Fora de escopo

Não testar:
- UI interativa completa do Streamlit
- clique visual, reruns ou automação pesada
- `ui/review_view.py` além do estritamente necessário
- confirmação/sorteio
- fluxos congelados

## Artefatos da etapa

Artefatos canônicos:
- `tests/test_core_smoke.py`
- `tests/test_state_smoke.py`
- `tests/test_ui_safe_smoke.py`
- `scripts/validation/smoke_test_base.py`

Artefatos de compatibilidade temporária:
- `tests/test_smoke_base.py`

## Critério de aceite

A etapa só é considerada concluída quando a base passar conjuntamente por:

```bash
python scripts/quality/check_base.py
python scripts/validation/smoke_test_base.py
python scripts/quality/release_metadata_guard.py
python scripts/quality/release_guard.py
```


## Organização atual da suíte leve

A suíte canônica foi dividida em:
- `tests/test_core_smoke.py`
- `tests/test_state_smoke.py`
- `tests/test_ui_safe_smoke.py`

O arquivo `tests/test_smoke_base.py` foi mantido apenas como agregador de compatibilidade.
