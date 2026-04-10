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

## Fora de escopo

Não testar:
- UI interativa completa do Streamlit
- clique visual, reruns ou automação pesada
- `ui/review_view.py` além do estritamente necessário
- confirmação/sorteio
- fluxos congelados

## Artefatos da etapa

- `tests/test_smoke_base.py`
- `scripts/smoke_test_base.py`

## Critério de aceite

A etapa só é considerada concluída quando a base passar conjuntamente por:

```bash
python scripts/check_base.py
python scripts/smoke_test_base.py
python scripts/release_guard.py
```
