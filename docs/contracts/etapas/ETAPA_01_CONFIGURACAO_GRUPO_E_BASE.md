# Etapa 01 — Configuração do Grupo e Base

**Microetapa:** v137-docs-contratos-operacionais-etapas  
**Baseline documental de entrada:** v136  
**Commit base:** `6349d3eab92b7cb82d79e21843c109bdb16093b7`  
**Natureza:** contrato operacional por etapa, sem alteração funcional

Este documento detalha a etapa de configuração inicial do Sorteador Pelada PRO. Ele complementa o contrato mestre `docs/contracts/CONTRATO_OPERACIONAL_APP.md` e não autoriza mudança de comportamento no app.

---

## 1. Finalidade

A etapa de configuração define o modo inicial de uso do app e estabelece se a sessão trabalhará com:

1. sorteio apenas por lista;
2. base do grupo;
3. Excel próprio carregado pelo usuário.

Essa decisão condiciona as etapas seguintes, especialmente cadastro, revisão, critérios e sorteio.

---

## 2. Entradas operacionais

As entradas reconhecidas nesta etapa são:

- modo de uso selecionado;
- nome da pelada, quando aplicável;
- senha administrativa, quando aplicável;
- arquivo Excel próprio, quando aplicável;
- decisão explícita de usar somente lista, quando não houver base.

---

## 3. Estados envolvidos

Os estados de sessão relacionados são:

| Estado | Papel operacional |
|---|---|
| `grupo_origem_fluxo` | Registra a origem escolhida para o fluxo. |
| `grupo_busca_status` | Registra status de busca ou carregamento da base do grupo. |
| `grupo_nome_pelada` | Armazena a identificação da pelada/grupo. |
| `grupo_senha_admin` | Armazena senha administrativa quando exigida. |
| `senha_admin_confirmada` | Indica se a credencial foi validada. |
| `df_base` | Guarda a base ativa, quando existente. |
| `base_admin_carregada` | Indica carregamento administrativo da base. |
| `ultimo_arquivo` | Registra o arquivo carregado mais recentemente. |

---

## 4. Regras contratuais

1. A configuração deve ocorrer antes da revisão e antes do sorteio.
2. O modo somente lista não deve exigir base carregada.
3. O modo com base deve preservar a origem da base para diagnóstico posterior.
4. O carregamento de Excel próprio não deve sobrescrever silenciosamente estados críticos sem reinicialização compatível.
5. A troca de origem deve invalidar resultados ou revisões que dependam da origem anterior.
6. A etapa não deve executar sorteio nem otimização.
7. A etapa não deve alterar critérios de equilíbrio diretamente.

---

## 5. Saídas esperadas

A etapa pode produzir:

- status visual de configuração;
- base ativa carregada;
- ausência deliberada de base, no modo somente lista;
- mensagens de erro ou orientação;
- liberação condicional da etapa de lista e base.

---

## 6. Bloqueios

A etapa deve bloquear avanço operacional quando:

- a credencial exigida não foi confirmada;
- o arquivo Excel próprio não pode ser lido;
- a base carregada está ausente quando o modo exige base;
- há inconsistência estrutural impeditiva na origem dos dados.

---

## 7. Não regressão

Alterações futuras nesta etapa não devem:

- alterar o otimizador;
- alterar o cadastro guiado;
- alterar o sorteio de capitão;
- alterar regras de goleiros;
- alterar arquivos protegidos sem manifesto;
- mascarar erros de carregamento de base.

---

## 8. Validação mínima recomendada

```bash
python -m pytest tests/test_ui_safe_smoke.py
python -m pytest tests/test_state_smoke.py
python scripts/quality/protected_scope_hash_guard.py
python scripts/quality/release_artifacts_hygiene_guard.py
python scripts/quality/script_exit_codes_contract_guard.py
git status --short
```
