# Etapa 07 — Sorteio dos Times

**Microetapa:** v137-docs-contratos-operacionais-etapas  
**Baseline documental de entrada:** v136  
**Commit base:** `6349d3eab92b7cb82d79e21843c109bdb16093b7`  
**Natureza:** contrato operacional por etapa, sem alteração funcional

Este documento define o contrato da execução do sorteio dos times.

---

## 1. Finalidade

A etapa de sorteio converte a lista operacional confirmada em times, usando sorteio aleatório por lista ou sorteio balanceado com base.

---

## 2. Entradas operacionais

A etapa recebe:

- lista final confirmada;
- número de times;
- base ativa, quando aplicável;
- critérios ativos;
- parâmetro de goleiros;
- parâmetro de capitão;
- assinatura de entrada.

---

## 3. Estados envolvidos

| Estado | Papel operacional |
|---|---|
| `resultado` | Guarda times sorteados vigentes. |
| `resultado_contexto` | Guarda contexto do sorteio. |
| `resultado_assinatura` | Guarda assinatura da entrada usada. |
| `resultado_invalidado_msg` | Registra mensagem de invalidação. |
| `sortear_goleiros` | Define restrição operacional de goleiros quando aplicável. |
| `sortear_capitao` | Define marcação posterior de capitão. |

---

## 4. Regras contratuais

1. O sorteio só deve ocorrer após a lista estar pronta conforme o modo escolhido.
2. No modo aleatório, os nomes únicos são embaralhados e distribuídos em rodízio.
3. No modo aleatório, critérios de equilíbrio e odds não se aplicam.
4. No modo balanceado, o sorteio usa a base e os critérios ativos.
5. No modo balanceado, a montagem dos times é delegada ao otimizador.
6. Se houver goleiros compatíveis, o modelo deve impor exatamente um goleiro por time no sorteio balanceado.
7. O capitão deve ser marcado após a montagem dos times.
8. O resultado deve armazenar assinatura suficiente para invalidação posterior.

---

## 5. Saídas esperadas

A etapa produz:

- times sorteados;
- odds, quando aplicáveis;
- contexto do sorteio;
- assinatura do resultado;
- status de critérios e parâmetros;
- mensagem de sucesso ou bloqueio.

---

## 6. Bloqueios

O sorteio deve ser bloqueado quando:

- lista final não foi confirmada no fluxo balanceado;
- cadastro guiado está ativo;
- há faltantes ou inconsistências bloqueantes;
- número de times é incompatível;
- base exigida está ausente ou inválida;
- entrada atual diverge do estado validado.

---

## 7. Não regressão

Alterações futuras não devem:

- liberar sorteio com pendência de revisão;
- alterar o comportamento do otimizador sem microetapa funcional específica;
- permitir goleiros incompatíveis como restrição válida;
- fazer capitão interferir na otimização;
- deixar resultado antigo ativo após alteração de entrada.

---

## 8. Validação mínima recomendada

```bash
python -m pytest tests/test_ui_safe_smoke.py
python -m pytest tests/test_state_smoke.py
python -m pytest tests/test_goleiros_smoke.py
python scripts/quality/protected_scope_hash_guard.py
python scripts/quality/release_artifacts_hygiene_guard.py
python scripts/quality/script_exit_codes_contract_guard.py
git status --short
```
