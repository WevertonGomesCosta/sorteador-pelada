# Etapa 06 — Critérios e Parâmetros Opcionais

**Microetapa:** v137-docs-contratos-operacionais-etapas  
**Baseline documental de entrada:** v136  
**Commit base:** `6349d3eab92b7cb82d79e21843c109bdb16093b7`  
**Natureza:** contrato operacional por etapa, sem alteração funcional

Este documento define o contrato dos critérios de equilíbrio e dos parâmetros opcionais usados antes do sorteio.

---

## 1. Finalidade

A etapa define como o usuário escolhe os critérios de equilíbrio e parâmetros opcionais que influenciam a montagem ou a apresentação dos times.

---

## 2. Entradas operacionais

A etapa recebe:

- critério de posição;
- critério de nota;
- critério de velocidade;
- critério de movimentação;
- parâmetro `sortear_goleiros`;
- parâmetro `sortear_capitao`;
- número de times já definido.

---

## 3. Estados envolvidos

| Estado | Papel operacional |
|---|---|
| `criterio_posicao` | Ativa ou desativa equilíbrio por posição. |
| `criterio_nota` | Ativa ou desativa equilíbrio por nota. |
| `criterio_velocidade` | Ativa ou desativa equilíbrio por velocidade. |
| `criterio_movimentacao` | Ativa ou desativa equilíbrio por movimentação. |
| `sortear_goleiros` | Controla inclusão operacional dos goleiros detectados. |
| `sortear_capitao` | Controla marcação de capitão após montagem dos times. |
| `qtd_times_sorteio` | Define quantidade de times. |

---

## 4. Regras contratuais

1. Critérios de equilíbrio se aplicam ao sorteio balanceado.
2. No modo aleatório por lista, critérios de equilíbrio não devem gerar odds nem otimização.
3. Por padrão, os critérios podem operar como conjunto completo.
4. Desativar todos ou parte dos critérios caracteriza configuração personalizada.
5. `sortear_goleiros` só deve ser operacional quando a quantidade de goleiros é compatível com o número de times.
6. `sortear_capitao` deve ser aplicado após a montagem dos times.
7. O capitão não deve alterar o resultado da otimização.
8. Mudanças em parâmetros relevantes devem participar da assinatura de invalidação do resultado.

---

## 5. Saídas esperadas

A etapa pode produzir:

- resumo dos critérios ativos;
- indicação de perfil padrão ou personalizado;
- status dos goleiros;
- status do capitão;
- pendências pré-sorteio;
- parâmetros para o gate de sorteio.

---

## 6. Bloqueios

A etapa deve bloquear ou orientar quando:

- goleiros foram solicitados com quantidade incompatível;
- lista ainda não foi confirmada;
- cadastro guiado ainda está ativo;
- critérios ou parâmetros estão incoerentes com o modo escolhido.

---

## 7. Não regressão

Alterações futuras não devem:

- fazer capitão alterar o otimizador;
- permitir goleiro incompatível como se estivesse válido;
- aplicar critérios balanceados ao modo aleatório como se houvesse base;
- deixar resultado antigo ativo após mudança de parâmetro relevante.

---

## 8. Validação mínima recomendada

```bash
python -m pytest tests/test_ui_safe_smoke.py
python -m pytest tests/test_goleiros_smoke.py
python scripts/quality/protected_scope_hash_guard.py
python scripts/quality/release_artifacts_hygiene_guard.py
python scripts/quality/script_exit_codes_contract_guard.py
git status --short
```
