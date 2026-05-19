# Auditoria de Consistência dos Contratos — V139

**Microetapa:** v139-docs-auditoria-consistencia-contratos  
**Baseline documental de entrada:** main após merge da v138  
**Commit base:** `e8df6e04c1fb33c04a425814042ead2ca787b487`  
**Natureza:** auditoria documental, sem alteração funcional  
**Escopo alterado:** apenas este relatório

Este relatório audita a consistência entre o índice de governança, o contrato mestre e os contratos operacionais por etapa do Sorteador Pelada PRO.

A auditoria não altera contratos existentes, código funcional, testes, README raiz, manifesto protegido ou arquivos protegidos.

---

## 1. Arquivos auditados

Foram auditados os seguintes documentos:

```text
docs/contracts/README.md
docs/contracts/CONTRATO_OPERACIONAL_APP.md
docs/contracts/etapas/ETAPA_01_CONFIGURACAO_GRUPO_E_BASE.md
docs/contracts/etapas/ETAPA_02_BASE_DE_JOGADORES.md
docs/contracts/etapas/ETAPA_03_CADASTRO_MANUAL_E_GUIADO.md
docs/contracts/etapas/ETAPA_04_LISTA_DA_PELADA.md
docs/contracts/etapas/ETAPA_05_REVISAO_DA_LISTA.md
docs/contracts/etapas/ETAPA_06_CRITERIOS_E_PARAMETROS_OPCIONAIS.md
docs/contracts/etapas/ETAPA_07_SORTEIO_DOS_TIMES.md
docs/contracts/etapas/ETAPA_08_RESULTADO_COMPARTILHAMENTO_E_HISTORICO.md
```

---

## 2. Critérios de auditoria

A consistência foi avaliada pelos seguintes critérios:

1. O índice de governança referencia todos os contratos existentes.
2. O contrato mestre preserva visão global compatível com os contratos por etapa.
3. As oito etapas documentadas cobrem o fluxo operacional descrito no contrato mestre.
4. As regras de sorteio, goleiros, capitão, revisão, cadastro guiado e histórico permanecem coerentes entre os documentos.
5. A documentação preserva a fronteira entre documentação e alteração funcional.
6. Os documentos não criam autorização indevida para alterar código, testes, manifesto protegido ou arquivos protegidos.
7. Divergências históricas ou residuais são classificadas quanto ao impacto.

---

## 3. Resultado executivo

**Status geral:** APROVADO COMO BASE DOCUMENTAL CONSISTENTE, COM OBSERVAÇÕES NÃO BLOQUEANTES.

A estrutura documental formada por v136, v137 e v138 está consistente para uso como referência operacional e de governança.

A v139 não identifica divergência documental bloqueante que exija alteração imediata em contratos existentes antes de seguir para uma próxima microetapa documental ou estrutural.

---

## 4. Consistência do índice de governança

O arquivo `docs/contracts/README.md` cumpre o papel de índice de governança.

### 4.1 Pontos consistentes

- Lista o contrato mestre em `docs/contracts/CONTRATO_OPERACIONAL_APP.md`.
- Lista os oito contratos por etapa em `docs/contracts/etapas/`.
- Define ordem recomendada de leitura.
- Define autoridade documental e limites da documentação.
- Explicita que documentação não altera comportamento funcional.
- Define quando uma microetapa funcional separada deve ser aberta.
- Inclui validação mínima para frente documental.
- Registra a estrutura documental esperada após a v138.

### 4.2 Parecer

O índice está coerente com a estrutura documental atual e não exige correção imediata.

---

## 5. Consistência do contrato mestre

O arquivo `docs/contracts/CONTRATO_OPERACIONAL_APP.md` preserva a visão global do app e continua consistente como referência mestre.

### 5.1 Pontos consistentes

O contrato mestre documenta adequadamente:

- finalidade do app;
- modos de sorteio;
- fluxo operacional geral;
- estados principais da sessão;
- entradas e saídas globais;
- regras de sorteio aleatório e balanceado;
- regras de goleiros;
- regras de capitão;
- revisão e cadastro guiado;
- invalidação de resultado;
- proteção de arquivos;
- validação e critérios de estabilidade.

### 5.2 Observação histórica não bloqueante

O contrato mestre foi criado na v136, antes da criação efetiva dos contratos por etapa na v137. Por isso, a seção de índice dos contratos por etapa ainda contém redação histórica indicando que os contratos detalhados deveriam ser criados em microetapa posterior, preferencialmente v137, e que na v136 eles ainda não existiam.

Essa redação não é um erro funcional nem uma contradição bloqueante, pois descreve corretamente o estado da v136 no momento em que o contrato mestre foi criado. Após v137 e v138, o estado corrente passa a ser governado pelo índice `docs/contracts/README.md`, que registra a estrutura atual.

### 5.3 Classificação

- **Tipo:** residual documental histórico.
- **Impacto funcional:** nenhum.
- **Impacto de governança:** baixo.
- **Correção imediata:** não necessária.
- **Ação futura opcional:** em microetapa documental própria, atualizar o contrato mestre para distinguir explicitamente entre “estado na v136” e “estado documental corrente após v137/v138”.

---

## 6. Consistência dos contratos por etapa

Os oito contratos por etapa cobrem adequadamente o fluxo lógico descrito no contrato mestre.

| Etapa | Documento | Parecer |
|---|---|---|
| 01 | `ETAPA_01_CONFIGURACAO_GRUPO_E_BASE.md` | Coerente com configuração inicial, origem do fluxo e base. |
| 02 | `ETAPA_02_BASE_DE_JOGADORES.md` | Coerente com estrutura da base, atributos e integridade. |
| 03 | `ETAPA_03_CADASTRO_MANUAL_E_GUIADO.md` | Coerente com cadastro manual, cadastro guiado e faltantes. |
| 04 | `ETAPA_04_LISTA_DA_PELADA.md` | Coerente com lista textual, nomes, número de times e seção `Goleiros:`. |
| 05 | `ETAPA_05_REVISAO_DA_LISTA.md` | Coerente com revisão, confirmação, pendências, duplicidades e scrollfix. |
| 06 | `ETAPA_06_CRITERIOS_E_PARAMETROS_OPCIONAIS.md` | Coerente com critérios, goleiros, capitão e parâmetros pré-sorteio. |
| 07 | `ETAPA_07_SORTEIO_DOS_TIMES.md` | Coerente com gate de sorteio, modo aleatório, modo balanceado e assinatura. |
| 08 | `ETAPA_08_RESULTADO_COMPARTILHAMENTO_E_HISTORICO.md` | Coerente com resultado, compartilhamento, snapshots, histórico e risco residual de capitão. |

### 6.1 Parecer

A decomposição em oito etapas é compatível com o fluxograma e com os macroblocos do contrato mestre.

---

## 7. Coerência temática entre documentos

### 7.1 Sorteio aleatório

O contrato mestre define que o sorteio aleatório usa nomes únicos, ignora critérios de equilíbrio e não aplica odds. O contrato da Etapa 07 preserva essa regra.

**Parecer:** consistente.

### 7.2 Sorteio balanceado

O contrato mestre define que o sorteio balanceado depende de base, lista revisada, confirmação e critérios ativos. Os contratos das Etapas 02, 05, 06 e 07 preservam essa sequência.

**Parecer:** consistente.

### 7.3 Goleiros

O contrato mestre define leitura opcional da seção `Goleiros:`, compatibilidade entre quantidade de goleiros e número de times, posição `G` no cadastro guiado e restrição de um goleiro por time quando aplicável. Os contratos das Etapas 02, 03, 04, 06 e 07 preservam essa lógica.

**Parecer:** consistente.

### 7.4 Capitão

O contrato mestre define que o capitão é opcional, pós-montagem dos times, marcado com `(C)` e não altera a otimização. Os contratos das Etapas 06, 07 e 08 preservam essa separação.

**Parecer:** consistente.

### 7.5 Revisão e cadastro guiado

O contrato mestre define revisão obrigatória no fluxo balanceado, cadastro guiado para faltantes e bloqueio do sorteio enquanto houver pendências. Os contratos das Etapas 03 e 05 preservam essa regra.

**Parecer:** consistente.

### 7.6 Resultado, histórico e risco residual

O contrato mestre registra risco residual relacionado ao status de capitão em snapshot histórico. A Etapa 08 preserva esse risco como observação e não tenta corrigi-lo por documentação.

**Parecer:** consistente.

---

## 8. Lacunas ou divergências identificadas

### 8.1 Divergências bloqueantes

Nenhuma divergência documental bloqueante foi identificada.

### 8.2 Lacunas funcionais documentais

Nenhuma lacuna funcional impeditiva foi identificada dentro do escopo documental v136–v138.

### 8.3 Observações não bloqueantes

Foram identificadas apenas observações documentais de baixa prioridade:

1. O contrato mestre preserva redação histórica da v136 indicando que os contratos por etapa seriam criados em microetapa posterior.
2. Os contratos por etapa usam validações mínimas ligeiramente diferentes entre si, enquanto o índice v138 fornece uma validação mínima documental mais completa e atualizada.
3. Alguns contratos por etapa são intencionalmente sintéticos; futuras auditorias funcionais podem detalhar fluxos específicos se houver nova frente de implementação.

Nenhuma dessas observações exige alteração imediata.

---

## 9. Arquivos protegidos e fronteira funcional

A auditoria confirma que a frente documental preserva a fronteira definida nos contratos:

- documentação não autoriza alteração funcional;
- mudanças em `app.py`, `core/`, `ui/`, `state/`, testes ou manifesto protegido exigem microetapa própria;
- arquivos protegidos permanecem fora do escopo desta auditoria;
- risco residual de snapshot/capitão continua registrado sem correção funcional nesta frente.

---

## 10. Recomendações

### 10.1 Recomendação imediata

A v139 pode ser aprovada como auditoria documental, desde que o diff permaneça restrito a:

```text
docs/contracts/audits/AUDITORIA_CONSISTENCIA_CONTRATOS_V139.md
```

### 10.2 Recomendação futura opcional

Uma microetapa documental futura pode atualizar o contrato mestre para registrar explicitamente o estado pós-v137/v138, sem alterar regras funcionais. Essa ação é opcional e não bloqueia a v139.

### 10.3 Recomendação funcional futura

O risco residual sobre capitão em snapshots históricos deve continuar fora da frente documental. Se priorizado, deve ser tratado em microetapa funcional própria, com escopo único em resultado/histórico e validação específica.

---

## 11. Validação recomendada para o PR da v139

Como a v139 é exclusivamente documental, recomenda-se executar:

```bash
python -m pytest tests/test_ui_safe_smoke.py
python -m pytest tests/test_state_smoke.py
python -m pytest tests/test_goleiros_smoke.py
python scripts/quality/protected_scope_hash_guard.py
rm -rf .pytest_cache
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
python scripts/quality/release_artifacts_hygiene_guard.py
echo "STATUS_RELEASE_GUARD=$?"
python scripts/quality/script_exit_codes_contract_guard.py
echo "STATUS_EXIT_CODES_GUARD=$?"
git status --short
```

---

## 12. Parecer final

A documentação contratual do Sorteador Pelada PRO, após as microetapas v136, v137 e v138, está consistente para uso como base documental estável.

A v139 deve ser tratada como auditoria documental aprovada em termos de conteúdo, pendente apenas da validação local padrão antes de ready/merge.
