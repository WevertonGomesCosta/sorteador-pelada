# Índice de Governança dos Contratos — Sorteador Pelada PRO

**Microetapa:** v138-docs-indice-governanca-contratos  
**Baseline documental de entrada:** main após merge da v137  
**Commit base:** `5eace8c521aad6e15295c2d43ec81c0232d30b8b`  
**Natureza:** documentação de governança, sem alteração funcional

Este índice organiza os contratos operacionais do Sorteador Pelada PRO e define como eles devem ser lidos, mantidos e usados em microetapas futuras. Este documento não altera comportamento do app e não autoriza mudança funcional.

---

## 1. Objetivo

O objetivo deste índice é centralizar a governança documental dos contratos do app, indicando:

- qual documento possui papel mestre;
- quais documentos detalham etapas específicas;
- qual é a ordem recomendada de leitura;
- quais regras mínimas devem ser preservadas em alterações documentais futuras;
- quais limites impedem que documentação seja usada como justificativa para mudança funcional não aprovada.

---

## 2. Contrato mestre

O contrato mestre do app é:

```text
docs/contracts/CONTRATO_OPERACIONAL_APP.md
```

Esse documento descreve o comportamento vigente do aplicativo de forma global, incluindo:

- finalidade do app;
- escopo funcional e não funcional;
- visão geral do fluxo;
- estados principais de sessão;
- entradas e saídas globais;
- regras contratuais de sorteio;
- regras de goleiros;
- regras de capitão;
- revisão, cadastro guiado e invalidação de resultado;
- arquivos protegidos;
- scripts de validação;
- critérios para aprovação de versão estável.

O contrato mestre prevalece como referência global. Se houver divergência entre o contrato mestre e um contrato por etapa, a divergência deve ser tratada como pendência documental antes de qualquer promoção de baseline.

---

## 3. Contratos por etapa

Os contratos por etapa detalham partes específicas do fluxo operacional.

| Etapa | Documento | Função |
|---|---|---|
| 01 | `docs/contracts/etapas/ETAPA_01_CONFIGURACAO_GRUPO_E_BASE.md` | Configuração inicial, origem do fluxo e base. |
| 02 | `docs/contracts/etapas/ETAPA_02_BASE_DE_JOGADORES.md` | Estrutura e integridade da base de jogadores. |
| 03 | `docs/contracts/etapas/ETAPA_03_CADASTRO_MANUAL_E_GUIADO.md` | Cadastro manual, cadastro guiado e faltantes. |
| 04 | `docs/contracts/etapas/ETAPA_04_LISTA_DA_PELADA.md` | Entrada textual da lista e seção `Goleiros:`. |
| 05 | `docs/contracts/etapas/ETAPA_05_REVISAO_DA_LISTA.md` | Revisão, correções, pendências e confirmação final. |
| 06 | `docs/contracts/etapas/ETAPA_06_CRITERIOS_E_PARAMETROS_OPCIONAIS.md` | Critérios de equilíbrio, goleiros e capitão. |
| 07 | `docs/contracts/etapas/ETAPA_07_SORTEIO_DOS_TIMES.md` | Gate de sorteio, modo aleatório e modo balanceado. |
| 08 | `docs/contracts/etapas/ETAPA_08_RESULTADO_COMPARTILHAMENTO_E_HISTORICO.md` | Resultado, compartilhamento, snapshots e histórico. |

---

## 4. Ordem recomendada de leitura

A ordem recomendada para auditoria ou manutenção é:

1. Ler o contrato mestre.
2. Identificar a etapa operacional afetada.
3. Ler o contrato específico da etapa.
4. Verificar se a mudança pretendida é documental ou funcional.
5. Conferir se há arquivos protegidos envolvidos.
6. Definir validações mínimas antes de abrir PR.

Para correções funcionais, os contratos devem orientar o escopo, mas não substituem auditoria de código nem validação operacional.

---

## 5. Regra de autoridade documental

A documentação contratual tem autoridade para:

- descrever comportamento vigente;
- registrar regras de manutenção;
- definir escopo de auditoria;
- apoiar análise de regressão;
- orientar futuras microetapas.

A documentação contratual não tem autoridade para:

- alterar comportamento do app por si só;
- liberar mudança funcional sem microetapa própria;
- dispensar testes ou guards;
- modificar arquivos protegidos sem atualização de manifesto;
- reclassificar regressão como comportamento aceitável sem auditoria.

---

## 6. Critérios para mudanças documentais futuras

Uma mudança documental futura deve informar, no mínimo:

- baseline de entrada;
- microetapa candidata;
- arquivos documentais alterados;
- motivo da alteração;
- se há ou não impacto funcional;
- validações executadas;
- estado do PR e parecer de auditoria.

Mudanças documentais que apenas organizam, indexam ou esclarecem contratos não devem alterar código funcional, testes, manifesto protegido ou arquivos protegidos.

---

## 7. Quando abrir microetapa funcional separada

Deve-se abrir microetapa funcional separada quando a análise documental identificar necessidade de:

- alterar `app.py`;
- alterar `core/`;
- alterar `ui/`;
- alterar `state/`;
- alterar testes;
- alterar comportamento de cadastro guiado;
- alterar scrollfix;
- alterar sorteio, otimizador, goleiros ou capitão;
- alterar assinatura ou invalidação de resultado;
- atualizar manifesto de arquivos protegidos.

Nesses casos, a documentação pode registrar a pendência, mas a correção deve ocorrer em PR próprio, com escopo único e validação específica.

---

## 8. Validação mínima para frente documental

Para microetapas documentais sem alteração funcional, a validação mínima recomendada é:

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

Se a microetapa for estritamente documental e não tocar arquivos protegidos, os testes smoke ainda são recomendados como evidência de não regressão operacional mínima.

---

## 9. Estado esperado após a v138

Após a v138, a estrutura documental esperada é:

```text
docs/contracts/
├── README.md
├── CONTRATO_OPERACIONAL_APP.md
└── etapas/
    ├── ETAPA_01_CONFIGURACAO_GRUPO_E_BASE.md
    ├── ETAPA_02_BASE_DE_JOGADORES.md
    ├── ETAPA_03_CADASTRO_MANUAL_E_GUIADO.md
    ├── ETAPA_04_LISTA_DA_PELADA.md
    ├── ETAPA_05_REVISAO_DA_LISTA.md
    ├── ETAPA_06_CRITERIOS_E_PARAMETROS_OPCIONAIS.md
    ├── ETAPA_07_SORTEIO_DOS_TIMES.md
    └── ETAPA_08_RESULTADO_COMPARTILHAMENTO_E_HISTORICO.md
```

---

## 10. Regra de não regressão

Este índice não autoriza regressão em:

- regras do otimizador;
- distribuição de goleiros;
- sorteio de capitão;
- cadastro guiado;
- scrollfix;
- revisão de lista;
- assinatura/invalidação de resultado;
- componentes visuais de resultado;
- testes funcionais existentes;
- manifesto de arquivos protegidos.

Qualquer alteração nesses pontos deve ser tratada fora desta frente documental.
