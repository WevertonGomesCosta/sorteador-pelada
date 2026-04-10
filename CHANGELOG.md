# CHANGELOG

## Padrão oficial para novas entradas

Para cada nova versão, registrar sempre:
- **Versão**
- **Data**
- **Tipo da mudança** (`correção`, `ux`, `reorganização`, `endurecimento`, `documentação`)
- **Resumo objetivo**
- **Arquivos afetados**
- **Validação mínima executada**
- **Observações de congelamento**, quando houver

Modelo sugerido:

```md
## vXX — AAAA-MM-DD
Tipo: correção | ux | reorganização | endurecimento | documentação

Resumo:
- item 1
- item 2

Arquivos afetados:
- caminho/arquivo_1.py
- caminho/arquivo_2.md

Validação:
- python scripts/check_base.py
- CHECKLIST_REGRESSAO.md
```

---

## Histórico técnico consolidado

## v66 — 2026-04-10
Tipo: endurecimento | documentação

Resumo:
- Ampliação incremental do smoke test para cobrir validadores leves, resumos textuais auxiliares e escape seguro do painel de status da sessão.
- Preservação integral da lógica do app e das áreas congeladas, sem tocar em revisão, confirmação ou sorteio.
- Atualização da documentação do plano de smoke test e sincronização da baseline oficial.

Arquivos afetados:
- `tests/test_smoke_base.py`
- `docs/PLANO_SMOKE_TEST_MINIMO.md`
- `docs/BASELINE_OFICIAL.md`
- `CHANGELOG.md`
- `ui/primitives.py`

Validação:
- `python scripts/check_base.py`
- `python scripts/smoke_test_base.py`
- `python scripts/release_guard.py`
- `python -m compileall .`

## v65 — 2026-04-10
Tipo: correção

Resumo:
- Remoção cirúrgica de uma linha indevida inserida dentro do HTML inline usado no scroll da confirmação de senha em `ui/group_config_view.py`.
- Preservação integral do comportamento do fluxo de grupo, sem alteração de lógica, UI sensível ou etapas congeladas.

Arquivos afetados:
- `ui/group_config_view.py`
- `docs/BASELINE_OFICIAL.md`
- `CHANGELOG.md`
- `ui/primitives.py`

Validação:
- `python scripts/check_base.py`
- `python scripts/smoke_test_base.py`
- `python scripts/release_guard.py`
- `python -m compileall .`

## v64 — 2026-04-10
Tipo: endurecimento | documentação

Resumo:
- Criação de uma camada mínima de smoke test funcional para módulos neutros da baseline.
- Adição de runner dedicado por `unittest` sem automação pesada de UI.
- Restauração dos documentos `docs/BASELINE_OFICIAL.md` e `docs/PLANO_SMOKE_TEST_MINIMO.md` para sincronizar a governança citada no contexto operacional.
- Atualização do README e do protocolo de release para incluir a nova validação comportamental mínima.

Arquivos afetados:
- `tests/test_smoke_base.py`
- `scripts/smoke_test_base.py`
- `docs/BASELINE_OFICIAL.md`
- `docs/PLANO_SMOKE_TEST_MINIMO.md`
- `docs/RELEASE_OPERACIONAL.md`
- `README.md`
- `CHANGELOG.md`
- `ui/primitives.py`

Validação:
- `python scripts/check_base.py`
- `python scripts/smoke_test_base.py`
- `python scripts/release_guard.py`

## v63 — 2026-04-09

### Endurecimento / arquitetura
- concluída a micro-etapa 2 de desacoplamento entre `core/flow_guard.py` e a camada de UI
- `obter_criterios_ativos` e `resumo_criterios_ativos` foram movidos para `state/criteria_state.py`
- `core/flow_guard.py` deixou de depender de `ui.summary_strings`
- `scripts/check_base.py` foi ampliado para proteger o novo contrato arquitetural

## v57 — 2026-04-09
Tipo: documentação | endurecimento

Resumo:
- Criação do `CHANGELOG.md` com histórico técnico consolidado da base.
- Formalização do padrão oficial de registro para futuras versões.
- Ampliação do `scripts/check_base.py` para exigir e validar o changelog como artefato de governança.

Arquivos afetados:
- `CHANGELOG.md`
- `scripts/check_base.py`
- `README.md`
- `ui/primitives.py`

Validação:
- `python scripts/check_base.py`
- compilação sintática do projeto

## v56 — 2026-04-09
Tipo: ux

Resumo:
- Padronização textual e visual dos blocos institucionais e informativos não críticos.
- Reorganização do bloco **Sobre este app** e ajuste do painel de status para **Resumo da sessão** / **Próximo passo**.

Arquivos afetados:
- `ui/primitives.py`
- `ui/panels.py`
- `ui/styles.py`

Validação:
- `python scripts/check_base.py`
- compilação sintática do projeto

## v55 — 2026-04-09
Tipo: documentação

Resumo:
- Registro formal da validação curta da UX mobile.
- Consolidação da decisão de manter confirmação/sorteio congelados até evidência recorrente de atrito real.

Arquivos afetados:
- `docs/VALIDACAO_UX_MOBILE_2026-04-09.md`

Validação:
- `python scripts/check_base.py`

## v54 a v51 — 2026-04-09
Tipo: ux

Resumo:
- Rodada curta de UX mobile focada em status, revisão, pendências e densidade visual.
- Ajustes de responsividade, compactação de banners e melhora de legibilidade em telas estreitas.
- Compactação dos cartões de pendência e da apresentação da revisão.

Arquivos afetados:
- `ui/styles.py`
- `ui/review_view.py`
- `ui/primitives.py`

Validação:
- `python scripts/check_base.py`
- `CHECKLIST_REGRESSAO.md`

## v50 — 2026-04-09
Tipo: documentação

Resumo:
- Criação do protocolo oficial de manutenção da base.
- Formalização de rituais antes/depois de editar, tipos de mudança permitidos e validação mínima obrigatória.

Arquivos afetados:
- `docs/MANUTENCAO_OPERACIONAL.md`
- `scripts/check_base.py`
- `README.md`

Validação:
- `python scripts/check_base.py`

## v49 — 2026-04-09
Tipo: endurecimento

Resumo:
- Ampliação do `scripts/check_base.py` para validar contratos da arquitetura documentada.
- Inclusão de ownership de funções críticas, ausência de wrappers legados e preservação de `app.py` como orquestrador.

Arquivos afetados:
- `scripts/check_base.py`

Validação:
- execução real de `python scripts/check_base.py`

## v48 — 2026-04-09
Tipo: documentação

Resumo:
- Documentação formal da arquitetura atual da base.
- Registro dos módulos oficiais, contratos de manutenção e fronteiras entre responsabilidades.

Arquivos afetados:
- `docs/ARQUITETURA_BASE.md`

Validação:
- `python scripts/check_base.py`

## v47 — 2026-04-09
Tipo: endurecimento

Resumo:
- Criação de `state/view_models.py`.
- Extração da interpretação do estado visual do fluxo, da próxima ação e da abertura dos blocos.

Arquivos afetados:
- `state/view_models.py`
- `app.py`
- `scripts/check_base.py`

Validação:
- `python scripts/check_base.py`
- compilação sintática do projeto

## v44 a v46 — 2026-04-09
Tipo: endurecimento | documentação | ux

Resumo:
- Criação do checklist de regressão funcional e do `scripts/check_base.py`.
- Aprimoramentos do rodapé institucional e cálculo automático da última atualização.
- Centralização de governança mínima da base reorganizada.

Arquivos afetados:
- `CHECKLIST_REGRESSAO.md`
- `scripts/check_base.py`
- `ui/primitives.py`
- `ui/styles.py`
- `README.md`

Validação:
- `python scripts/check_base.py`

## v43 — 2026-04-09
Tipo: endurecimento

Resumo:
- Centralização das chaves críticas do `session_state` em `state/keys.py`.
- Atualização dos módulos principais para consumir constantes em vez de strings espalhadas.

Arquivos afetados:
- `state/keys.py`
- `app.py`
- `state/session.py`
- `state/ui_state.py`
- `core/flow_guard.py`
- `ui/group_config_view.py`
- `ui/base_view.py`
- `ui/review_view.py`
- `ui/manual_card.py`

Validação:
- `python scripts/check_base.py`
- compilação sintática do projeto

## v42 — 2026-04-09
Tipo: ux

Resumo:
- Controle automático dos blocos abertos/recolhidos conforme a etapa ativa.
- Priorização de foco visual e redução de scroll, especialmente no mobile.

Arquivos afetados:
- `app.py`
- `ui/group_config_view.py`
- `ui/review_view.py`

Validação:
- compilação sintática do projeto

## v36 a v38 — 2026-04-09
Tipo: reorganização

Resumo:
- Conclusão das Etapas 6, 7 e 8 da reorganização.
- Extração de helpers de fluxo, estado local, pré-sorteio e resultado.
- Limpeza final de wrappers temporários e consolidação de `app.py` como orquestrador.

Arquivos afetados:
- `core/flow_guard.py`
- `state/ui_state.py`
- `ui/pre_sort_view.py`
- `ui/result_view.py`
- `app.py`

Validação:
- `python scripts/check_base.py`
- compilação sintática do projeto

## v28 a v35 — 2026-04-09
Tipo: reorganização | correção | ux

Resumo:
- Conclusão das Etapas 3, 4 e 5 da reorganização (`ui/base_view.py`, `ui/review_view.py`, `ui/group_config_view.py`).
- Correções do fluxo de scroll para pendências e cadastro guiado.
- Ajustes do botão **Revisar lista** para o comportamento final por clique e apresentação estável.

Arquivos afetados:
- `ui/base_view.py`
- `ui/review_view.py`
- `ui/group_config_view.py`
- `app.py`

Validação:
- compilação sintática do projeto
- testes funcionais nos fluxos de revisão

## v24 a v27 — 2026-04-09
Tipo: reorganização | correção

Resumo:
- Conclusão da Etapa 2 (`ui/summary_strings.py`).
- Correções localizadas de revisão: scroll para pendências, cadastro guiado e reconhecimento de escopo da lista principal.

Arquivos afetados:
- `ui/summary_strings.py`
- `app.py`
- `ui/review_view.py` / `ui/sections.py` na época da alteração

Validação:
- compilação sintática do projeto

## v14 a v23 — 2026-04-09
Tipo: reorganização | correção | ux

Resumo:
- Etapa 1 da reorganização (`ui/primitives.py`, `ui/panels.py`, `ui/actions.py`).
- Consolidação do painel de pendências, correções inline de inconsistências e estabilização de duplicados, lista principal e lista de espera.
- Correções sucessivas de revisão com base carregada.

Arquivos afetados:
- `ui/primitives.py`
- `ui/panels.py`
- `ui/actions.py`
- `ui/review_view.py` / `ui/sections.py` na época da alteração
- `app.py`

Validação:
- compilação sintática do projeto
- testes funcionais nos fluxos de revisão e inconsistências

## v02 a v13 — 2026-04-08 a 2026-04-09
Tipo: correção | ux

Resumo:
- Estabilização inicial do scroll de revisão, CTA contextual, painel de status, microcopy inline, limpeza de redundância, seção de resultado e primeiro painel acionável de pendências.
- Consolidação da base funcional que serviu de ponto de partida para a reorganização posterior.

Arquivos afetados:
- `app.py`
- `ui/styles.py`
- `ui/primitives.py`
- `ui/result_view.py`
- `ui/review_view.py` / `ui/sections.py` na época da alteração

Validação:
- compilação sintática do projeto
- validações funcionais sucessivas do fluxo principal

## Observações de congelamento vigentes

- A etapa de **confirmação/sorteio** permanece congelada.
- Só deve ser reaberta se surgirem evidências recorrentes de atrito real nas transições de confirmação e sorteio no uso mobile.
- A lógica central do sorteio e os fluxos estabilizados de revisão por clique e scroll devem permanecer preservados.
