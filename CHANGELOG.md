## v106 — 2026-04-23

- versiona o formulário guiado de cadastro de faltantes por `indice_atual`;
- versiona também todas as keys dos widgets (`Posição`, `Nota`, `Velocidade`, `Movimentação`) por `indice_atual`;
- preserva o scroll para o painel superior de contexto do cadastro (`Etapa atual`) sem alterar a lógica funcional do fluxo.

## v105 — 2026-04-23

### Hotfix funcional — scrollfix reforçado no cadastro guiado sequencial
- reposiciona a âncora de destino do scroll para o topo do painel `Etapa atual`, deixando visível o contexto do jogador antes do formulário;
- aplica scroll estabilizado em múltiplos pulsos com desfoco do elemento ativo, reduzindo a disputa entre rerun, foco residual do botão e assentamento tardio do layout;
- isola a âncora do formulário em id próprio e recria o container do submit por índice do faltante, preservando lógica central, confirmação/sorteio e o algoritmo do app.

## v104 — 2026-04-23

### Hotfix funcional — prioridade de scroll no cadastro guiado
- prioriza `SCROLL_DESTINO_REVISAO = "cadastro"` sempre que `CADASTRO_GUIADO_ATIVO` estiver verdadeiro;
- evita que o destino seja sobrescrito por `pendencias` durante o cadastro sequencial de múltiplos faltantes;
- preserva a lógica central do app, confirmação/sorteio, histórico da sessão e o algoritmo do sorteio.

## v103 — 2026-04-22

### Hotfix funcional — duplicados com qualificadores distintivos
- preserva qualificadores distintivos na leitura e na revisão da lista, evitando falso positivo de duplicados em casos como `Douglas` vs `Douglas (pimpim)` e `Joel` vs `Joel (convidado)`;
- mantém a detecção de duplicados reais quando a diferença é apenas estrutural, como numeração inicial, espaços ou caixa;
- preserva a lógica central do app, confirmação/sorteio, histórico recente da sessão e o algoritmo do sorteio.

## v102 — 2026-04-22

### Hotfix funcional — cadastro guiado e duplicados da base
- corrige a continuidade do cadastro guiado quando há múltiplos faltantes na base, reconstruindo a fila a partir do diagnóstico atualizado após cada cadastro;
- elimina o estado residual de pós-cadastro ao final da fila de faltantes, usando o diagnóstico recalculado como fonte de verdade da revisão;
- exibe imediatamente um alerta de nomes duplicados na base logo após o carregamento na etapa 1, sem alterar lógica central, confirmação/sorteio ou o algoritmo do sorteio.

## v101 — 2026-04-13

### Simplificação funcional — revisão de inconsistências
- remove os botões auxiliares `↺ Restaurar` e `✕ Limpar` da revisão de inconsistências em `ui/review_view.py`, reduzindo ruído visual e reruns intermediários;
- preserva intactos os fluxos principais de `Fora da base` e `Duplicados na lista`, sem alterar lógica central, confirmação/sorteio ou histórico recente da sessão.

## v100 — 2026-04-13

### Pós-resultado — histórico recente da sessão completo
- adiciona snapshot leve do resultado em memória com `times`, `odds`, cabeçalho e texto de compartilhamento para sustentar reexibição fiel sem recalcular o sorteio;
- registra automaticamente os últimos resultados da sessão, evita duplicação por rerun e exibe o bloco visual `Últimos sorteios desta sessão` como elemento secundário na tela final;
- permite reabrir um snapshot anterior da sessão, voltar ao resultado atual e reaproveitar copiar/compartilhar sobre o resultado reexibido, sem tocar no algoritmo do sorteio nem em confirmação/sorteio.

## v99 — 2026-04-11

### UX funcional — micro-melhoria 3
- reescreve o topo interno da revisão em `ui/review_view.py` com status principal mais decisório, orientação contextual curta e indicadores essenciais em ordem de criticidade;
- garante que o resumo pré-sorteio seja renderizado tanto no estado com pendências quanto no estado limpo, sem alterar a lógica do diagnóstico nem os grupos internos da revisão;
- mantém intactos `app.py`, confirmação/sorteio, lógica central, persistência e contratos de compatibilidade temporária.

## v98 — 2026-04-11

### UX funcional — micro-melhoria 2
- adiciona ações rápidas auxiliares dentro de form em `ui/review_view.py` para acelerar pequenas correções sem alterar a lógica do app;
- aplica `↺ Restaurar` e `✕ Limpar` ao campo `Nome corrigido` do grupo `Fora da base`;
- aplica `↺ Restaurar` e `✕ Limpar` por ocorrência editável no grupo `Duplicados na lista`, preservando validação, remoção e submits principais;
- mantém intactos `app.py`, confirmação/sorteio, lógica central, persistência e contratos de compatibilidade temporária.

## v97 — 2026-04-11

### UX funcional — micro-melhoria 1
- melhora o painel de pendências em `ui/review_view.py` com cabeçalhos semânticos e ação principal mais clara nos grupos `Bloqueios da base`, `Fora da base` e `Duplicados na lista`;
- reordena o painel globalmente por criticidade para `Bloqueios da base` → `Fora da base` → `Duplicados na lista`, preservando a lógica interna de cada grupo;
- mantém intactos `app.py`, confirmação/sorteio, lógica central, fluxos de correção existentes e contratos de compatibilidade temporária;
- passa a adotar controle versionado por etapa validada para as próximas micro-melhorias funcionais.

## v96 — 2026-04-11

- adiciona `scripts/reports/maintenance_reports_index.py` para gerar um índice canônico dos artefatos operacionais mais recentes em `reports/`;
- integra o novo índice ao `maintenance_refresh_bundle`, ao cleanup seguro, aos smoke tests e à documentação operacional, sem tocar no núcleo funcional nem nos contratos de compatibilidade temporária.

## v95 — 2026-04-11

- adiciona `scripts/reports/maintenance_refresh_bundle.py` para regenerar, em ordem canônica, os artefatos de manutenção já existentes da baseline;
- sincroniza documentação operacional, inventário estrutural e smoke tests para incluir o novo orquestrador leve de relatórios;

- adiciona `scripts/reports/maintenance_command_journal.py` para consolidar, em modo somente leitura, a ordem prática dos comandos operacionais essenciais da baseline;
- integra o novo utilitário ao snapshot, ao handoff pack, ao resumo curto, ao cleanup de `reports/`, aos smoke tests e à documentação operacional, sem tocar no núcleo funcional nem nos contratos de compatibilidade temporária.

## v93 — 2026-04-11

- adiciona `scripts/reports/maintenance_reports_cleanup.py` para higienização segura de artefatos transitórios em `reports/`, com arquivamento padrão fora do repositório e opção explícita de remoção definitiva;
- integra o novo utilitário aos relatórios de manutenção, ao inventário estrutural, aos smoke tests e à documentação operacional sem tocar no núcleo funcional nem nos contratos de compatibilidade temporária.

## v92 — 2026-04-11

Resumo:
- adiciona `maintenance_resume_brief` em `scripts/reports/maintenance_resume_brief.py` para gerar um resumo operacional curto e pronto para retomada do projeto
- integra o resumo curto ao `maintenance_handoff_pack` como artefato de apoio a handoff e continuidade técnica
- sincroniza inventário estrutural, smoke tests e documentação operacional com o novo utilitário de manutenção, sem tocar em `app.py`, `ui/review_view.py`, confirmação/sorteio, lógica central ou contratos de compatibilidade temporária

## v91 — 2026-04-11

Resumo:
- materializa o `maintenance_snapshot_report` canônico em `scripts/reports/maintenance_snapshot_report.py` como diagnóstico operacional somente leitura da baseline
- adiciona `maintenance_handoff_pack` em `scripts/reports/maintenance_handoff_pack.py` para consolidar snapshot e referências canônicas em um único artefato de handoff
- sincroniza inventário estrutural, smoke tests e documentação operacional com os novos utilitários de manutenção, sem tocar em `app.py`, `ui/review_view.py`, confirmação/sorteio, lógica central ou contratos de compatibilidade temporária

## v89 — 2026-04-11

Resumo:
- adiciona `quality_gate_composition_guard` para validar que `scripts/quality/quality_gate.py` compõe a rotina oficial de checks de forma determinística, completa e sem duplicidade a partir de `scripts/quality/checks_registry.py`
- sincroniza registry, wrappers temporários, relatório de saúde, guards auxiliares, testes e documentação operacional com o novo contrato de composição do runner composto
- mantém intactos `app.py`, `ui/review_view.py`, confirmação/sorteio, lógica central e contratos de compatibilidade temporária

## v88 — 2026-04-11

Resumo:
- adiciona `checks_registry_consumers_guard` para validar que os consumidores oficiais continuam usando exclusivamente `scripts/quality/checks_registry.py` como fonte única de verdade
- sincroniza `quality_gate.py`, `release_health_report.py`, guards auxiliares, wrappers e documentação operacional com o novo contrato de consumo do registro canônico
- mantém intactos `app.py`, `ui/review_view.py`, confirmação/sorteio, lógica central e contratos de compatibilidade temporária

## v87 — 2026-04-11

### Governança e operação
- adiciona `scripts/quality/checks_registry_schema_guard.py` e o wrapper histórico temporário correspondente;
- endurece `scripts/quality/checks_registry.py` com schema explícito por check, incluindo categoria, timeout e flags de inclusão;
- mantém `quality_gate.py` e `scripts/reports/release_health_report.py` consumindo a fonte única de verdade, agora com contrato estrutural validado;
- sincroniza guards auxiliares e documentação operacional com o novo contrato do registro canônico, sem tocar no núcleo funcional.

## v86 — 2026-04-10

### Governança e operação
- adiciona `scripts/quality/checks_registry.py` como fonte única de verdade da lista oficial de checks;
- adiciona `scripts/quality/checks_registry_contract_guard.py` e o wrapper histórico temporário correspondente;
- faz `scripts/quality/quality_gate.py` e `scripts/reports/release_health_report.py` consumirem o registro canônico de checks;
- sincroniza a documentação operacional oficial com o novo contrato, mantendo o núcleo funcional e a compatibilidade temporária intactos.

## v85 — 2026-04-10

### Governança e operação
- adiciona `scripts/quality/protected_scope_hash_guard.py` e o wrapper histórico temporário correspondente;
- formaliza `docs/releases/PROTECTED_SCOPE_HASHES.json` como manifesto oficial de hashes para `app.py` e `ui/review_view.py`;
- integra o novo contrato de escopo protegido à rotina canônica de checks, ao relatório de saúde da release e à documentação operacional oficial;
- mantém o núcleo funcional e os contratos de compatibilidade temporária intactos.

## v84 — 2026-04-10

- adiciona governance_docs_crosslinks_guard à governança operacional leve
- sincroniza crosslinks canônicos entre README, baseline, release, operação local, política e validação
- mantém wrappers temporários e núcleo funcional intactos

## v83 — 2026-04-10

- adiciona script_exit_codes_contract_guard à governança operacional leve
- integra o novo guard ao quality_gate, release_health_report e contratos/documentação canônicos
- mantém wrappers temporários e núcleo funcional intactos

## v82 — 2026-04-10

### Tipo
Endurecimento leve de governança operacional.

### O que mudou
- criado `scripts/quality/quality_runtime_budget_guard.py`;
- criado `scripts/quality_runtime_budget_guard.py` como wrapper histórico temporário;
- integrado o novo guard à rotina oficial de checks, ao relatório de saúde da release e à documentação operacional;
- adicionados timeouts explícitos aos runners compostos canônicos para reduzir risco de degradação silenciosa da rotina de validação;
- ampliados os smoke tests e os contratos operacionais para cobrir o orçamento de execução dos checks.

### Validação
- `python scripts/quality/check_base.py`;
- `python scripts/validation/smoke_test_base.py`;
- `python -m compileall .`;
- `python scripts/quality/release_metadata_guard.py`;
- `python scripts/quality/compatibility_contract_guard.py`;
- `python scripts/quality/operational_checks_contract_guard.py`;
- `python scripts/quality/canonical_paths_reference_guard.py`;
- `python scripts/quality/script_cli_contract_guard.py`;
- `python scripts/quality/release_artifacts_hygiene_guard.py`;
- `python scripts/quality/runtime_dependencies_contract_guard.py`;
- `python scripts/quality/documentation_commands_examples_guard.py`;
- `python scripts/quality/release_manifest_guard.py`;
- `python scripts/quality/quality_runtime_budget_guard.py`;
- `python scripts/quality/release_guard.py`;
- `python scripts/quality/quality_gate.py`.

## v81 — 2026-04-10

### Tipo
Endurecimento leve de governança operacional.

### O que mudou
- criado `scripts/quality/release_manifest_guard.py`;
- criado `scripts/release_manifest_guard.py` como wrapper histórico temporário;
- integrado o novo guard ao `scripts/quality/quality_gate.py` e ao `scripts/reports/release_health_report.py`;
- sincronizados README, documentos operacionais, baseline oficial e protocolo de release com a nova verificação do inventário estrutural obrigatório da release;
- ampliados os smoke tests e os contratos operacionais para cobrir a nova verificação de manifesto estrutural.

### Validação
- `python scripts/quality/check_base.py`;
- `python scripts/validation/smoke_test_base.py`;
- `python -m compileall .`;
- `python scripts/quality/release_metadata_guard.py`;
- `python scripts/quality/compatibility_contract_guard.py`;
- `python scripts/quality/operational_checks_contract_guard.py`;
- `python scripts/quality/canonical_paths_reference_guard.py`;
- `python scripts/quality/script_cli_contract_guard.py`;
- `python scripts/quality/release_artifacts_hygiene_guard.py`;
- `python scripts/quality/runtime_dependencies_contract_guard.py`;
- `python scripts/quality/documentation_commands_examples_guard.py`;
- `python scripts/quality/release_manifest_guard.py`;
- `python scripts/quality/release_guard.py`;
- `python scripts/quality/quality_gate.py`.

## v80 — 2026-04-10

### Tipo
Endurecimento leve de governança operacional.

### O que mudou
- criado `scripts/quality/documentation_commands_examples_guard.py`;
- criado `scripts/documentation_commands_examples_guard.py` como wrapper histórico temporário;
- integrado o novo guard ao `scripts/quality/quality_gate.py` e ao `scripts/reports/release_health_report.py`;
- sincronizados README, documentos operacionais e guia de validação manual para promover apenas exemplos canônicos de comandos;
- ampliados os smoke tests e os contratos operacionais para cobrir a nova verificação dos exemplos documentados.

### Validação
- `python scripts/quality/check_base.py`;
- `python scripts/validation/smoke_test_base.py`;
- `python -m compileall .`;
- `python scripts/quality/release_metadata_guard.py`;
- `python scripts/quality/compatibility_contract_guard.py`;
- `python scripts/quality/operational_checks_contract_guard.py`;
- `python scripts/quality/canonical_paths_reference_guard.py`;
- `python scripts/quality/script_cli_contract_guard.py`;
- `python scripts/quality/release_artifacts_hygiene_guard.py`;
- `python scripts/quality/runtime_dependencies_contract_guard.py`;
- `python scripts/quality/documentation_commands_examples_guard.py`;
- `python scripts/quality/release_guard.py`;
- `python scripts/quality/quality_gate.py`.

## v79 — 2026-04-10

### Tipo
Endurecimento leve de governança operacional.

### O que mudou
- criado `scripts/quality/runtime_dependencies_contract_guard.py`;
- criado `scripts/runtime_dependencies_contract_guard.py` como wrapper histórico temporário;
- integrado o novo guard ao `scripts/quality/quality_gate.py` e ao `scripts/reports/release_health_report.py`;
- sincronizados `requirements.txt`, `runtime_preflight.py`, README e documentação operacional oficial sob contrato verificável;
- ampliados os smoke tests e os contratos operacionais para cobrir a nova verificação de runtime.

### Validação
- `python scripts/quality/check_base.py`;
- `python scripts/validation/smoke_test_base.py`;
- `python -m compileall .`;
- `python scripts/quality/release_metadata_guard.py`;
- `python scripts/quality/compatibility_contract_guard.py`;
- `python scripts/quality/operational_checks_contract_guard.py`;
- `python scripts/quality/canonical_paths_reference_guard.py`;
- `python scripts/quality/script_cli_contract_guard.py`;
- `python scripts/quality/release_artifacts_hygiene_guard.py`;
- `python scripts/quality/runtime_dependencies_contract_guard.py`;
- `python scripts/quality/release_guard.py`;
- `python scripts/quality/quality_gate.py`.

## v78 — 2026-04-10

### Tipo
Endurecimento leve de governança operacional.

### O que mudou
- criado `scripts/quality/release_artifacts_hygiene_guard.py`;
- criado `scripts/release_artifacts_hygiene_guard.py` como wrapper histórico temporário;
- integrado o novo guard ao `scripts/quality/quality_gate.py` e ao `scripts/reports/release_health_report.py`;
- sincronizados README, operação local, protocolo de release e smoke tests com a nova verificação de higiene;
- consolidada a regra de que `reports/` deve permanecer limpo no pacote oficial, contendo apenas `.gitkeep`.

### Validação
- `python scripts/quality/check_base.py`;
- `python scripts/validation/smoke_test_base.py`;
- `python -m compileall .`;
- `python scripts/quality/release_metadata_guard.py`;
- `python scripts/quality/compatibility_contract_guard.py`;
- `python scripts/quality/operational_checks_contract_guard.py`;
- `python scripts/quality/canonical_paths_reference_guard.py`;
- `python scripts/quality/script_cli_contract_guard.py`;
- `python scripts/quality/release_artifacts_hygiene_guard.py`;
- `python scripts/quality/release_guard.py`;
- `python scripts/quality/quality_gate.py`;
- `python scripts/reports/release_health_report.py`.

## v77 — 2026-04-10

### Tipo
Endurecimento leve de governança operacional.

### O que mudou
- criado `scripts/quality/script_cli_contract_guard.py`;
- criado `scripts/script_cli_contract_guard.py` como wrapper histórico temporário;
- integrado o novo guard ao `scripts/quality/quality_gate.py`;
- sincronizado `scripts/reports/release_health_report.py` e os contratos operacionais oficiais com o novo check;
- atualizado README, documentação operacional e smoke tests para cobrir a nova verificação leve de CLI.

### Validação
- `python scripts/quality/check_base.py`;
- `python scripts/validation/smoke_test_base.py`;
- `python -m compileall .`;
- `python scripts/quality/release_metadata_guard.py`;
- `python scripts/quality/compatibility_contract_guard.py`;
- `python scripts/quality/operational_checks_contract_guard.py`;
- `python scripts/quality/canonical_paths_reference_guard.py`;
- `python scripts/quality/script_cli_contract_guard.py`;
- `python scripts/quality/release_guard.py`;
- `python scripts/quality/quality_gate.py`;
- `python scripts/reports/release_health_report.py`.

## v76 — 2026-04-10

### Tipo
Endurecimento leve de governança operacional.

### O que mudou
- criado `scripts/quality/canonical_paths_reference_guard.py`;
- criado `scripts/canonical_paths_reference_guard.py` como wrapper histórico temporário;
- integrado o novo guard ao `scripts/quality/quality_gate.py`;
- sincronizado `scripts/reports/release_health_report.py` com o novo check;
- atualizado o contrato operacional e a documentação oficial para reforçar os caminhos canônicos como padrão.

### Validação
- `python scripts/quality/check_base.py`;
- `python scripts/validation/smoke_test_base.py`;
- `python -m compileall .`;
- `python scripts/quality/release_metadata_guard.py`;
- `python scripts/quality/compatibility_contract_guard.py`;
- `python scripts/quality/operational_checks_contract_guard.py`;
- `python scripts/quality/canonical_paths_reference_guard.py`;
- `python scripts/quality/release_guard.py`;
- `python scripts/quality/quality_gate.py`;
- `python scripts/reports/release_health_report.py`.

## v75 — 2026-04-10

### Objetivo
- endurecer o contrato operacional dos checks canônicos da release, garantindo sincronização entre quality gate, release health report e documentação oficial.

### O que foi feito
- criação do guard canônico `scripts/quality/operational_checks_contract_guard.py` para verificar a sincronização da rotina oficial de checks entre `quality_gate`, `release_health_report` e documentação operacional.
- criação do wrapper histórico temporário `scripts/operational_checks_contract_guard.py`, preservando a compatibilidade temporária já formalizada.
- integração do novo guard ao `quality_gate` e ao `release_health_report`, mantendo a lista oficial de checks canônicos sincronizada.
- atualização de `check_base.py`, `release_guard.py`, `README.md` e documentos operacionais para reconhecer e orientar o uso do novo guard.
- ampliação do smoke leve em `tests/test_scripts_smoke.py` para cobrir import e execução básica do novo contrato operacional.

### Arquivos criados
- `scripts/quality/operational_checks_contract_guard.py`
- `scripts/operational_checks_contract_guard.py`

### Arquivos atualizados
- `CHANGELOG.md`
- `README.md`
- `docs/releases/BASELINE_OFICIAL.md`
- `docs/releases/RELEASE_OPERACIONAL.md`
- `docs/operations/OPERACAO_LOCAL.md`
- `docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md`
- `docs/validation/PLANO_SMOKE_TEST_MINIMO.md`
- `docs/validation/VALIDACAO_MANUAL_GUIA.md`
- `scripts/quality/check_base.py`
- `scripts/quality/release_guard.py`
- `scripts/quality/quality_gate.py`
- `scripts/quality/compatibility_contract_guard.py`
- `scripts/reports/release_health_report.py`
- `tests/test_scripts_smoke.py`
- `ui/primitives.py`

### Validação
- `python scripts/quality/check_base.py`
- `python scripts/validation/smoke_test_base.py`
- `python -m compileall .`
- `python scripts/quality/release_metadata_guard.py`
- `python scripts/quality/compatibility_contract_guard.py`
- `python scripts/quality/operational_checks_contract_guard.py`
- `python scripts/quality/release_guard.py`
- `python scripts/quality/quality_gate.py`
- `python scripts/reports/release_health_report.py`

# CHANGELOG
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

## v73 — 2026-04-10
Tipo: endurecimento | documentação

Resumo:
- Criação do guard canônico `scripts/quality/compatibility_contract_guard.py` para validar wrappers históricos, arquivos-ponte em `docs/` e o agregador `tests/test_smoke_base.py`.
- Integração do novo guard ao `scripts/quality/quality_gate.py`, ao `release_guard.py` e à política oficial de compatibilidade temporária, sem tocar no núcleo funcional do app.
- Promoção do novo check na documentação canônica de operação local, baseline e release.

Arquivos afetados:
- `CHANGELOG.md`
- `README.md`
- `docs/POLITICA_COMPATIBILIDADE_TEMPORARIA.md`
- `docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md`
- `docs/operations/OPERACAO_LOCAL.md`
- `docs/releases/BASELINE_OFICIAL.md`
- `docs/releases/RELEASE_OPERACIONAL.md`
- `docs/validation/PLANO_SMOKE_TEST_MINIMO.md`
- `docs/validation/VALIDACAO_MANUAL_GUIA.md`
- `scripts/compatibility_contract_guard.py`
- `scripts/quality/compatibility_contract_guard.py`
- `scripts/quality/quality_gate.py`
- `scripts/quality/check_base.py`
- `scripts/quality/release_guard.py`
- `tests/test_scripts_smoke.py`
- `ui/primitives.py`

Validação:
- `python scripts/quality/check_base.py`
- `python scripts/validation/smoke_test_base.py`
- `python -m compileall .`
- `python scripts/quality/release_metadata_guard.py`
- `python scripts/quality/compatibility_contract_guard.py`
- `python scripts/quality/release_guard.py`
- `python scripts/quality/quality_gate.py`

## v72 — 2026-04-10
Tipo: endurecimento | documentação

Resumo:
- Criação do guard canônico `scripts/quality/release_metadata_guard.py` para validar a sincronização entre rodapé, changelog e baseline oficial.
- Integração do novo guard ao `scripts/quality/quality_gate.py` e à documentação operacional canônica, sem tocar no núcleo funcional do app.
- Sincronização dos metadados oficiais da release em `ui/primitives.py`, `CHANGELOG.md` e `docs/releases/BASELINE_OFICIAL.md`.

Arquivos afetados:
- `CHANGELOG.md`
- `README.md`
- `docs/releases/BASELINE_OFICIAL.md`
- `docs/releases/RELEASE_OPERACIONAL.md`
- `docs/operations/OPERACAO_LOCAL.md`
- `docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md`
- `docs/validation/PLANO_SMOKE_TEST_MINIMO.md`
- `docs/validation/VALIDACAO_MANUAL_GUIA.md`
- `scripts/release_metadata_guard.py`
- `scripts/quality/release_metadata_guard.py`
- `scripts/quality/quality_gate.py`
- `scripts/quality/check_base.py`
- `scripts/quality/release_guard.py`
- `tests/test_scripts_smoke.py`
- `tests/test_smoke_base.py`
- `ui/primitives.py`

Validação:
- `python scripts/quality/check_base.py`
- `python scripts/validation/smoke_test_base.py`
- `python -m compileall .`
- `python scripts/quality/release_metadata_guard.py`
- `python scripts/quality/release_guard.py`
- `python scripts/quality/quality_gate.py`

## v71 — 2026-04-10
Tipo: documentação | endurecimento

Resumo:
- Formalização da política oficial de compatibilidade temporária para wrappers em `scripts/`, arquivos-ponte em `docs/` e o agregador histórico `tests/test_smoke_base.py`.
- Definição de critérios objetivos para futura remoção controlada do legado temporário, sem tocar no núcleo funcional do app.
- Endurecimento dos guards para exigir a nova política de governança e verificar a presença dos critérios mínimos dessa transição.

Arquivos afetados:
- `README.md`
- `CHANGELOG.md`
- `docs/README.md`
- `docs/MANUTENCAO_OPERACIONAL.md`
- `docs/RELEASE_OPERACIONAL.md`
- `docs/BASELINE_OFICIAL.md`
- `docs/POLITICA_COMPATIBILIDADE_TEMPORARIA.md`
- `docs/operations/MANUTENCAO_OPERACIONAL.md`
- `docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md`
- `docs/releases/BASELINE_OFICIAL.md`
- `docs/releases/RELEASE_OPERACIONAL.md`
- `scripts/quality/check_base.py`
- `scripts/quality/release_guard.py`
- `ui/primitives.py`

Validação:
- `python scripts/quality/check_base.py`
- `python scripts/validation/smoke_test_base.py`
- `python -m compileall .`
- `python scripts/quality/release_guard.py`
- `python scripts/quality/quality_gate.py`

## v70 — 2026-04-10
Tipo: documentação | endurecimento

Resumo:
- Consolidação dos caminhos canônicos reorganizados como padrão oficial de uso em documentação, comandos operacionais e runners auxiliares.
- Manutenção explícita dos wrappers e arquivos-ponte históricos apenas como compatibilidade temporária, sem tocar no núcleo funcional do app.
- Remoção da dependência do runner canônico de smoke test em relação ao agregador legado `tests/test_smoke_base.py`.

Arquivos afetados:
- `README.md`
- `CHECKLIST_REGRESSAO.md`
- `CHANGELOG.md`
- `docs/README.md`
- `docs/ARQUITETURA_BASE.md`
- `docs/MANUTENCAO_OPERACIONAL.md`
- `docs/RELEASE_OPERACIONAL.md`
- `docs/BASELINE_OFICIAL.md`
- `docs/PLANO_SMOKE_TEST_MINIMO.md`
- `docs/OPERACAO_LOCAL.md`
- `docs/VALIDACAO_MANUAL_GUIA.md`
- `docs/architecture/ARQUITETURA_BASE.md`
- `docs/operations/MANUTENCAO_OPERACIONAL.md`
- `docs/operations/OPERACAO_LOCAL.md`
- `docs/releases/BASELINE_OFICIAL.md`
- `docs/releases/RELEASE_OPERACIONAL.md`
- `docs/validation/PLANO_SMOKE_TEST_MINIMO.md`
- `docs/validation/VALIDACAO_MANUAL_GUIA.md`
- `scripts/check_base.py`
- `scripts/release_guard.py`
- `scripts/smoke_test_base.py`
- `scripts/quality_gate.py`
- `scripts/runtime_preflight.py`
- `scripts/manual_validation_pack.py`
- `scripts/quality/check_base.py`
- `scripts/quality/release_guard.py`
- `scripts/quality/quality_gate.py`
- `scripts/quality/runtime_preflight.py`
- `scripts/validation/smoke_test_base.py`
- `scripts/reports/manual_validation_pack.py`
- `tests/test_smoke_base.py`
- `ui/primitives.py`

Validação:
- `python scripts/quality/check_base.py`
- `python scripts/validation/smoke_test_base.py`
- `python -m compileall .`
- `python scripts/quality/release_guard.py`
- `python scripts/quality/quality_gate.py`

## v69 — 2026-04-10
Tipo: reorganização | documentação

Resumo:
- Reorganização operacional leve da documentação em subpastas canônicas por domínio, sem tocar no núcleo funcional do app.
- Reorganização dos scripts auxiliares em `scripts/quality/`, `scripts/validation/` e `scripts/reports/`, preservando wrappers compatíveis nos caminhos históricos.
- Divisão da suíte leve de smoke test em módulos menores, mantendo `tests/test_smoke_base.py` como agregador de compatibilidade.

Arquivos afetados:
- `docs/README.md`
- `docs/architecture/ARQUITETURA_BASE.md`
- `docs/operations/MANUTENCAO_OPERACIONAL.md`
- `docs/operations/OPERACAO_LOCAL.md`
- `docs/releases/BASELINE_OFICIAL.md`
- `docs/releases/RELEASE_OPERACIONAL.md`
- `docs/validation/PLANO_SMOKE_TEST_MINIMO.md`
- `docs/validation/VALIDACAO_MANUAL_GUIA.md`
- `docs/validation/VALIDACAO_UX_MOBILE_2026-04-09.md`
- `docs/ARQUITETURA_BASE.md`
- `docs/MANUTENCAO_OPERACIONAL.md`
- `docs/RELEASE_OPERACIONAL.md`
- `docs/BASELINE_OFICIAL.md`
- `docs/PLANO_SMOKE_TEST_MINIMO.md`
- `docs/VALIDACAO_MANUAL_GUIA.md`
- `scripts/quality/check_base.py`
- `scripts/quality/release_guard.py`
- `scripts/quality/quality_gate.py`
- `scripts/quality/runtime_preflight.py`
- `scripts/validation/smoke_test_base.py`
- `scripts/reports/manual_validation_pack.py`
- `scripts/check_base.py`
- `scripts/release_guard.py`
- `scripts/quality_gate.py`
- `scripts/runtime_preflight.py`
- `scripts/smoke_test_base.py`
- `scripts/manual_validation_pack.py`
- `tests/_smoke_shared.py`
- `tests/test_core_smoke.py`
- `tests/test_state_smoke.py`
- `tests/test_ui_safe_smoke.py`
- `tests/test_smoke_base.py`
- `README.md`
- `CHANGELOG.md`
- `ui/primitives.py`

Validação:
- `python scripts/check_base.py`
- `python scripts/smoke_test_base.py`
- `python -m compileall .`
- `python scripts/release_guard.py`
- `python scripts/quality_gate.py`

## v68 — 2026-04-10
Tipo: endurecimento | documentação

Resumo:
- Adição de um gerador padronizado de relatório para registrar a validação manual local com base no CHECKLIST_REGRESSAO.md.
- Consolidação do fluxo de validação manual em documentação própria, sem tocar na lógica do app nem nas áreas congeladas.
- Endurecimento dos gates para proteger o novo artefato operacional e a pasta oficial de relatórios.

Arquivos afetados:
- `scripts/manual_validation_pack.py`
- `docs/VALIDACAO_MANUAL_GUIA.md`
- `reports/.gitkeep`
- `scripts/check_base.py`
- `scripts/release_guard.py`
- `docs/OPERACAO_LOCAL.md`
- `docs/RELEASE_OPERACIONAL.md`
- `docs/BASELINE_OFICIAL.md`
- `README.md`
- `CHANGELOG.md`
- `ui/primitives.py`

Validação:
- `python scripts/manual_validation_pack.py`
- `python scripts/check_base.py`
- `python scripts/smoke_test_base.py`
- `python -m compileall .`
- `python scripts/release_guard.py`
- `python scripts/quality_gate.py`

## v67 — 2026-04-10
Tipo: endurecimento | documentação

Resumo:
- Adição de um runner único de quality gate para executar os quatro checks técnicos oficiais em sequência.
- Adição de um preflight leve de runtime para verificar dependências e prontidão mínima do ambiente local antes de abrir o app.
- Consolidação do fluxo de operação local em documentação própria, sem tocar na lógica do app nem nas áreas congeladas.

Arquivos afetados:
- `scripts/quality_gate.py`
- `scripts/runtime_preflight.py`
- `scripts/check_base.py`
- `scripts/release_guard.py`
- `docs/OPERACAO_LOCAL.md`
- `docs/RELEASE_OPERACIONAL.md`
- `docs/PLANO_SMOKE_TEST_MINIMO.md`
- `docs/BASELINE_OFICIAL.md`
- `README.md`
- `CHANGELOG.md`
- `ui/primitives.py`

Validação:
- `python scripts/runtime_preflight.py`
- `python scripts/check_base.py`
- `python scripts/smoke_test_base.py`
- `python -m compileall .`
- `python scripts/release_guard.py`
- `python scripts/quality_gate.py`

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
