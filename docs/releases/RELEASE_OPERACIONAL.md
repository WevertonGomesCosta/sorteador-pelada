# RELEASE_OPERACIONAL

> Nesta baseline, os caminhos canônicos reorganizados são o padrão oficial de uso. Wrappers e arquivos-ponte históricos continuam disponíveis apenas como compatibilidade temporária.

## Organização operacional atual

Caminhos canônicos desta baseline:
- documentação principal em `docs/architecture/`, `docs/operations/`, `docs/validation/` e `docs/releases/`
- scripts canônicos em `scripts/quality/`, `scripts/validation/` e `scripts/reports/`
- suíte leve dividida em `tests/test_core_smoke.py`, `tests/test_state_smoke.py` e `tests/test_ui_safe_smoke.py`

Os caminhos históricos na raiz de `docs/`, `scripts/` e `tests/test_smoke_base.py` foram preservados apenas como ponte temporária de compatibilidade. O padrão oficial de uso desta baseline é a estrutura canônica reorganizada.

A política oficial para manutenção e futura remoção controlada desses elementos está em `docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md`.


## Objetivo

Este documento define o protocolo oficial para fechar uma nova release da base do **Sorteador Pelada PRO** com consistência entre:

- comportamento funcional do app
- `CHANGELOG.md`
- rodapé com versão da base
- documentação de governança
- artefatos entregues no `.zip`

A release só deve ser considerada concluída quando a base estiver **funcionalmente estável**, **validada tecnicamente** e **documentalmente sincronizada**.

---

## Regra de ouro

Nenhuma release deve ser fechada se houver qualquer divergência entre:

- versão exibida no rodapé do app
- versão registrada no `CHANGELOG.md`
- artefatos/documentos exigidos pela governança da base
- resultado de `python scripts/quality/check_base.py`
- resultado de `python scripts/validation/smoke_test_base.py`
- resultado de `python scripts/quality/release_metadata_guard.py`
- resultado de `python scripts/quality/canonical_paths_reference_guard.py`
- resultado de `python scripts/quality/script_cli_contract_guard.py`
- resultado de `python scripts/quality/release_artifacts_hygiene_guard.py`
- resultado de `python scripts/quality/documentation_commands_examples_guard.py`
- resultado de `python scripts/quality/release_guard.py`

Se houver divergência, a release deve ser interrompida e corrigida antes de gerar o `.zip` final.

---

## Tipos de mudança aceitos em release

### 1. Correção
Mudança localizada para corrigir bug, regressão, texto incorreto ou fluxo quebrado.

Exemplos:
- correção de `NameError`
- correção de scroll
- correção de renderização mobile

### 2. UX
Mudança incremental de apresentação, densidade visual, microcopy ou legibilidade.

Exemplos:
- compactação de cards
- ajustes mobile
- melhoria de rodapé ou painéis informativos

### 3. Reorganização
Mudança estrutural entre scripts ou módulos, sem alterar o comportamento funcional do app.

Exemplos:
- extração de funções para novos módulos
- separação por domínio funcional
- limpeza de wrappers legados

### 4. Endurecimento
Mudança de proteção da base, governança, checagem ou manutenção.

Exemplos:
- `state/keys.py`
- `scripts/quality/check_base.py`
- `state/view_models.py`
- documentação de arquitetura/manutenção

### 5. Documentação
Mudança documental sem impacto funcional no app.

Exemplos:
- arquitetura
- manutenção operacional
- changelog
- guia de release

---

## Convenção de versão da base

A versão da base segue o padrão interno simples:

- `vN`
- onde `N` é um inteiro sequencial crescente

### Quando subir a versão

Subir a versão sempre que houver qualquer nova entrega oficial da base, incluindo:
- correção
- UX
- reorganização
- endurecimento
- documentação

### O que não conta como nova versão

Não gerar nova versão oficial apenas para:
- rascunhos locais
- testes incompletos
- tentativas não validadas
- mudanças que foram revertidas antes do fechamento da release

---

## Ritual obrigatório antes de editar

Antes de iniciar uma nova mudança:

1. identificar o **tipo da mudança**
2. confirmar se a área está **congelada** ou não
3. localizar o **módulo oficial** onde a mudança deve acontecer
4. revisar:
   - `docs/architecture/ARQUITETURA_BASE.md`
   - `docs/operations/MANUTENCAO_OPERACIONAL.md`
   - `CHANGELOG.md`
5. rodar:

```bash
python scripts/quality/runtime_preflight.py
python scripts/quality/check_base.py
python scripts/validation/smoke_test_base.py
python -m compileall .
python scripts/quality/release_metadata_guard.py
python scripts/quality/compatibility_contract_guard.py
python scripts/quality/operational_checks_contract_guard.py
python scripts/quality/canonical_paths_reference_guard.py
python scripts/quality/script_cli_contract_guard.py
python scripts/quality/release_artifacts_hygiene_guard.py
python scripts/quality/runtime_dependencies_contract_guard.py
python scripts/quality/documentation_commands_examples_guard.py
python scripts/quality/release_manifest_guard.py
python scripts/quality/quality_runtime_budget_guard.py
python scripts/quality/protected_scope_hash_guard.py
```

Se a base já falhar antes da mudança, não iniciar a release sem primeiro estabilizar o projeto.

---

## Ritual obrigatório durante a mudança

Durante a implementação:

1. manter a mudança **local e proporcional ao escopo**
2. não reabrir fluxos congelados sem evidência prática consistente
3. não misturar numa mesma release:
   - correção funcional sensível
   - reorganização estrutural ampla
   - UX em área congelada
4. atualizar o `CHANGELOG.md` somente quando a solução já estiver estável

---

## Ritual obrigatório depois de editar

Antes de fechar a release:

1. rodar novamente:

```bash
python scripts/quality/release_metadata_guard.py
python scripts/quality/compatibility_contract_guard.py
python scripts/quality/operational_checks_contract_guard.py
python scripts/quality/canonical_paths_reference_guard.py
python scripts/quality/script_cli_contract_guard.py
python scripts/quality/release_artifacts_hygiene_guard.py
python scripts/quality/runtime_dependencies_contract_guard.py
python scripts/quality/documentation_commands_examples_guard.py
python scripts/quality/release_manifest_guard.py
python scripts/quality/quality_runtime_budget_guard.py
python scripts/quality/protected_scope_hash_guard.py
python scripts/quality/quality_gate.py
```

2. gerar o relatório-base da validação manual com `python scripts/reports/manual_validation_pack.py`
3. gerar o relatório consolidado de saúde da release com `python scripts/reports/release_health_report.py`
4. executar a validação mínima manual conforme `CHECKLIST_REGRESSAO.md`
4. atualizar o `CHANGELOG.md`
5. sincronizar a versão exibida no rodapé do app
6. revisar a data da última atualização, quando a base depender de data explícita
7. garantir que o `.zip` final não contenha:
   - `__pycache__`
   - `.pyc`
   - arquivos transitórios de teste
   - relatórios antigos ou locais em `reports/`

---

## Inventário estrutural obrigatório da release

Além dos checks canônicos, a baseline deve preservar o inventário estrutural mínimo do pacote oficial. Essa verificação agora é formalizada por:

```bash
python scripts/quality/release_manifest_guard.py
python scripts/quality/quality_runtime_budget_guard.py
python scripts/quality/protected_scope_hash_guard.py
```

Esse guard confirma a presença e a coerência do inventário estrutural obrigatório da release, incluindo artefatos canônicos, diretórios operacionais e componentes de compatibilidade temporária já formalizados.

## Proteção do escopo congelado

O manifesto oficial de hashes do escopo protegido fica em `docs/releases/PROTECTED_SCOPE_HASHES.json` e deve ser validado por:

```bash
python scripts/quality/protected_scope_hash_guard.py
```

Esse contrato bloqueia alterações acidentais em `app.py` e `ui/review_view.py` sem atualização formal do manifesto.

## Sincronização obrigatória da release

Toda release oficial precisa manter sincronizados:

- `CHANGELOG.md`
- versão no rodapé do app
- documentos de governança, se afetados
- arquivo `.zip` final entregue

### Checklist curto de sincronização

- [ ] versão nova registrada no `CHANGELOG.md`
- [ ] versão nova refletida no rodapé
- [ ] `python scripts/quality/runtime_preflight.py` executado com sucesso
- [ ] `python scripts/quality/check_base.py` executado com sucesso
- [ ] `python scripts/validation/smoke_test_base.py` executado com sucesso
- [ ] `python -m compileall .` executado com sucesso
- [ ] `python scripts/quality/release_metadata_guard.py` executado com sucesso
- [ ] `python scripts/quality/compatibility_contract_guard.py` executado com sucesso
- [ ] `python scripts/quality/operational_checks_contract_guard.py` executado com sucesso
- [ ] `python scripts/quality/canonical_paths_reference_guard.py` executado com sucesso
- [ ] `python scripts/quality/script_cli_contract_guard.py` executado com sucesso
- [ ] `python scripts/quality/release_artifacts_hygiene_guard.py` executado com sucesso
- [ ] `python scripts/quality/runtime_dependencies_contract_guard.py` executado com sucesso
- [ ] `python scripts/quality/documentation_commands_examples_guard.py` executado com sucesso
- [ ] `python scripts/quality/release_manifest_guard.py`, `python scripts/quality/quality_runtime_budget_guard.py` e `python scripts/quality/protected_scope_hash_guard.py` executado com sucesso
- [ ] `python scripts/quality/release_guard.py` executado com sucesso
- [ ] `python scripts/quality/quality_gate.py` executado com sucesso
- [ ] `python scripts/reports/manual_validation_pack.py` executado
- [ ] `python scripts/reports/release_health_report.py` executado
- [ ] `CHECKLIST_REGRESSAO.md` seguido conforme o escopo
- [ ] `.zip` final limpo gerado

---

## Quando interromper uma release

Interromper imediatamente a release se ocorrer qualquer uma das situações abaixo:

- falha em `python scripts/quality/check_base.py`
- regressão funcional em fluxo estabilizado
- divergência entre rodapé e changelog
- mudança encostando em área congelada sem justificativa prática
- reorganização ampliando escopo além do planejado

Nesses casos, a release não deve ser fechada até a base voltar ao estado estável.

---

## Módulos oficiais a consultar antes de uma release

### Governança
- `CHANGELOG.md`
- `CHECKLIST_REGRESSAO.md`
- `docs/architecture/ARQUITETURA_BASE.md`
- `docs/operations/MANUTENCAO_OPERACIONAL.md`
- `docs/releases/RELEASE_OPERACIONAL.md`
- `docs/validation/VALIDACAO_MANUAL_GUIA.md`

### Validação
- `scripts/quality/runtime_preflight.py`
- `scripts/quality/check_base.py`
- `python -m compileall .`
- `scripts/quality/release_metadata_guard.py`
- `scripts/quality/compatibility_contract_guard.py`
- `scripts/quality/operational_checks_contract_guard.py`
- `scripts/quality/canonical_paths_reference_guard.py`
- `scripts/quality/script_cli_contract_guard.py`
- `scripts/quality/release_artifacts_hygiene_guard.py`
- `scripts/quality/runtime_dependencies_contract_guard.py`
- `scripts/quality/quality_gate.py`
- `scripts/reports/manual_validation_pack.py`
- `scripts/reports/release_health_report.py`

### Estado e fluxo
- `state/keys.py`
- `state/view_models.py`
- `state/session.py`
- `state/ui_state.py`
- `core/flow_guard.py`

### UI principal
- `ui/group_config_view.py`
- `ui/base_view.py`
- `ui/review_view.py`
- `ui/pre_sort_view.py`
- `ui/result_view.py`
- `ui/panels.py`
- `ui/primitives.py`

---

## Fechamento oficial da release

Uma release está oficialmente fechada quando:

1. a mudança está concluída e estável
2. `python scripts/quality/check_base.py` passou
3. o escopo foi validado pelo checklist funcional mínimo aplicável
4. `CHANGELOG.md` foi atualizado
5. o rodapé está coerente com a versão entregue
6. o `.zip` final limpo foi gerado

---

## Resumo operacional

### Antes
- classificar a mudança
- revisar governança
- rodar `python scripts/quality/check_base.py`

### Depois
- validar comportamento
- atualizar `CHANGELOG.md`
- sincronizar versão do rodapé
- gerar `.zip` limpo

### Regra final
Se houver dúvida entre liberar ou segurar, **segurar**. A base deve priorizar estabilidade e rastreabilidade.


## Orçamento operacional da rotina oficial

A rotina oficial de checks também deve preservar um **orçamento operacional** razoável. Essa verificação é formalizada por:

```bash
python scripts/quality/quality_runtime_budget_guard.py
python scripts/quality/protected_scope_hash_guard.py
```



python scripts/quality/script_exit_codes_contract_guard.py


Observação operacional: os scripts canônicos de governança devem manter **códigos de saída previsíveis** para sucesso e falha controlada.

## Crosslinks canônicos de governança

Este protocolo operacional deve permanecer conectado a:
- `docs/releases/BASELINE_OFICIAL.md`
- `docs/operations/OPERACAO_LOCAL.md`
- `docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md`
- `docs/validation/VALIDACAO_MANUAL_GUIA.md`

Guard leve desta coerência documental:

```bash
python scripts/quality/governance_docs_crosslinks_guard.py
```


Registro canônico dos checks:

python scripts/quality/checks_registry_contract_guard.py
python scripts/quality/checks_registry_schema_guard.py

fonte única de verdade: `scripts/quality/checks_registry.py`

Schema canônico do registro: `scripts/quality/checks_registry_schema_guard.py`
