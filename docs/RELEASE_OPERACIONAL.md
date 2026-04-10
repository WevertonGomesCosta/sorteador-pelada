# RELEASE_OPERACIONAL

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
- resultado de `python scripts/check_base.py`

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
- `scripts/check_base.py`
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
   - `docs/ARQUITETURA_BASE.md`
   - `docs/MANUTENCAO_OPERACIONAL.md`
   - `CHANGELOG.md`
5. rodar:

```bash
python scripts/check_base.py
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
python scripts/check_base.py
```

2. executar a validação mínima manual conforme `CHECKLIST_REGRESSAO.md`
3. atualizar o `CHANGELOG.md`
4. sincronizar a versão exibida no rodapé do app
5. revisar a data da última atualização, quando a base depender de data explícita
6. garantir que o `.zip` final não contenha:
   - `__pycache__`
   - `.pyc`
   - arquivos transitórios de teste

---

## Sincronização obrigatória da release

Toda release oficial precisa manter sincronizados:

- `CHANGELOG.md`
- versão no rodapé do app
- documentos de governança, se afetados
- arquivo `.zip` final entregue

### Checklist curto de sincronização

- [ ] versão nova registrada no `CHANGELOG.md`
- [ ] versão nova refletida no rodapé
- [ ] `python scripts/check_base.py` executado com sucesso
- [ ] `python scripts/release_guard.py` executado com sucesso
- [ ] `CHECKLIST_REGRESSAO.md` seguido conforme o escopo
- [ ] `.zip` final limpo gerado

---

## Quando interromper uma release

Interromper imediatamente a release se ocorrer qualquer uma das situações abaixo:

- falha em `python scripts/check_base.py`
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
- `docs/ARQUITETURA_BASE.md`
- `docs/MANUTENCAO_OPERACIONAL.md`
- `docs/RELEASE_OPERACIONAL.md`

### Validação
- `scripts/check_base.py`

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
2. `python scripts/check_base.py` passou
3. o escopo foi validado pelo checklist funcional mínimo aplicável
4. `CHANGELOG.md` foi atualizado
5. o rodapé está coerente com a versão entregue
6. o `.zip` final limpo foi gerado

---

## Resumo operacional

### Antes
- classificar a mudança
- revisar governança
- rodar `python scripts/check_base.py`

### Depois
- validar comportamento
- atualizar `CHANGELOG.md`
- sincronizar versão do rodapé
- gerar `.zip` limpo

### Regra final
Se houver dúvida entre liberar ou segurar, **segurar**. A base deve priorizar estabilidade e rastreabilidade.
