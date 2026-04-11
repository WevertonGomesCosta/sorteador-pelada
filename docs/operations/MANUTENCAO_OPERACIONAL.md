# MANUTENCAO_OPERACIONAL

## Objetivo deste documento

Este documento formaliza o **protocolo oficial de manutenção** da base estável do app **Sorteador Pelada PRO**.

O objetivo é reduzir regressões, evitar mudanças no módulo errado e garantir que qualquer nova iteração respeite a arquitetura consolidada, o checklist funcional e o endurecimento técnico já implementado.

Este documento deve ser lido em conjunto com:

- `docs/architecture/ARQUITETURA_BASE.md`
- `docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md`
- `CHECKLIST_REGRESSAO.md`
- `scripts/quality/check_base.py`

---

## Regra de ouro da base

A base atual deve ser tratada como **funcionalmente estável**.

Isso significa que:

1. a arquitetura atual **não deve ser reaberta** sem necessidade explícita;
2. a lógica central de sorteio **não deve ser alterada** em ajustes de UX ou reorganização;
3. revisão, confirmação, resultado, scroll e fluxo mobile só podem ser alterados de forma localizada e com validação completa;
4. qualquer reorganização estrutural deve ser incremental, documentada e acompanhada por validação técnica e funcional;
5. mudanças devem sempre partir do **módulo oficial da responsabilidade**, e não do arquivo mais “fácil de encontrar”.

---

## O que está congelado nesta base

Os seguintes comportamentos devem ser considerados **congelados** até necessidade explícita:

- revisão da lista iniciada por clique em **“🔎 Revisar lista”**;
- botão de revisão abaixo da lista, em estilo visual neutro;
- correção de inconsistências, faltantes, repetidos e bloqueios dentro do fluxo de revisão;
- tratamento da lista de espera fora do escopo da lista principal;
- confirmação final antes do sorteio;
- resultado final com copiar e compartilhar;
- arquitetura reorganizada com `app.py` como orquestrador;
- chaves críticas do `session_state` centralizadas em `state/keys.py`;
- interpretação visual centralizada em `state/view_models.py`.

---

## Rituais obrigatórios antes de editar

Antes de qualquer mudança, executar esta sequência:

### 1. Classificar o tipo de mudança

Toda mudança deve ser enquadrada em uma destas classes:

- **Correção localizada**
- **Melhoria de UX incremental**
- **Reorganização estrutural**
- **Endurecimento técnico/documental**

Se a mudança não se encaixar claramente em uma dessas classes, ela não deve ser iniciada sem redefinir o escopo.

### 2. Identificar o módulo oficial da mudança

Antes de editar, responder: **em qual módulo essa responsabilidade vive oficialmente hoje?**

Nunca começar a edição em `app.py` por conveniência se o domínio já tiver módulo próprio.

### 3. Conferir a política de compatibilidade temporária

Antes de qualquer limpeza de wrappers, arquivos-ponte ou caminhos históricos, revisar:

- `docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md`

Se a mudança pretendida for apenas remoção de legado temporário, ela só pode começar quando todos os critérios objetivos da política estiverem atendidos.

### 4. Rodar a checagem técnica mínima

Executar:

```bash
python scripts/quality/check_base.py
python scripts/reports/maintenance_snapshot_report.py
```

O objetivo é confirmar que a base está íntegra antes da alteração e registrar um snapshot operacional somente leitura da baseline antes do trabalho.

### 5. Revisar a arquitetura e o checklist

Consultar:

- `docs/architecture/ARQUITETURA_BASE.md`
- `CHECKLIST_REGRESSAO.md`

Isso evita editar módulo errado ou quebrar fluxo já estabilizado.

---

## Rituais obrigatórios depois de editar

Depois de qualquer alteração, executar esta sequência:

### 1. Rodar novamente a checagem técnica mínima

```bash
python scripts/quality/check_base.py
python scripts/reports/maintenance_snapshot_report.py
```

### 2. Rodar o checklist funcional mínimo

Usar `CHECKLIST_REGRESSAO.md` e validar pelo menos:

- entrada e configuração;
- botão **🔎 Revisar lista** e revisão manual por clique;
- correção de inconsistências, faltantes e repetidos;
- scroll para pendências e cadastro guiado;
- confirmação final;
- sorteio;
- resultado, copiar e compartilhar.

### 3. Atualizar metadados visíveis quando aplicável

Quando houver mudança relevante entregue ao usuário, revisar se precisa atualizar:

- versão da base exibida no rodapé;
- bloco “Sobre este app”;
- documentação técnica correspondente.

### 4. Registrar a mudança no módulo certo

Se a alteração introduzir nova responsabilidade, documentar no local certo:

- arquitetura → `docs/architecture/ARQUITETURA_BASE.md`
- protocolo de manutenção → `docs/operations/MANUTENCAO_OPERACIONAL.md`
- nova checagem estrutural → `scripts/quality/check_base.py`

---

## Módulos oficiais por tipo de problema

### Fluxo principal / orquestração
- `app.py`

Usar apenas para:
- encadear etapas do fluxo;
- integrar estado, lógica e UI;
- coordenar a ordem de renderização.

Não usar para reintroduzir helpers grandes já extraídos.

### Lógica de sorteio e guardas de fluxo
- `core/logic.py`
- `core/flow_guard.py`
- `core/optimizer.py`
- `core/validators.py`

### Dados e base
- `data/repository.py`
- `ui/base_view.py`

### Revisão da lista e correções
- `ui/review_view.py`

### Configuração inicial / origem da base
- `ui/group_config_view.py`

### Resultado final
- `ui/result_view.py`

### Estado visual e próxima ação
- `state/view_models.py`

### Chaves críticas do `session_state`
- `state/keys.py`

### Inicialização e helpers de estado
- `state/session.py`
- `state/ui_state.py`

### Componentes pequenos e textos auxiliares
- `ui/primitives.py`
- `ui/panels.py`
- `ui/actions.py`
- `ui/summary_strings.py`

---

## Tipos de mudança permitidos

### 1. Correção localizada

**Exemplo:** corrigir um scroll específico, um texto errado ou um caso de duplicidade.

**Escopo esperado:**
- alteração pequena;
- no módulo oficial do problema;
- sem reorganização paralela.

**Validação mínima:**
- `python scripts/quality/check_base.py`
- itens do `CHECKLIST_REGRESSAO.md` relacionados ao fluxo afetado.

---

### 2. Melhoria de UX incremental

**Exemplo:** reduzir scroll, ajustar microcopy, recolher bloco, lapidar rodapé.

**Escopo esperado:**
- sem mexer na lógica do sorteio;
- sem alterar arquitetura;
- sem mexer simultaneamente em revisão, scroll e mobile se não for indispensável.

**Validação mínima:**
- `python scripts/quality/check_base.py`
- checklist funcional do fluxo correspondente;
- conferência visual no navegador e, quando relevante, no mobile.

---

### 3. Reorganização estrutural

**Exemplo:** mover função entre módulos, criar novo módulo de responsabilidade, limpar wrappers.

**Escopo esperado:**
- sem alteração funcional intencional;
- por etapas pequenas;
- com ownership claro;
- com documentação atualizada.

**Validação mínima:**
- `python scripts/quality/check_base.py`
- compilação e verificação dos módulos afetados;
- checklist funcional mínimo da base.

---

### 4. Endurecimento técnico/documental

**Exemplo:** ampliar `check_base.py`, criar documentação, centralizar constantes, criar `view_models`.

**Escopo esperado:**
- sem impacto no comportamento funcional;
- aumento de auditabilidade e segurança da manutenção.

**Validação mínima:**
- `python scripts/quality/check_base.py`
- conferência de integridade da documentação/artefatos criados.

---

## O que evitar

Evitar explicitamente:

- recriar helpers extraídos dentro de `app.py`;
- reintroduzir `ui/sections.py` ou wrappers equivalentes;
- duplicar função em mais de um módulo;
- misturar reorganização estrutural com correção funcional grande no mesmo passo;
- mexer simultaneamente em revisão, scroll e mobile sem escopo muito claro;
- usar strings soltas para chaves críticas do `session_state` quando já existe constante em `state/keys.py`;
- colocar lógica de domínio dentro de componente visual;
- alterar o fluxo do sorteio em nome de melhoria visual.

---

## Sequência oficial para uma nova iteração

A sequência oficial de manutenção da base é:

1. classificar a mudança;
2. localizar o módulo oficial;
3. rodar `python scripts/quality/check_base.py`;
4. fazer a alteração local;
5. rodar novamente `python scripts/quality/check_base.py`;
6. executar o `CHECKLIST_REGRESSAO.md` compatível com a mudança;
7. atualizar documentação/metadados se necessário;
8. só então considerar a iteração concluída.

---

## Critério para interromper uma mudança

A alteração deve ser interrompida e reenquadrada quando ocorrer qualquer uma destas situações:

- necessidade de editar mais de um domínio sem planejamento claro;
- surgimento de regressão em revisão, scroll, mobile ou resultado;
- dúvida sobre qual é o módulo oficial da responsabilidade;
- necessidade de reabrir arquitetura para resolver um problema pequeno;
- crescimento da mudança para além do escopo original.

Nesses casos, a regra é:
- parar;
- reclassificar a mudança;
- redefinir o escopo antes de continuar.

---

## Resumo operacional

A manutenção oficial da base deve seguir esta lógica:

- **mudar pouco por vez**;
- **mudar no módulo certo**;
- **validar antes e depois**;
- **não reabrir o que já está congelado sem necessidade explícita**;
- **usar a arquitetura e o checklist como contrato real, e não como documentação decorativa**.


## Observação adicional de manutenção

- Critérios ativos e seus resumos neutros devem permanecer em `state/criteria_state.py`; `core/*` não deve voltar a importar `ui.summary_strings`.
