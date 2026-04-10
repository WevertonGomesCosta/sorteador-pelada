# ARQUITETURA_BASE

## Objetivo deste documento

Este documento registra a arquitetura funcional atual do app **Sorteador Pelada PRO** após a reorganização estrutural da base. O objetivo é reduzir manutenção no arquivo errado, evitar reintrodução de duplicações e preservar os contratos já estabilizados do fluxo.

Este documento descreve **como a base está organizada hoje**, **onde cada responsabilidade vive** e **quais regras devem ser respeitadas** para evitar regressões.

---

## Princípios da base atual

A base atual segue estes princípios:

1. **`app.py` é o orquestrador principal**
   - coordena o fluxo da tela
   - lê o estado
   - chama funções da camada de UI
   - dispara a lógica de negócio quando necessário
   - não deve voltar a concentrar helpers grandes de fluxo, resultado ou renderização

2. **A lógica de negócio fica fora da UI**
   - sorteio, processamento de lista, validações e regras operacionais não devem ser implementados dentro dos componentes visuais

3. **A UI é modular por domínio**
   - base, revisão, resultado, configuração inicial e blocos auxiliares vivem em módulos separados

4. **O `session_state` crítico é centralizado**
   - chaves críticas ficam em `state/keys.py`
   - módulos principais devem preferir constantes em vez de strings soltas

5. **A interpretação visual do fluxo é separada da renderização**
   - leitura da etapa ativa, status da sessão e abertura dos blocos fica em `state/view_models.py`
   - componentes de UI apenas consomem essas decisões

---

## Visão geral do fluxo funcional

Fluxo principal do app:

1. **Configuração inicial**
   - escolher modo de uso
   - carregar base do grupo, Excel próprio ou seguir só com lista

2. **Entrada da lista**
   - usuário digita ou cola a lista
   - a revisão só é iniciada ao clicar em **“🔎 Revisar lista”**

3. **Revisão da lista**
   - inconsistências
   - faltantes
   - repetidos
   - bloqueios da base
   - correções inline

4. **Cadastro guiado/manual**
   - quando necessário, jogadores faltantes podem ser cadastrados

5. **Confirmação final**
   - a lista revisada é confirmada

6. **Sorteio**
   - aleatório por lista ou balanceado com base

7. **Resultado**
   - visualização dos times
   - copiar
   - compartilhar

---

## Mapa da arquitetura por pasta

### `app.py`

**Papel oficial:** orquestrador do fluxo.

Responsabilidades:
- montar a página
- aplicar estilos
- inicializar estado básico
- integrar lógica + UI + estado
- coordenar os blocos do fluxo
- controlar a passagem entre etapas

Não deve voltar a concentrar:
- helpers grandes de renderização
- helpers de resultado
- helpers de fluxo puro
- helpers grandes de estado visual

---

### `core/`

Camada de lógica e regras operacionais.

#### `core/logic.py`
Responsável por:
- processamento da lista
- diagnóstico de nomes
- sorteio balanceado
- regras principais do domínio da pelada

#### `core/optimizer.py`
Responsável por:
- otimização ou heurísticas ligadas à montagem de times

#### `core/validators.py`
Responsável por:
- validações especializadas da base e do fluxo

#### `core/flow_guard.py`
Responsável por:
- helpers de proteção do fluxo operacional
- assinatura da entrada do sorteio
- invalidação de resultado quando a entrada muda
- gate pré-sorteio
- extração de nomes únicos no modo aleatório
- contagem de duplicados da base atual

**Contrato:** UI não deve reimplementar regra de negócio já existente em `core/`.

---

### `data/`

#### `data/repository.py`
Responsável por:
- leitura/carregamento da base
- integração com arquivos externos da base de jogadores

**Contrato:** leitura e transformação inicial de dados devem preferir esta camada, e não `app.py`.

---

### `state/`

Camada de estado, chaves e interpretação visual.

#### `state/keys.py`
Responsável por:
- concentrar as chaves críticas do `session_state`

**Contrato:** módulos principais devem usar as constantes deste arquivo para estado crítico do fluxo.

#### `state/session.py`
Responsável por:
- inicialização do estado
- registro da base carregada no estado
- atualização da integridade da base no estado
- limpeza do estado de revisão
- diagnóstico da lista no estado

#### `state/ui_state.py`
Responsável por:
- pequenos helpers de estado local da interface
- abertura programática de expanders/blocos quando necessário

#### `state/view_models.py`
Responsável por:
- interpretar o estado visual do fluxo
- determinar etapa ativa
- construir status da sessão
- definir abertura/recolhimento dos blocos
- determinar visibilidade da revisão

**Contrato:**
- `view_models` não renderiza UI
- `view_models` não executa sorteio
- `view_models` não deve escrever em `session_state`
- deve permanecer como camada de funções puras de leitura/decisão visual

---

### `ui/`

Camada de apresentação do app.

#### `ui/primitives.py`
Responsável por:
- componentes visuais pequenos e reutilizáveis
- cabeçalhos de seção
- notas inline
- bloco “Sobre este app”

#### `ui/panels.py`
Responsável por:
- painéis informativos de status e CTA visual

#### `ui/actions.py`
Responsável por:
- botão de ação padronizado

#### `ui/group_config_view.py`
Responsável por:
- etapa inicial de configuração
- escolha de origem/base
- fluxo de configuração do grupo

#### `ui/base_view.py`
Responsável por:
- resumo da base
- alerta de integridade
- expander de inconsistências
- prévia da base

#### `ui/review_view.py`
Responsável por:
- revisão da lista
- painel de pendências
- correções inline
- correção de bloqueios da base
- correções de nomes e inconsistências

#### `ui/manual_card.py`
Responsável por:
- cartão/bloco de cadastro manual

#### `ui/pre_sort_view.py`
Responsável por:
- resumo operacional pré-sorteio

#### `ui/result_view.py`
Responsável por:
- cabeçalho do resultado
- painel de resumo do resultado
- cards dos times
- timestamp do sorteio
- texto de compartilhamento
- ações de copiar/compartilhar

#### `ui/summary_strings.py`
Responsável por:
- resumos textuais auxiliares
- textos curtos de expanders e critérios

#### `ui/components.py`
Responsável por:
- componentes JS auxiliares
- copiar
- compartilhar
- instalar app

#### `ui/styles.py`
Responsável por:
- CSS global do app

**Contrato geral da camada UI:**
- pode consumir `state/keys.py` e `state/view_models.py`
- não deve reimplementar regras centrais do sorteio
- lógica de domínio deve permanecer fora da camada visual

---

### `scripts/`

#### `scripts/check_base.py`
Responsável por:
- checagem estrutural mínima da base
- validação de presença de arquivos e funções-chave
- compilação sintática

**Contrato:** deve continuar leve, rápido e independente da execução interativa do Streamlit.

---

### Documentação operacional

#### `CHECKLIST_REGRESSAO.md`
Responsável por:
- checklist funcional mínimo antes de fechar nova iteração

#### `docs/ARQUITETURA_BASE.md`
Responsável por:
- registrar a arquitetura consolidada
- servir como referência de manutenção

---

## Contratos de manutenção por tema

### 1. Revisão da lista

Se a mudança envolver:
- botão “🔎 Revisar lista”
- pendências
- faltantes
- duplicados
- correções inline
- scroll para revisão/cadastro

Arquivos oficiais a considerar primeiro:
- `ui/review_view.py`
- `state/session.py`
- `state/keys.py`
- `state/view_models.py`
- `app.py` (apenas se a coordenação do fluxo realmente exigir)

**Evitar:** recriar lógica de revisão em `app.py`.

---

### 2. Base de jogadores

Se a mudança envolver:
- carregamento da base
- resumo da base
- inconsistências da base
- preview da base

Arquivos oficiais:
- `data/repository.py`
- `ui/base_view.py`
- `state/session.py`
- `core/validators.py`

---

### 3. Resultado final

Se a mudança envolver:
- texto do resultado
- layout dos times
- copiar/compartilhar
- timestamp

Arquivos oficiais:
- `ui/result_view.py`
- `ui/components.py`

**Evitar:** recolocar helpers de resultado em `app.py`.

---

### 4. Estado visual

Se a mudança envolver:
- etapa ativa
- próxima ação sugerida
- abertura/recolhimento de blocos
- painel de status

Arquivos oficiais:
- `state/view_models.py`
- `ui/panels.py`
- `app.py` apenas como consumidor das decisões visuais

**Evitar:** espalhar novamente a interpretação visual em vários pontos do `app.py`.

---

### 5. `session_state`

Se a mudança envolver:
- flags do fluxo
- scroll
- cadastro guiado
- confirmação
- resultado

Arquivos oficiais:
- `state/keys.py`
- `state/session.py`
- `state/ui_state.py`

**Regra:** se a chave for crítica para o fluxo, ela deve existir em `state/keys.py`.

---

## Fluxo oficial de dependências

Fluxo recomendado de leitura/manutenção:

- `app.py`
  - consome `core/`, `state/` e `ui/`
- `ui/`
  - pode consumir `state/keys.py`, `state/view_models.py` e `ui/` auxiliar
- `state/`
  - pode consumir `core/validators.py` quando necessário para diagnóstico
- `core/`
  - não deve depender de `ui/`

### Regra prática

- `core/` **não importa** `ui/`
- `view_models` **não renderiza** componentes
- `ui/` **não deve virar nova camada de regra de negócio**
- `app.py` **não deve voltar a concentrar helpers grandes**

---

## Pontos oficiais de manutenção

### Onde mexer primeiro

| Tema | Arquivo principal |
|---|---|
| Fluxo principal | `app.py` |
| Chaves do estado | `state/keys.py` |
| Estado de sessão | `state/session.py` |
| Estado visual | `state/view_models.py` |
| Configuração inicial | `ui/group_config_view.py` |
| Base carregada | `ui/base_view.py` |
| Revisão e pendências | `ui/review_view.py` |
| Cadastro manual | `ui/manual_card.py` |
| Pré-sorteio | `ui/pre_sort_view.py` |
| Resultado | `ui/result_view.py` |
| Painéis/status | `ui/panels.py` |
| Componentes simples | `ui/primitives.py`, `ui/actions.py` |
| Estilos | `ui/styles.py` |
| Validação estrutural | `scripts/check_base.py` |
| Regressão funcional | `CHECKLIST_REGRESSAO.md` |

---

## Regras para evitar regressão

1. **Não reabrir arquitetura macro sem necessidade explícita**
2. **Não mover lógica de domínio para componentes visuais**
3. **Não duplicar helper entre `app.py` e `ui/`**
4. **Não criar nova string crítica de `session_state` fora de `state/keys.py`**
5. **Toda mudança deve passar por:**
   - `python scripts/check_base.py`
   - `CHECKLIST_REGRESSAO.md`
6. **Mudanças em revisão/scroll/mobile devem ser pequenas e isoladas**
7. **Mudanças de UX não devem reabrir critérios do sorteio**

---

## Próxima regra operacional recomendada

Sempre que uma nova melhoria for proposta, responder primeiro a estas perguntas:

1. A mudança é de **lógica**, **estado**, **renderização** ou **texto**?
2. Qual é o **arquivo oficial** desse tema?
3. A mudança pode ser feita sem tocar em `app.py`?
4. Ela cria nova chave crítica de fluxo?
5. Ela precisa atualizar o checklist de regressão?

---

## Resumo executivo

A base atual está organizada em quatro camadas principais:
- **core**: regra de negócio
- **state**: estado, chaves e interpretação visual
- **ui**: apresentação por domínio
- **app.py**: orquestração

O contrato mais importante da base hoje é:

> **não reespalhar responsabilidade entre módulos que já foram reorganizados.**

O caminho seguro para evoluir o projeto é:
- manter `app.py` como orquestrador
- manter regras em `core/`
- manter chaves em `state/keys.py`
- manter leitura visual em `state/view_models.py`
- manter UI por domínio em `ui/`
- validar sempre com `scripts/check_base.py` e `CHECKLIST_REGRESSAO.md`



## Atualização de desacoplamento

- `core/base_summary.py` é o módulo neutro responsável por resumos reutilizados de inconsistências da base entre `core` e `ui`.


## Ajuste estrutural recente

- `state/criteria_state.py` passou a concentrar os critérios ativos e seus resumos neutros para evitar dependência direta de `core` em `ui`.


## Organização operacional complementar

Sem alterar o núcleo do app, a baseline atual organiza artefatos auxiliares em:
- `docs/architecture/`, `docs/operations/`, `docs/validation/`, `docs/releases/`
- `scripts/quality/`, `scripts/validation/`, `scripts/reports/`
- `tests/test_core_smoke.py`, `tests/test_state_smoke.py`, `tests/test_ui_safe_smoke.py`

Os caminhos históricos foram preservados como wrappers ou arquivos-ponte para compatibilidade operacional.
