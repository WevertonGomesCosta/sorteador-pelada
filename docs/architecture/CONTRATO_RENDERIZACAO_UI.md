# Contrato de renderização da UI

> Documento de preparação da reorganização. Baseado na **v124** sem mover lógica entre arquivos.

## Objetivo

Registrar, antes de qualquer refatoração da UI, **quem renderiza o quê**, **em que ordem** e **quais blocos não podem mudar de dono** sem nova validação funcional.

A meta deste documento é reduzir regressões de:
- scroll e foco;
- reaparecimento indevido de blocos após confirmação/sorteio;
- vazamento de estado entre revisão, cadastro guiado, confirmação e resultado.

## Arquivos-fonte auditados

- `app.py`
- `ui/review_view.py`
- `state/keys.py`
- `state/ui_state.py`
- `state/view_models.py`

## Regra central de preservação

Na **v124**, a ordem de renderização é parte do comportamento funcional. Portanto:

1. **não** mover condicionais de exibição entre `app.py` e `ui/review_view.py` sem nova validação manual;
2. **não** trocar de lugar blocos de revisão, confirmação, critérios, sorteio e resultado na mesma etapa de reorganização;
3. **não** misturar extração de helpers com alteração de fluxo visual.

## Fronteiras de renderização

### 1. Shell principal da aplicação — `app.py`

`app.py` é o **orquestrador de alto nível**. Ele é dono de:

- inicialização global do `session_state`;
- leitura das entradas principais;
- decisão de visibilidade de etapas;
- shell das seções numeradas;
- scroll explícito de revisão, confirmação, sorteio e resultado;
- transição para o bloco final de resultado.

`app.py` **não deve perder** a responsabilidade por:
- `render_section_header(...)` das etapas principais;
- cálculo de `review_stage_visible`;
- cálculo de `etapa_visual_ativa`;
- gates para mostrar critérios/sorteio e resultado;
- scroll para `revisao-anchor`, `revisao-confirmar-anchor`, `sortear-anchor` e `resultado-anchor`.

### 2. Revisão da lista — `ui/review_view.py`

`ui/review_view.py` é dono do **conteúdo interno da etapa de revisão**, incluindo:

- resumo visual da revisão;
- pendências e correções inline;
- fluxo de cadastro guiado dos faltantes;
- prévia da lista final sugerida;
- botão de confirmação final.

`ui/review_view.py` **não deve assumir** a responsabilidade por:
- criar a seção numerada da revisão;
- decidir se a etapa de revisão existe ou não no fluxo;
- mostrar critérios/sorteio;
- mostrar resultado final.

### 3. View-model do shell — `state/view_models.py`

`state/view_models.py` centraliza somente decisões puras de interpretação do estado, sem renderização:

- `determinar_visibilidade_revisao(...)`
- `determinar_etapa_visual_ativa(...)`
- `construir_status_sessao_visual(...)`
- `construir_estado_blocos_visuais(...)`

Estas funções podem ser reorganizadas internamente, mas **não** devem começar a:
- escrever em `st.session_state`;
- chamar `st.rerun()`;
- renderizar componentes Streamlit.

### 4. Resultado do sorteio — `app.py`

O resultado final é renderizado diretamente em `app.py`. Isso inclui:

- âncora `resultado-anchor`;
- painel de sucesso;
- histórico de resultados da sessão;
- ações de copiar/compartilhar;
- cartões dos times;
- resumo detalhado do sorteio.

Na reorganização, este bloco deve continuar fora da revisão.

## Ordem canônica de exibição da UI na v124

A ordem funcional observada na baseline é:

1. configuração inicial / carga da base;
2. entrada da lista;
3. etapa 5 de revisão da lista, quando `review_stage_visible == True`;
4. critérios do sorteio, **somente** quando `lista_confirmada_ok == True`;
5. bloco de sorteio, **somente** quando `lista_confirmada_ok == True`;
6. fallback legado de `FALTANTES_TEMP`, **somente** quando esse fluxo específico estiver ativo;
7. resultado do sorteio, **somente** quando houver `RESULTADO` e não houver bloqueio por `AVISO_SEM_PLANILHA` nem `FALTANTES_TEMP`.

## Fronteiras internas da etapa 5 (revisão)

Dentro de `render_revisao_lista(...)`, a ordem canônica deve ser tratada como congelada para reorganização segura:

1. diagnóstico atual da revisão;
2. banner/estado da revisão;
3. resumo pré-sorteio e pendências;
4. aviso e lista explícita de faltantes, quando existirem;
5. botão `Cadastrar faltantes agora`;
6. bloco `Cadastro guiado dos faltantes`;
7. bloco `Lista final sugerida`;
8. bloco `Confirmar lista final`.

Qualquer reorganização futura deve preservar essa sequência enquanto a baseline v124 permanecer referência funcional.

## Blocos que não podem coexistir fora do contrato

### Cadastro guiado e confirmação final

Enquanto `CADASTRO_GUIADO_ATIVO == True`, o cadastro guiado pode coexistir com a revisão, mas **não** deve substituir a shell de revisão criada em `app.py`.

### Critérios/sorteio e confirmação pendente

Critérios e sorteio só entram quando:

- `LISTA_REVISADA_CONFIRMADA == True`
- `LISTA_REVISADA` está preenchida

### Resultado e faltantes temporários

O bloco de resultado só é exibido quando:

- `RESULTADO` existe
- `AVISO_SEM_PLANILHA == False`
- `FALTANTES_TEMP` está vazio

Isso significa que `FALTANTES_TEMP` é uma fronteira de renderização importante e não deve ser ignorada em futuras refatorações.

## Scroll: responsabilidades e limites

### Scroll que pertence ao shell (`app.py`)

`app.py` é o único dono dos scrolls explícitos para:
- revisão (`revisao-anchor`)
- pendências / primeiro faltante (`revisao-pendencias-anchor`, `revisao-primeiro-faltante-anchor`)
- confirmação final (`revisao-confirmar-anchor`)
- sorteio (`sortear-anchor`)
- resultado (`resultado-anchor`)

### Scroll que não deve voltar para a revisão

O fluxo intermediário do cadastro guiado entre faltantes **não** deve reintroduzir scroll programático novo sem reauditoria do contrato acima.

## Restrições de reorganização derivadas desta auditoria

Antes de quebrar `ui/review_view.py` em submódulos, manter:

- a shell das seções em `app.py`;
- a ordem canônica de exibição da revisão;
- a responsabilidade do shell por critérios, sorteio e resultado;
- a distinção entre fluxo principal de revisão e fallback legado `FALTANTES_TEMP`.

## Uso recomendado deste documento

Este documento deve ser lido junto com:
- `docs/operations/MAPA_ESTADOS_FLUXO_UI.md`
- `docs/releases/BASELINE_OFICIAL.md`
- `docs/validation/VALIDACAO_MANUAL_GUIA.md`
