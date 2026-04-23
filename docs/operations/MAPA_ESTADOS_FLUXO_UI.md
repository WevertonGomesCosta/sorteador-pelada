# Mapa de estados do fluxo de revisão, cadastro, confirmação e sorteio

> Documento de preparação da reorganização. Baseado na **v124** sem mover lógica entre arquivos.

## Objetivo

Registrar os estados mínimos que governam o fluxo da UI e o ciclo de vida esperado de cada um.

Este mapa existe para evitar três classes de regressão:
- reaparecimento de blocos fora de hora;
- vazamento de estado entre revisão e resultado;
- refatoração que muda exibição sem mudar motor, mas altera o comportamento do app.

## Grupo 1 — Estados de revisão da lista

### `DIAGNOSTICO_LISTA`

Tipo: `dict | None`

Responsabilidade:
- snapshot principal do diagnóstico da revisão atual;
- alimenta pendências, contadores, faltantes e bloqueios.

Contrato:
- `None` significa revisão ainda não materializada;
- quando preenchido, passa a sustentar a etapa 5.

### `LISTA_TEXTO_REVISADO`

Tipo: `str`

Responsabilidade:
- texto-base que serviu de referência para a revisão consolidada.

Contrato:
- usado para comparar se a confirmação ainda corresponde à entrada atual;
- não deve ficar divergente de `LISTA_REVISADA_CONFIRMADA`.

### `LISTA_REVISADA`

Tipo: `list | None`

Responsabilidade:
- lista canônica pronta para seguir ao sorteio.

Contrato:
- só é confiável quando a revisão foi concluída e a lista atual corresponde ao texto revisado;
- não deve ser usada como "lista confirmada" sozinha.

### `LISTA_REVISADA_CONFIRMADA`

Tipo: `bool`

Responsabilidade:
- gate oficial para liberar critérios e sorteio.

Contrato:
- deve ser `False` sempre que houver nova edição, novo cadastro, remoção ou nova revisão;
- só pode ser `True` quando a revisão atual estiver coerente com `LISTA_TEXTO_REVISADO` e `LISTA_REVISADA`.

### `REVISAO_LISTA_EXPANDIDA`

Tipo: `bool`

Responsabilidade:
- estado visual do expander principal de revisão.

Contrato:
- é estado de UI, não estado de negócio.

### `REVISAO_PENDENTE_POS_CADASTRO`

Tipo: `bool`

Responsabilidade:
- marca que houve ação de cadastro/remoção e a revisão precisa ser reapresentada em seguida.

Contrato:
- não deve sobreviver indefinidamente ao fluxo;
- é um estado transitório de continuidade visual.

## Grupo 2 — Estados do cadastro guiado

### `CADASTRO_GUIADO_ATIVO`

Tipo: `bool`

Responsabilidade:
- indica se o fluxo guiado de faltantes está em andamento.

Contrato:
- deve estar sincronizado com `FALTANTES_REVISAO`;
- quando `False`, o bloco de cadastro guiado não deve governar CTA nem scroll interno.

### `FALTANTES_REVISAO`

Tipo: `list[str]`

Responsabilidade:
- fila canônica de atletas faltantes da revisão atual.

Contrato:
- é a fonte primária da sequência do cadastro guiado;
- quando vazia, `CADASTRO_GUIADO_ATIVO` deve tender a `False`.

### `FALTANTES_CADASTRADOS_NA_RODADA`

Tipo: `list[str]`

Responsabilidade:
- memória da rodada atual de faltantes concluídos.

Contrato:
- ajuda a calcular progresso do cadastro guiado;
- não deve ser confundida com a base oficial persistida.

## Grupo 3 — Estados de scroll da revisão

### `SCROLL_PARA_REVISAO`
### `SCROLL_DESTINO_REVISAO`
### `SCROLL_ALVO_ID_REVISAO`

Responsabilidade conjunta:
- definir se o shell deve rolar para revisão e para qual alvo.

Contrato:
- pertencem ao **shell de `app.py`**;
- não devem ser reapropriados por submódulos da revisão sem nova auditoria;
- destinos válidos observados na v124: `top`, `pendencias`, `confirmar`.

## Grupo 4 — Estados de sorteio e resultado

### `RESULTADO`

Tipo: payload do sorteio ou equivalente truthy

Responsabilidade:
- habilita o bloco final do resultado.

Contrato:
- deve ser invalidado quando a entrada muda de forma material.

### `RESULTADO_CONTEXTO`
### `RESULTADO_ASSINATURA`
### `RESULTADO_INVALIDADO_MSG`
### `RESULTADOS_SESSAO_HISTORICO`
### `RESULTADO_HISTORICO_ATIVO_ID`
### `RESULTADO_HISTORICO_ULTIMO_SNAPSHOT_ID`

Responsabilidade conjunta:
- contexto, integridade e histórico do resultado da sessão.

Contrato:
- pertencem à fase pós-sorteio;
- não devem ser reutilizados para governar revisão ou cadastro.

### `SCROLL_PARA_SORTEIO`
### `SCROLL_PARA_RESULTADO`

Responsabilidade:
- deslocamentos explícitos para etapa de sorteio e bloco final de resultado.

Contrato:
- pertencem ao shell principal;
- não devem ser usados para corrigir fluxo do cadastro guiado.

## Grupo 5 — Estados legados que exigem isolamento

### `FALTANTES_TEMP`

Tipo: `list[str]`

Responsabilidade:
- fallback legado de cadastro fora do fluxo principal da revisão.

Contrato:
- enquanto estiver preenchido, bloqueia a exibição do resultado final;
- não deve ser confundido com `FALTANTES_REVISAO`;
- exige vigilância especial em qualquer reorganização, porque pode reabrir um bloco residual de cadastro.

### `NOVOS_JOGADORES`

Tipo: `list[dict]`

Responsabilidade:
- cadastros temporários usados no fallback legado.

Contrato:
- não deve vazar para o fluxo principal da revisão sem validação explícita.

### `AVISO_SEM_PLANILHA`

Tipo: `bool`

Responsabilidade:
- gate de fallback quando falta base pronta no fluxo balanceado.

Contrato:
- quando ativo, muda a rota da UI e interfere no resultado.

## Transições canônicas de alto nível

### A. Da lista para a revisão

Condição típica:
- `DIAGNOSTICO_LISTA` passa a existir
- `review_stage_visible == True`

### B. Da revisão para o cadastro guiado

Condição típica:
- existem `nao_encontrados` no diagnóstico
- `FALTANTES_REVISAO` é sincronizado
- `CADASTRO_GUIADO_ATIVO == True`

### C. Do cadastro guiado para a confirmação

Condição típica:
- `FALTANTES_REVISAO` fica vazio
- `CADASTRO_GUIADO_ATIVO` deixa de governar o fluxo
- `SCROLL_DESTINO_REVISAO = confirmar`

### D. Da confirmação para critérios e sorteio

Condição obrigatória:
- `LISTA_REVISADA_CONFIRMADA == True`
- `LISTA_REVISADA` coerente com a revisão atual

### E. Do sorteio para resultado

Condição típica:
- `RESULTADO` preenchido
- `FALTANTES_TEMP` vazio
- `AVISO_SEM_PLANILHA == False`

## Invariantes que a reorganização não pode quebrar

1. `CADASTRO_GUIADO_ATIVO` e `FALTANTES_REVISAO` devem permanecer coerentes.
2. `LISTA_REVISADA_CONFIRMADA` não pode sobreviver a novas edições/cadastros.
3. `FALTANTES_TEMP` deve continuar tratado como fluxo legado isolado.
4. `RESULTADO` e seu histórico não podem governar a revisão.
5. Scroll de revisão, sorteio e resultado continua sendo responsabilidade do shell.

## Checklist de preparação para refatoração

Antes de mover código entre arquivos, conferir:

- qual função escreve em cada uma das flags acima;
- quais flags são de negócio e quais são apenas de UI;
- quais flags são transitórias e devem ser limpas ao concluir uma etapa;
- se o bloco a extrair depende de `st.rerun()` ou de scroll explícito;
- se há risco de confundir `FALTANTES_REVISAO` com `FALTANTES_TEMP`.

## Uso recomendado deste documento

Este mapa deve ser lido junto com:
- `docs/architecture/CONTRATO_RENDERIZACAO_UI.md`
- `docs/releases/BASELINE_OFICIAL.md`
- `CHECKLIST_REGRESSAO.md`
