## v128 — 2026-06-16

- preserva goleiros (`Posição = G`) no carregamento e limpeza da base, impedindo que atletas já cadastrados sejam removidos do `DF_BASE`;
- corrige o parser de listas numeradas para aceitar o formato `1 - Nome` sem incorporar o hífen ao nome do atleta;
- alinha os helpers da revisão ao mesmo parser numerado, mantendo correção, remoção e localização de ocorrências coerentes com a leitura principal;
- adiciona testes de regressão para preservação de goleiros na base e parsing de listas com espaço antes do hífen;
- fixa no rodapé a data controlada da release, substituindo a data dinâmica baseada no `mtime` local dos arquivos.

## v127 — 2026-04-23

- prepara a próxima microetapa da reorganização a partir da v126, sem alterar o motor do app;
- extrai helpers puros de `ui/review_view.py` para `ui/review_helpers.py`;
- extrai componentes visuais passivos de `ui/review_view.py` para `ui/review_passive_components.py`;
- preserva em `ui/review_view.py` a orquestração com `st.form`, `st.button`, `st.rerun` e escrita em `session_state`;
- mantém o comportamento funcional da baseline v124/v126 e reduz o acoplamento local da camada de revisão.

## v126 — 2026-04-23

- baseline mantida funcionalmente equivalente à v124; esta versão não move lógica entre arquivos nem altera o motor do app;
- adiciona documentação canônica das fronteiras de renderização da UI em `docs/architecture/CONTRATO_RENDERIZACAO_UI.md`;
- adiciona mapa dos estados de revisão, cadastro guiado, confirmação e sorteio em `docs/operations/MAPA_ESTADOS_FLUXO_UI.md`;
- adiciona plano faseado de reorganização segura em `docs/operations/PLANO_REORGANIZACAO_FASEADA_V124.md`, formalizando a sequência das próximas microetapas.

## v124 — 2026-04-23

- Limpeza da release com remoção de `__pycache__/`, `.pytest_cache/` e artefatos transitórios do pacote entregue.
- Reforço do `.gitignore` para caches Python, cobertura, logs e resíduos operacionais locais.
- Isolamento dos wrappers históricos em `scripts/compat/`, preservando os caminhos canônicos em `scripts/quality/`, `scripts/reports/` e `scripts/validation/` como padrão oficial.
- Isolamento dos documentos-ponte históricos em `docs/compat/`, mantendo `docs/architecture/`, `docs/operations/`, `docs/releases/` e `docs/validation/` como árvore canônica oficial.
- Atualização dos índices e guards canônicos para refletir a nova organização de compatibilidade sem alterar o motor do app.

## v123 — 2026-04-23

- move o bloco `Lista final sugerida` para baixo do fluxo de `Cadastre os nomes faltantes para seguir`, preservando o cadastro guiado acima da prévia final;
- adiciona a listagem explícita dos atletas faltantes logo abaixo do alerta `A revisão encontrou N nome(s) fora da base`;
- ao clicar em `Cadastrar faltantes agora`, passa a direcionar o scroll para o bloco `Cadastro guiado dos faltantes`, logo acima de `Lista final sugerida`.

## v122 — 2026-04-23

- move todo o cadastro do atleta atual para dentro de um único `st.form`, incluindo `Nome do atleta`, parâmetros e ações de salvar/remover;
- extrai o bloco `Cadastro guiado dos faltantes` para fora de `render_revisao_pendencias_panel(...)`, mantendo-o como bloco próprio logo após o resumo pré-sorteio da revisão;
- preserva apenas o redirecionamento final para `Confirmar lista final`, sem reintroduzir scroll intermediário do cadastro guiado.

## v121 — 2026-04-23

- saneamento arquitetural do scroll da revisão: removida do `app.py` toda lógica intermediária de destino `cadastro`/`cadastro_inline`;
- removida a preservação intermediária por `scrollY` absoluto entre faltantes;
- eliminadas âncoras duplicadas ligadas ao cadastro guiado, preservando apenas o redirecionamento final para `Confirmar lista final`;
- mantido o fluxo de entrada em `Revisão da lista` e o redirecionamento terminal para confirmação.

## v120 — 2026-04-23
- preserva a posição visual do bloco `Cadastro guiado dos faltantes` entre faltantes intermediários, sem scroll direcionado para a subseção;
- mantém o redirecionamento apenas no último faltante para `Confirmar lista final`.

## v119 — 2026-04-23
- remove o scroll direcionado ao entrar no `Cadastro guiado dos faltantes`;
- blinda `app.py` para neutralizar destinos residuais de scroll do cadastro guiado.

## v118 — 2026-04-23
- estabiliza o primeiro avanço entre faltantes pela reutilização do mesmo formulário guiado;
- mantém o redirecionamento terminal para `Confirmar lista final`.

## v117 — manter posição entre faltantes e rolar só no último

- Removido o scroll programático após **Salvar e próximo faltante** quando ainda restam nomes a tratar.
- Mantido o redirecionamento para **Confirmar lista final** apenas quando a fila de faltantes termina.

## v116 — 2026-04-23
- remove a subseção separada `Fora da base` da revisão da lista;
- coloca o `Cadastro guiado dos faltantes` diretamente abaixo do resumo pré-sorteio da seção 5;
- passa a tratar um atleta por vez no fluxo guiado, com edição do nome, preenchimento dos parâmetros e botões `Salvar e próximo faltante` e `Remover` no próprio bloco principal.


## v115 — 2026-04-23
- consolida a revisão de nomes fora da base em uma lista totalmente visível, sem expanders por item;
- mantém para cada nome os botões diretos de `Corrigir nome`, `Cadastrar` e `Remover`, com edição sempre exposta;
- move o cadastro guiado para aparecer inline no item atualmente selecionado, reduzindo a dependência de scroll para uma subseção distante.


## v114 — 2026-04-23
- altera a revisão de nomes fora da base para exibir todos os faltantes de uma vez, sem expanders individuais;
- mantém abaixo de cada nome os botões de `Corrigir nome`, `Cadastrar` e `Remover`, preservando o fluxo operacional da revisão;
- reduz a dependência de navegação item a item dentro da seção `Revisão da lista`, simplificando a triagem visual dos faltantes.


## v111
- corrige regressão por `AttributeError` ao clicar em **Revisar lista** quando a release executada ainda não expõe `SCROLL_ALVO_ID_REVISAO` e `SCROLL_DESTINO_REVISAO` em `state.keys`;
- adiciona shim de compatibilidade em `app.py` para manter o fluxo de revisão funcional mesmo em ambientes com módulo `state.keys` parcialmente desatualizado.
## v110 — 2026-04-23
Tipo: correção | ux

Resumo:
- Retomada dos scrolls internos da revisão sem reabrir o clique inicial de **Revisar lista** já estabilizado na v109.
- Introdução de alvo explícito de scroll interno para cadastro guiado, confirmação e retorno ao painel de pendências.
- Ajuste do scroll estabilizado para priorizar âncoras internas da revisão (`revisao-cadastro-atual-anchor`, `revisao-cadastro-form-anchor` e `revisao-confirmar-anchor`) em vez de retornar ao topo da seção.

Arquivos afetados:
- `app.py`
- `ui/review_view.py`
- `state/keys.py`
- `state/ui_state.py`

Validação:
- `python -m py_compile app.py ui/review_view.py state/keys.py state/ui_state.py`
- `pytest -q tests/test_ui_safe_smoke.py tests/test_smoke_base.py tests/test_core_smoke.py tests/test_state_smoke.py`
- `python scripts/quality/release_metadata_guard.py`
- `python scripts/quality/protected_scope_hash_guard.py`

## v109 — 2026-04-23
- restaura o comportamento de entrada na seção de revisão ao clicar em `Revisar lista`, reaproveitando a lógica de destino `pendencias/top` da baseline que já funcionava corretamente;
- usa o scroll simples original para os destinos `top` e `pendencias`, preservando o scroll reforçado apenas para `cadastro` e `confirmar`;
- remove a priorização de `primeiro_faltante` no clique inicial de revisão, evitando desvio do viewport antes do topo da seção `Revisão da lista`.

## v108
- reforça o scrollfix da revisão para levar ao primeiro nome fora da base ao clicar em revisar a lista
- reforça a abertura da seção Revisão da lista antes do alinhamento do scroll
- adiciona âncora específica do cadastro guiado no nome atual do faltante

## v107 — 2026-04-23

### Hotfix funcional — duplicados com qualificadores distintivos na revisão
- separa a chave de duplicidade da lista da chave geral de comparação, preservando qualificadores distintivos como `Douglas (pimpim)` e `Joel (convidado)`;
- impede que a revisão de duplicados agrupe ocorrências de nomes-base diferentes apenas por heurística de origem/correção, mantendo apenas duplicados reais;
- mantém intactos o fluxo de cadastro guiado, o scroll para `Etapa atual`, confirmação/sorteio e a lógica central do app.

## v106 — 2026-04-23

- versiona o formulário guiado de cadastro de faltantes por `indice_atual`;
