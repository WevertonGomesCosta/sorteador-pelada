# Plano de reorganização faseada a partir da v124

> Plano preparatório. Não move lógica. Não altera o motor do app.

## Objetivo

Executar a reorganização estrutural em microetapas para evitar regressões como as observadas quando a refatoração altera simultaneamente:
- ordem de renderização;
- condições de exibição;
- ciclo de vida do `session_state`;
- scroll/foco.

## Fase A — preparação documental

Entregáveis:
- `docs/architecture/CONTRATO_RENDERIZACAO_UI.md`
- `docs/operations/MAPA_ESTADOS_FLUXO_UI.md`
- este plano faseado

Regra:
- nenhuma movimentação de lógica entre arquivos.

## Fase B — extração de helpers puros

Pode extrair:
- funções de texto;
- funções de resumo;
- formatação passiva.

Não pode extrair ainda:
- `st.form`
- `st.button`
- `st.rerun()`
- escrita em `st.session_state`
- gates de visibilidade.

## Fase C — componentes visuais passivos

Pode extrair:
- banners;
- badges;
- cartões de resumo;
- listas visuais sem efeito colateral.

Não pode alterar:
- ordem de montagem da revisão;
- lugar onde critérios, sorteio e resultado aparecem.

## Fase D — contrato de estado

Revisar e documentar, antes de modularizar:
- flags transitórias;
- estados legados (`FALTANTES_TEMP`, `NOVOS_JOGADORES`);
- limpeza obrigatória após confirmação/sorteio.

## Fase E — modularização controlada do renderer

Só começa depois das fases anteriores.

Regras:
- manter `app.py` como shell principal;
- manter `ui/review_view.py` como fachada estável enquanto a migração estiver em curso;
- copiar a ordem funcional da v124, sem reinterpretá-la.

## Critério para encerrar cada fase

Cada fase só avança se:
- a baseline funcional continuar equivalente à v124;
- `CHECKLIST_REGRESSAO.md` não apontar regressões;
- revisão, confirmação e sorteio continuarem operacionais.
