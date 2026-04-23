# Organização da documentação

A estrutura canônica atual da documentação é o **padrão oficial de uso** desta baseline.

- `docs/architecture/` — arquitetura e desenho da base
- `docs/operations/` — manutenção, operação local, rotinas técnicas e política de compatibilidade temporária
- `docs/validation/` — smoke test, validação manual e registros UX
- `docs/releases/` — baseline oficial, protocolo de release, histórico operacional e modo de manutenção

Os arquivos históricos de documentação foram movidos para `docs/compat/` e permanecem apenas como **ponte temporária de compatibilidade**.

Ao criar ou atualizar documentação nova:
- use sempre os caminhos canônicos acima;
- evite criar novos documentos na raiz de `docs/`;
- trate os arquivos-ponte como legados temporários.


- `docs/architecture/CONTRATO_RENDERIZACAO_UI.md` — contrato canônico das fronteiras de renderização da UI, usado como preparação para refatoração segura da baseline v124.
- `docs/operations/MAPA_ESTADOS_FLUXO_UI.md` — mapa dos estados de revisão, cadastro guiado, confirmação e sorteio, com invariantes e ciclo de vida esperado.
- `docs/operations/PLANO_REORGANIZACAO_FASEADA_V124.md` — plano oficial das próximas microetapas de reorganização sem alterar o motor do app.

## Documento adicional de governança

- `docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md` — política oficial para wrappers, arquivos-ponte e critérios objetivos de remoção futura do legado temporário.

- `docs/releases/MAINTENANCE_MODE.md` — nota canônica curta que formaliza a entrada da baseline vigente em manutenção sob demanda e os critérios objetivos para reabrir trabalho.


## Compatibilidade

- `docs/compat/` — documentos-ponte legados isolados da árvore canônica.
- `scripts/compat/` — wrappers históricos isolados da árvore canônica de execução.
