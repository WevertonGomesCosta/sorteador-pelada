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

## Documento adicional de governança

- `docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md` — política oficial para wrappers, arquivos-ponte e critérios objetivos de remoção futura do legado temporário.

- `docs/releases/MAINTENANCE_MODE.md` — nota canônica curta que formaliza a entrada da baseline vigente em manutenção sob demanda e os critérios objetivos para reabrir trabalho.


## Compatibilidade

- `docs/compat/` — documentos-ponte legados isolados da árvore canônica.
- `scripts/compat/` — wrappers históricos isolados da árvore canônica de execução.
