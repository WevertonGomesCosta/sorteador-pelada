# Organização da documentação

A estrutura canônica atual da documentação é o **padrão oficial de uso** desta baseline.

- `docs/architecture/` — arquitetura e desenho da base
- `docs/operations/` — manutenção, operação local e rotinas técnicas
- `docs/validation/` — smoke test, validação manual e registros UX
- `docs/releases/` — baseline oficial, protocolo de release e histórico operacional

Os arquivos históricos diretamente na raiz de `docs/` foram preservados apenas como **ponte temporária de compatibilidade**.

Ao criar ou atualizar documentação nova:
- use sempre os caminhos canônicos acima;
- evite criar novos documentos na raiz de `docs/`;
- trate os arquivos-ponte como legados temporários.
