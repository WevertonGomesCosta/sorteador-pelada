# BASELINE_OFICIAL

> Nesta baseline, os caminhos canônicos reorganizados são o padrão oficial de uso. Wrappers e arquivos-ponte históricos continuam disponíveis apenas como compatibilidade temporária.

## Versão oficial vigente

A baseline oficial vigente desta base é **v72**.

## Princípios de preservação

Esta baseline deve ser tratada como a referência única, fixa e estável do projeto para manutenção incremental.

Não reabrir sem necessidade concreta:
- arquitetura ampla
- lógica do sorteio
- regras da revisão
- fluxo estabilizado de scroll
- confirmação/sorteio
- áreas sensíveis já congeladas

## Estado consolidado da base

A base atual preserva:
- arquitetura modular por domínio
- governança documental e operacional
- gates estruturais e de release
- desacoplamento entre `core/flow_guard.py` e a camada de UI
- camada leve ampliada de smoke test funcional para módulos neutros e auxiliares seguros
- runner único de quality gate técnico e pré-checagem de runtime local
- gerador padronizado de registro para validação manual no navegador
- reorganização operacional leve da documentação, scripts e testes com compatibilidade preservada

## Área sensível congelada

`ui/review_view.py` permanece como área sensível.

Mudanças nesse módulo só devem ocorrer se houver necessidade concreta, localizada e justificada por regressão ou bloqueio real de uso.

## Validação mínima obrigatória

Antes de fechar uma nova iteração ou release oficial, executar:

```bash
python scripts/quality/runtime_preflight.py
python scripts/quality/check_base.py
python scripts/validation/smoke_test_base.py
python scripts/quality/release_metadata_guard.py
python scripts/quality/release_guard.py
python scripts/quality/quality_gate.py
python scripts/reports/manual_validation_pack.py
```

Wrappers e arquivos-ponte históricos continuam válidos apenas como compatibilidade temporária.

A política oficial dessa compatibilidade está formalizada em `docs/operations/POLITICA_COMPATIBILIDADE_TEMPORARIA.md`.
