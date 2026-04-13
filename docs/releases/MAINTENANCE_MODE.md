# MAINTENANCE_MODE

## Status operacional

Este documento registra que a **v96** marcou a entrada da base em **manutenção sob demanda** para a frente estrutural e operacional leve.

A frente de endurecimento estrutural leve e a frente de conveniências operacionais estão consideradas **encerradas** nesta versão.

## Escopo congelado

Permanecem congelados e não devem ser reabertos sem necessidade concreta:
- `app.py`
- `ui/review_view.py`
- confirmação/sorteio
- lógica central do app
- contratos de compatibilidade temporária já formalizados

## Comandos operacionais oficiais

Para uso, revisão, handoff e retomada, os comandos operacionais oficiais passam a ser:

```bash
python scripts/reports/maintenance_refresh_bundle.py
python scripts/reports/maintenance_reports_index.py
python scripts/reports/maintenance_reports_cleanup.py
```

Uso recomendado:
- `maintenance_refresh_bundle.py` para regenerar, em ordem canônica, os artefatos operacionais de manutenção;
- `maintenance_reports_index.py` para localizar rapidamente os artefatos mais recentes em `reports/`;
- `maintenance_reports_cleanup.py` para higienizar `reports/` antes de empacotamento ou fechamento de rodada.

## Critérios objetivos para reabrir trabalho

Só reabrir trabalho fora da manutenção sob demanda quando houver pelo menos uma destas condições:
- bug funcional real e reproduzível;
- deriva operacional recorrente em uso local ou no fluxo oficial de manutenção;
- necessidade clara de revisão, handoff ou retomada não coberta pelos utilitários já existentes na v96/v97;
- exigência documental concreta para suportar uma nova release oficial.

## Regra de prudência

Não adicionar novo guard, novo utilitário ou nova camada de governança sem ganho operacional concreto, direto e verificável.

Na ausência desses critérios, a conduta padrão deve ser:
- preservar a v96 como marco estrutural estável e a v97 como baseline funcional controlada;
- operar com os comandos oficiais já consolidados;
- aplicar apenas manutenção pontual sob demanda.
