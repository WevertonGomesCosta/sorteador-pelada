# OPERACAO_LOCAL

> Nesta baseline, os caminhos canônicos reorganizados são o padrão oficial de uso. Wrappers e arquivos-ponte históricos continuam disponíveis apenas como compatibilidade temporária.

## Objetivo

Padronizar a execução local mínima da base para:
- validar dependências do ambiente;
- abrir o app com segurança;
- executar os checks técnicos oficiais;
- registrar a validação manual no navegador.

## Fluxo recomendado

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Fazer a pré-checagem do ambiente

```bash
python scripts/quality/runtime_preflight.py
```

Essa etapa confirma, de forma leve, que:
- os arquivos essenciais da base estão presentes;
- as dependências declaradas estão importáveis;
- o ambiente está apto para abrir o app.

### 3. Rodar o quality gate técnico

```bash
python scripts/quality/quality_gate.py
```

Validação leve dos metadados de release, quando a rodada alterar versão, changelog, baseline ou rodapé:

```bash
python scripts/quality/release_metadata_guard.py
```

Verificação do contrato de compatibilidade temporária durante a fase de transição estável:

```bash
python scripts/quality/compatibility_contract_guard.py
```

Verificação de que README, documentos operacionais e wrappers continuam promovendo os caminhos canônicos como padrão oficial:

```bash
python scripts/quality/canonical_paths_reference_guard.py
```

Esse runner executa, em sequência:
- `python scripts/quality/check_base.py`
- `python scripts/validation/smoke_test_base.py`
- `python -m compileall .`
- `python scripts/quality/release_metadata_guard.py`
- `python scripts/quality/compatibility_contract_guard.py`
- `python scripts/quality/operational_checks_contract_guard.py`
- `python scripts/quality/canonical_paths_reference_guard.py`
- `python scripts/quality/release_guard.py`

### 4. Gerar o relatório-base da validação manual

```bash
python scripts/reports/manual_validation_pack.py
python scripts/reports/release_health_report.py
```

Esse comando cria um arquivo em `reports/` com o checklist completo e um bloco padronizado para registrar falhas reproduzidas.

## Execução do app

Com o ambiente pronto:

```bash
streamlit run app.py
```

## Validação manual final

Depois da abertura do app, executar o checklist usando o relatório gerado em `reports/`.

Referências da rodada:
- `CHECKLIST_REGRESSAO.md`
- `docs/validation/VALIDACAO_MANUAL_GUIA.md`

Registrar apenas falhas reproduzíveis, com:
- item do checklist;
- passos executados;
- observado;
- esperado;
- frequência;
- ambiente;
- evidência visual, quando houver.

## Restrições operacionais

Durante a validação local:
- não reabrir arquitetura ampla;
- não mexer em `ui/review_view.py` sem necessidade concreta;
- não alterar confirmação/sorteio sem defeito comprovado;
- preservar a baseline oficial vigente.


## Nota sobre organização

Os caminhos canônicos em `scripts/quality/`, `scripts/validation/` e `scripts/reports/` são o padrão oficial.
Os comandos históricos em `scripts/` continuam válidos apenas como compatibilidade temporária.
