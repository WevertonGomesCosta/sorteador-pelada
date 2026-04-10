# POLITICA_COMPATIBILIDADE_TEMPORARIA

> Nesta baseline, os caminhos canônicos reorganizados são o padrão oficial de uso. Wrappers em `scripts/`, arquivos-ponte na raiz de `docs/` e o agregador histórico `tests/test_smoke_base.py` permanecem ativos apenas como compatibilidade temporária controlada.

## Objetivo

Formalizar a política de convivência entre:
- a estrutura canônica atual do repositório;
- os caminhos históricos preservados para transição;
- os critérios objetivos para futura remoção do legado temporário.

Esta política existe para evitar dois riscos opostos:
1. limpar cedo demais e quebrar rotina operacional consolidada;
2. manter compatibilidade legada indefinidamente sem critério explícito.

## Escopo da compatibilidade temporária

Nesta fase, os seguintes elementos são tratados como compatibilidade temporária:

### 1. Wrappers em `scripts/`
Arquivos históricos na raiz de `scripts/` que apenas encaminham a execução para os caminhos canônicos em:
- `scripts/quality/`
- `scripts/validation/`
- `scripts/reports/`

### 2. Arquivos-ponte na raiz de `docs/`
Arquivos históricos em `docs/` que apontam para a documentação canônica em:
- `docs/architecture/`
- `docs/operations/`
- `docs/validation/`
- `docs/releases/`

### 3. Agregador histórico de smoke test
- `tests/test_smoke_base.py`

Ele continua permitido apenas como ponto de compatibilidade para transição, e não como centro oficial de evolução da suíte.

## Regra operacional vigente

A partir desta baseline:
- **os caminhos canônicos são o padrão oficial de uso**;
- **novos arquivos não devem nascer nos caminhos históricos**;
- **qualquer nova documentação deve citar primeiro os caminhos canônicos**;
- **qualquer novo comando oficial deve usar os caminhos canônicos**.

Compatibilidade temporária não é equivalência de status. Ela existe apenas para transição segura.

O guard canônico desta política é:

```bash
python scripts/quality/compatibility_contract_guard.py
```

## O que é proibido enquanto a compatibilidade existir

Durante a fase de transição, não fazer:
- criação de novos documentos permanentes na raiz de `docs/`;
- criação de novos scripts oficiais na raiz de `scripts/` quando já houver subpasta canônica adequada;
- ampliação funcional do legado temporário;
- dependência nova do agregador `tests/test_smoke_base.py` como centro da suíte;
- remoção dos wrappers/arquivos-ponte sem passar pelos critérios desta política.

## Critérios objetivos para futura remoção

A limpeza controlada dos wrappers e arquivos-ponte só pode ser aberta quando **todos** os critérios abaixo forem atendidos:

1. **Janela mínima de estabilidade**  
   Existirem pelo menos **2 releases oficiais estáveis completas após a v70** sem necessidade de restaurar caminhos históricos como padrão.

2. **Uso oficial totalmente migrado**  
   README, documentação operacional e exemplos de comando devem usar apenas caminhos canônicos como referência principal.

3. **Ausência de expansão do legado**  
   Nenhum novo arquivo oficial deve ter sido criado nos caminhos históricos desde a consolidação da v70.

4. **Gates preparados para a remoção**  
   `scripts/quality/check_base.py`, `scripts/quality/release_metadata_guard.py` e `scripts/quality/release_guard.py` devem conseguir ser ajustados para um modo canônico sem abrir regressão estrutural.

5. **Validação operacional concluída**  
   Os comandos canônicos devem ter sido usados normalmente nas rotinas locais e nas releases subsequentes, sem depender dos wrappers como caminho principal.

6. **Escopo de limpeza isolado**  
   A remoção do legado deve ser a única frente estrutural da release, sem misturar correção funcional sensível, UX em área congelada ou reorganização ampla paralela.

## Critérios de bloqueio para NÃO remover

Mesmo no futuro, a limpeza deve ser adiada se houver qualquer um destes sinais:
- dúvidas operacionais reais sobre os caminhos canônicos;
- documentação ainda inconsistente entre caminhos antigos e novos;
- wrappers ainda sendo necessários como orientação principal para uso local;
- outra mudança estrutural em andamento na mesma release;
- qualquer alteração prevista em `app.py`, `ui/review_view.py`, confirmação/sorteio ou lógica central.

## Ritual obrigatório quando a limpeza futura for aberta

Quando os critérios forem atingidos, a limpeza controlada deve seguir este protocolo:

1. abrir uma release dedicada apenas ao legado temporário;
2. remover primeiro as referências documentais residuais;
3. ajustar `check_base.py`, `release_metadata_guard.py`, `compatibility_contract_guard.py`, `release_guard.py` e `quality_gate.py` para o modo estritamente canônico;
4. só depois remover wrappers e arquivos-ponte;
5. executar:

```bash
python scripts/quality/runtime_preflight.py
python scripts/quality/check_base.py
python scripts/validation/smoke_test_base.py
python scripts/quality/release_metadata_guard.py
python scripts/quality/compatibility_contract_guard.py
python scripts/quality/operational_checks_contract_guard.py
python scripts/quality/canonical_paths_reference_guard.py
python scripts/quality/release_guard.py
python scripts/quality/quality_gate.py
python scripts/reports/manual_validation_pack.py
python scripts/reports/release_health_report.py
```

6. registrar explicitamente no `CHANGELOG.md` que a remoção do legado temporário ocorreu sob esta política.

## Situação desta baseline

Na baseline atual, a decisão oficial é:
- manter compatibilidade histórica temporária;
- usar somente caminhos canônicos como padrão oficial;
- adiar a limpeza até cumprir integralmente os critérios acima.
