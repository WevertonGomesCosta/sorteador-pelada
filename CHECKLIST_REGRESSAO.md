# Checklist fixo de regressão funcional

Use este checklist antes de considerar qualquer nova iteração da base como estável.

## 1. Entrada e configuração
- [ ] O app abre sem erro.
- [ ] A etapa inicial mostra corretamente as três opções: apenas lista, base do grupo e Excel próprio.
- [ ] O carregamento da base do grupo continua funcionando.
- [ ] O upload de Excel próprio continua funcionando.
- [ ] A limpeza/troca de base continua funcionando.

## 2. Lista e revisão manual
- [ ] O botão **🔎 Revisar lista** aparece logo abaixo da lista.
- [ ] No mobile, a revisão só acontece ao clicar no botão, sem depender de tocar fora do campo.
- [ ] A lista continua editável antes da revisão.
- [ ] Ao clicar em **🔎 Revisar lista**, o scroll leva ao ponto correto da revisão.

## 3. Pendências e correções
- [ ] Nome fora da base abre a opção **➕ Cadastrar na base**.
- [ ] O clique em **➕ Cadastrar na base** leva ao cadastro guiado abaixo.
- [ ] Inconsistências da base continuam abrindo a correção inline.
- [ ] Nomes duplicados na lista abrem correção adequada, sem unificação automática indevida.
- [ ] A lista de espera não entra como duplicidade da lista principal.
- [ ] Goleiros não entram como duplicidade da lista principal.

## 4. Revisão, confirmação e sorteio
- [ ] A revisão continua manual e só começa por clique no botão principal.
- [ ] A confirmação da lista final continua funcionando.
- [ ] O sorteio aleatório por lista continua funcionando.
- [ ] O sorteio com base carregada continua funcionando.
- [ ] O resumo pré-sorteio continua coerente com o estado atual.

## 5. Resultado
- [ ] O resultado aparece sem erro.
- [ ] O cabeçalho do resultado continua correto.
- [ ] O botão de copiar continua funcionando.
- [ ] O botão de compartilhar continua funcionando.

## 6. Regressão visual mínima
- [ ] O botão **🔎 Revisar lista** mantém o estilo visual neutro do app.
- [ ] Os expanders automáticos não escondem a etapa ativa.
- [ ] O painel de status continua renderizando sem HTML bruto.

## 7. Conferência técnica mínima
- [ ] `python scripts/check_base.py` retorna sucesso.
- [ ] `python -m compileall .` executa sem erro.
