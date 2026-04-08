# ⚽ Sorteador Pelada PRO

Aplicação web desenvolvida em **Streamlit** para sorteio de times de futebol de forma mais equilibrada, utilizando notas e atributos dos jogadores.

O app permite carregar uma base de jogadores, colar a lista de participantes do dia e gerar times balanceados com base em critérios como:

- **Nota**
- **Posição**
- **Velocidade**
- **Movimentação**

Também oferece suporte para:

- carregamento de base original da pelada via planilha
- upload de planilha própria em Excel
- cadastro manual de jogadores
- correção de nomes digitados com base na planilha
- cópia rápida do resultado
- instruções de instalação do app no celular ou desktop

---

## 🚀 Acesso ao app

App publicado:

**https://sorteador-pelada.streamlit.app/**

Repositório:

**https://github.com/WevertonGomesCosta/sorteador-pelada**

---

## 🎯 Objetivo

O projeto foi criado para facilitar o sorteio de times em peladas, reduzindo desequilíbrios causados por sorteios puramente aleatórios.

Em vez de apenas distribuir nomes, o app utiliza uma lógica de otimização para montar times mais equilibrados a partir dos atributos cadastrados de cada jogador.

---

## 🧠 Como funciona

O fluxo principal do app é:

1. carregar uma base de jogadores
2. colar a lista dos jogadores que vão participar no dia
3. identificar jogadores já existentes na base
4. corrigir nomes digitados com variações simples
5. cadastrar manualmente jogadores faltantes, se necessário
6. definir o número de times e os critérios de balanceamento
7. gerar os times otimizados
8. copiar o resultado para compartilhamento

A otimização é feita com `PuLP`, buscando distribuir os jogadores entre os times de forma equilibrada conforme os critérios selecionados.

---

## 📂 Estrutura atual dos dados

A base de jogadores utiliza as seguintes colunas:

- `Nome`
- `Nota`
- `Posição`
- `Velocidade`
- `Movimentação`

### Exemplo de estrutura esperada

| Nome              | Nota | Posição | Velocidade | Movimentação |
|-------------------|------|---------|------------|--------------|
| João Silva        | 8.0  | A       | 4          | 4            |
| Pedro Souza       | 6.5  | M       | 3          | 3            |
| Carlos Oliveira   | 7.0  | D       | 2          | 2            |

### Convenções usadas
- `A` = Atacante
- `M` = Meio
- `D` = Defesa
- `G` = Goleiro

> Observação: no fluxo atual, jogadores com posição `G` são removidos da base utilizada no sorteio principal.

---

## ✅ Funcionalidades atuais

- Interface web simples e rápida em Streamlit
- Modo administrador com acesso à base original
- Upload de planilha `.xlsx`
- Download de modelo de planilha
- Download da base atual
- Cadastro manual de jogadores
- Identificação de nomes duplicados
- Correção de nomes ignorando acentos e diferenças de caixa
- Sorteio equilibrado por otimização
- Cálculo de odds por time
- Cópia formatada do resultado
- Botão de instalação com instruções por navegador/dispositivo

---

## 🔐 Modo administrador

O app possui um modo administrador para acesso à base principal da pelada.

As credenciais são lidas via `st.secrets`:

- `nome_admin`
- `senha_admin`

Caso não estejam configuradas, o app utiliza valores padrão locais apenas como fallback.

---

## 📥 Formas de carregar dados

Atualmente o app aceita três caminhos principais:

### 1. Base original da pelada

Carregada a partir de uma planilha vinculada à URL padrão configurada no código.

### 2. Upload de planilha própria

O usuário pode enviar um arquivo Excel `.xlsx` com a estrutura esperada.

### 3. Cadastro manual

Também é possível cadastrar jogadores manualmente direto na interface.

---

## ⚙️ Critérios de balanceamento

Durante o sorteio, o usuário pode ativar ou desativar os seguintes critérios:

* Equilibrar **Posição**
* Equilibrar **Nota**
* Equilibrar **Velocidade**
* Equilibrar **Movimentação**

Isso permite ajustar o sorteio conforme o perfil da pelada e a qualidade da base cadastrada.

---

## 🛠️ Tecnologias utilizadas

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **PuLP**
* **XlsxWriter**
* **OpenPyXL**

---

## 📦 Instalação local

### 1. Clonar o repositório

```bash
git clone https://github.com/WevertonGomesCosta/sorteador-pelada.git
cd sorteador-pelada
```

### 2. Criar ambiente virtual

```bash
python -m venv .venv
```

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux/macOS

```bash
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Executar o app

```bash
streamlit run app.py
```

---


## 🧪 Validação mínima da base

Antes de fechar uma nova iteração da aplicação, use:

```bash
python scripts/check_base.py
```

Além disso, siga o checklist funcional em:

- `CHECKLIST_REGRESSAO.md`

E consulte a documentação de governança e manutenção em:

- `docs/ARQUITETURA_BASE.md`
- `docs/MANUTENCAO_OPERACIONAL.md`

Esse fluxo ajuda a detectar regressões estruturais e a manter a base estável antes de novas mudanças.

---

## ☁️ Deploy

O projeto pode ser publicado facilmente no **Streamlit Community Cloud**.

### Requisitos básicos

* repositório no GitHub
* branch publicada
* arquivo principal definido como `app.py`

---

## 📱 Instalação como aplicativo

O app pode ser instalado no celular ou desktop usando os recursos do navegador.

### No celular

No Chrome/Edge:

* abrir o menu do navegador
* tocar em **Adicionar à tela inicial** ou **Instalar aplicativo**

No iPhone/iPad:

* abrir no Safari
* tocar em **Compartilhar**
* selecionar **Adicionar à Tela de Início**

### No desktop

No Chrome/Edge:

* abrir o menu do navegador
* acessar a opção de instalação do aplicativo

> Observação: por limitações do navegador e do Streamlit Cloud, a instalação pode depender do menu do navegador, e o nome/ícone instalados podem não ter controle total.

---

## 📌 Limitações atuais

* a lógica, a interface e o acesso aos dados ainda estão concentrados em um único arquivo `app.py`
* a base principal ainda depende de uma URL de exportação de planilha
* o branding do app instalado (nome/ícone) pode ser limitado no Streamlit Cloud
* ainda não há backend separado nem persistência estruturada via API do Google Sheets

---

## 🔭 Próximos passos recomendados

* separar o projeto em módulos (`lógica`, `dados`, `interface`)
* substituir a leitura por URL exportada por integração estruturada com Google Sheets
* criar persistência de histórico de sorteios
* adicionar autenticação mais robusta
* evoluir o projeto para uma arquitetura com frontend próprio caso o objetivo seja um app com branding real

---

## 👨‍💻 Autor

**Weverton Gomes Costa**

Repositório:
[https://github.com/WevertonGomesCosta/sorteador-pelada](https://github.com/WevertonGomesCosta/sorteador-pelada)

---

## 📄 Licença

Definir licença do projeto.