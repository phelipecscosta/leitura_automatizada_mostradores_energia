# Leitura automatizada de mostradores de energia

Software que lê automaticamente as fotografias de medidores de energia
coletadas nas rotinas de leitura de consumo de uma concessionária de energia
do Nordeste. Ele confere a leitura registrada em campo, separa as fotos que
não permitem leitura e organiza as que precisam de verificação humana.

**Situação:** em desenvolvimento. Ainda não há versão para instalação.

## O que este repositório contém

- O código do software entregue ao cliente, para instalação nativa no Windows
  ou em contêiner Docker.
- O pipeline de treino dos modelos (pasta `training/`), para que cada peso
  publicado possa ser reproduzido a partir do código que o gerou.

## O que este repositório não contém

- **Dados do cliente.** Nenhuma foto, planilha ou identificador do cliente é
  versionado. O software lê as fotos de uma pasta local, escolhida pelo
  utilizador.
- **Pesos dos modelos.** São publicados nos Releases deste repositório e
  baixados automaticamente na instalação.

## Ambiente de desenvolvimento

Estas instruções preparam o ambiente para desenvolver e treinar. A instalação
do software pelo cliente terá instalador próprio e manual separado.

### Requisitos

- Windows 10 ou 11, 64 bits
- Python 3.11
- Git
- Opcional: GPU NVIDIA com driver compatível com CUDA 12.6. Sem GPU, tudo
  funciona na CPU.

### Instalação

No PowerShell:

```powershell
git clone https://github.com/phelipecscosta/leitura_automatizada_mostradores_energia.git
cd leitura_automatizada_mostradores_energia

# Ambiente virtual com Python 3.11
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Dependências com as versões exatas do travamento
python -m pip install -r requirements-lock-cu126.txt

# O próprio pacote, em modo editável, sem buscar outras dependências
python -m pip install -e ".[dev]" --no-deps

# Configuração local: copie o modelo e preencha o caminho dos dados
Copy-Item .env.example .env
```

Se a ativação do ambiente falhar com a mensagem "running scripts is disabled",
execute uma vez `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
e tente de novo.

### Testes

```powershell
python -m pytest
```

Os testes não dependem dos dados. Sem a base configurada no `.env`, o teste
que a utiliza é pulado automaticamente.

## Estrutura do repositório

```
├── src/
│ └── meter_reader/ pacote do produto: o que o cliente instala
├── tests/ testes automatizados
├── training/ pipeline de treino, fora do pacote instalado
├── scripts/ utilitários, como o download de pesos
├── packaging/ instalador para Windows e imagem Docker
└── docs/ manuais de instalação e de uso
```




## Licença

O código está sob a licença MIT (arquivo `LICENSE`). A licença cobre apenas o
código; não se aplica a dados nem a pesos de modelos.
