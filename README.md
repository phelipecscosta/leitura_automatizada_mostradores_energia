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

## Estrutura do repositório

├── src/
│ └── meter_reader/ pacote do produto: o que o cliente instala
├── tests/ testes automatizados
├── training/ pipeline de treino, fora do pacote instalado
├── scripts/ utilitários, como o download de pesos
├── packaging/ instalador para Windows e imagem Docker
└── docs/ manuais de instalação e de uso


*** REMOVER AO FINAL ****As pastas que ainda não têm conteúdo contêm apenas um arquivo `.gitkeep`, usado para que o Git as mantenha no repositório.

## Licença

O código está sob a licença MIT (arquivo `LICENSE`). A licença cobre apenas o
código; não se aplica a dados nem a pesos de modelos.