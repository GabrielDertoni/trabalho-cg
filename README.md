# Trabalho de Computação Gráfica

## Integrantes do grupo

- Álvaro José Lopes - 10873365
- Gabriel da Cunha Dertoni - 11795717
- Natan Henrique Sanches - 11795680
- Pedro Lucas de Moliner de Castro - 11795784
- João Guilherme Jarochinski Marinho - 10698193

## Como instalar e executar o programa?

1. Instale o compilador e ferramenta de build do Rust de acordo com as instruções especificadas [aqui](https://www.rust-lang.org/tools/install)
2. Clone o repositório do git `git clone --recursive https://github.com/GabrielDertoni/trabalho-cg.git`
3. Compile o projeto em modo "release" `cargo build --release`
4. Rode o programa `cargo run --release -- config.toml`

## Alterando configurações

O arquivo `config.toml` contém as configurações da cena que será carregada pelo programa. Através dele é possível modificar diversas
configurações, bem como a resolução de display, os modelos que serão carregados e suas posições no mundo.
