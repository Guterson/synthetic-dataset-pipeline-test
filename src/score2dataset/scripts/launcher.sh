#!/bin/bash

# 1. Configurações de Caminhos (Baseadas no seu NFS)
PASTA_PROJETO_NFS="$HOME/seu_caminho_do_projeto_python"
PASTA_RESULTADOS_NFS="$HOME/sfizz_resultados"

# Configura o espaço local na máquina atual
PASTA_LOCAL="/tmp/gutemberg_render_$(whoami)"
ENTRADA_LOCAL="$PASTA_LOCAL/input"
SAIDA_LOCAL="$PASTA_LOCAL/output"

echo "=== [1/5] Preparando ambiente local no SSD da máquina ==="
# Limpa execuções antigas e recria as pastas locais
rm -rf "$PASTA_LOCAL"
mkdir -p "$ENTRADA_LOCAL" "$SAIDA_LOCAL"

echo "=== [2/5] Copiando dados do NFS para o hardware local ==="
# Copia seus scripts Python e arquivos necessários para o /tmp local
# The -L flag forces Linux to follow the symlink and copy the REAL data to the local SSD
cp -rL "$PASTA_PROJETO_NFS"/* "$ENTRADA_LOCAL/"

# Se você tiver uma pasta de amostras de áudio pesadas, copie-a aqui também
# cp -r "$HOME/minhas_amostras" "$PASTA_LOCAL/"

echo "=== [3/5] Ativando ambiente Python (Conda/Mamba) ==="
# Carrega o Conda e ativa o seu ambiente base (onde o g++ e bibliotecas estão)
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate base

echo "=== [4/5] Iniciando Renderização Paralela em Lote ==="
cd "$ENTRADA_LOCAL"

# Executa o seu script Python. Passamos as pastas locais como argumentos
# para garantir que o Python leia e grave no SSD local (/tmp)
python wrapper_sfizz.py --input "$ENTRADA_LOCAL" --output "$SAIDA_LOCAL"

echo "=== [5/5] Salvando resultados permanentemente no NFS ==="
mkdir -p "$PASTA_RESULTADOS_NFS"
# Transfere apenas o resultado final processado de volta para a sua conta de rede
cp -r "$SAIDA_LOCAL"/* "$PASTA_RESULTADOS_NFS/"

# Limpeza de segurança para não deixar rastro no computador do laboratório
rm -rf "$PASTA_LOCAL"

echo "=== PROCESSO CONCLUÍDO COM SUCESSO! ==="

