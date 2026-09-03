# Aula 5 — resultados dos experimentos

## Método

As 20 imagens do split `valid` de `datasets/epi-v1` foram avaliadas em cada cenário. As predições foram obtidas do modelo EPI treinado no Roboflow (`epi-detection-rpi5-6q1f3-gk9hv/1`) e o mAP@0.5 foi calculado localmente, com as três classes `capacete`, `colete` e `pessoa`. A API é autenticada por variável de ambiente; a chave não é registrada neste repositório.

A referência exigida no enunciado é E1-A. Por isso, os deltas abaixo são contra E1-A. O arquivo de dados reproduzíveis está em `evidencias-aula5/resultados.json`.

| Experimento | Configuração | mAP@0.5 | Delta vs. E1-A | Observação objetiva |
|---|---|---:|---:|---|
| E1-A | BGR sem conversão | 0.2818 | +0.0000 | Referência; canais invertidos. |
| E1-B | RGB correto | 0.4098 | +0.1280 | Melhor resultado em condição normal. |
| E2-A | resize simples 320×320 | 0.4034 | +0.1217 | Bom, mas altera a proporção. |
| E2-B | letterbox 320, padding 114 | 0.4034 | +0.1217 | Mesmo mAP e geometria preservada. |
| E3-A | sem filtro | 0.4098 | +0.1280 | Melhor entre os filtros. |
| E3-B | GaussianBlur 3×3, sigma 0.8 | 0.4022 | +0.1205 | Pequena perda de detalhe. |
| E3-C | GaussianBlur 5×5, sigma 1.5 | 0.3950 | +0.1132 | Perda maior de detalhe. |
| E3-D | medianBlur kernel 3 | 0.4015 | +0.1198 | Também abaixo de sem filtro. |
| E4-A | escurecida, sem equalização | 0.3721 | +0.0903 | Baixa iluminação degrada a detecção. |
| E4-B | escurecida, equalizeHist global | 0.3214 | +0.0396 | Pior cenário de baixa luz. |
| E4-C | escurecida, CLAHE clip 2, tile 8 | 0.3875 | +0.1058 | Melhor recuperação na baixa luz. |

## Síntese técnica e configuração adotada

`preprocessor.py` mantém `PreprocessConfig()` com `infer_size=320`, `convert_rgb=True`, `use_letterbox=True`, sem blur e sem CLAHE. Esta é a configuração de operação padrão para a Raspberry: RGB é obrigatório porque E1-B superou E1-A em 0.1280 mAP; 320 reduz custo de inferência; e letterbox é mantido porque empatou com resize simples, porém preserva proporção e permite reverter as caixas corretamente.

Filtros Gaussian e median permanecem desativados: todos reduziram o mAP em comparação a E3-A, portanto não justificam custo e perda de detalhe no fluxo normal. `CONFIG_LOW_LIGHT` continua como alternativa explícita (`clahe=True`, `clip=2.0`, `tile=8`, espaço LAB): E4-C foi melhor que a imagem escura sem equalização (+0.0155), enquanto equalização global foi prejudicial. CLAHE não entra no padrão porque, mesmo melhorando baixa luz, ficou abaixo da condição iluminada normal.

Os resultados medem este modelo, este split e estas transformações; não devem ser generalizados sem repetir a avaliação em novas imagens da câmera.
