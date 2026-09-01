# Dataset EPI v1

## 1. Captura na Raspberry

Antes de executar, encerre o servidor MJPEG, pois ele usa a mesma câmera CSI.
Enquadre pessoas usando capacete e colete, varie distância, pose, iluminação e
oclusão. Não use imagens sem uma das classes pretendidas.

```bash
cd ~/yolo-edge-api
python3 scripts/capture_frames.py --count 180 --min-change 2
```

O comando grava JPEGs e `capture_summary.json` em `~/epi-capture/raw`. Registre a
saída final, que informa frames salvos e descartados.

## 2. Roboflow

1. Crie um projeto **Object Detection** chamado `epi`.
2. Crie exatamente as classes: `capacete`, `colete` e `pessoa`.
3. Envie os frames de `~/epi-capture/raw`, anote todas as instâncias e revise os
   rótulos antes de gerar a versão.
4. Gere a versão 1 com split **70% train / 15% valid / 15% test**. Aplique
   augmentations até a tela de prévia indicar pelo menos **350 imagens em train**.
5. Exporte a versão no formato **YOLOv8** para `datasets/epi-v1` no repositório.

Registre as telas do projeto anotado e da versão 1 na pasta local
`evidencias-epi-dataset-v1`.

## 3. Inspeção e DVC

```bash
python3 scripts/inspect_dataset.py \
  --dataset datasets/epi-v1 \
  --report ~/epi-capture/inspect_dataset_report.json

dvc add datasets/epi-v1
git add datasets/epi-v1.dvc .gitignore
git commit -m "data: add epi dataset v1"
dvc push
git push origin main
```

O inspetor exige os três splits, classes exatamente nomeadas, rótulos YOLO válidos,
proporção 70/15/15 (tolerância de 3 pontos percentuais), no mínimo 350 imagens de
treino e anotações das três classes.
