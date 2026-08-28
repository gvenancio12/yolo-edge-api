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

### Alternativa com fonte pública

Se não houver EPI físico disponível, use o dataset público **Ultralytics
Construction-PPE** com atribuição e licença AGPL-3.0. Esta alternativa deve ser
descrita como **importação de dados públicos**, não como captura própria da câmera.

```bash
mkdir -p ~/epi-public-source && cd ~/epi-public-source
curl -L -o construction-ppe.zip \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip
unzip construction-ppe.zip

cd ~/yolo-edge-api
python3 scripts/prepare_epi_dataset.py \
  --source ~/epi-public-source/construction-ppe \
  --target datasets/epi-v1 \
  --count 500
```

O resultado possui 350 imagens de treino, 75 de validação e 75 de teste, com as
classes da atividade. O arquivo `datasets/epi-v1/SOURCE.md` guarda a proveniência.

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
