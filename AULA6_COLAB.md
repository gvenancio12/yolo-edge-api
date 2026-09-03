# Aula 6 — execução no Colab

## Transfer Learning MobileNetV2

Abra o notebook pelo Colab:

https://colab.research.google.com/github/gvenancio12/yolo-edge-api/blob/main/notebooks/aula6_mobilenetv2_transfer_learning.ipynb

Antes de executar, selecione **Runtime > Change runtime type > T4 GPU**. Depois use **Runtime > Run all**.

O notebook baixa o repositório, valida a GPU, mostra `train/` e `validation/` com as classes `capacete`, `colete` e `pessoa`, treina MobileNetV2 por exatamente 10 épocas e executa `model.evaluate(val_ds)` no final. Tire as três capturas após essas células: estrutura do dataset, acurácia da época 10 e avaliação.

O dataset de classificação é derivado de `datasets/epi-v1` por `scripts/prepare_aula6_classification_dataset.py`; não é evidência de execução. Resultados e capturas gerados no Colab devem ficar fora do repositório, na pasta externa `evidencias`.

## YOLOv8n — 50 épocas

Abra o notebook pelo Colab:

https://colab.research.google.com/github/gvenancio12/yolo-edge-api/blob/main/notebooks/aula6_yolov8_50_epochs.ipynb

Selecione **Runtime > Change runtime type > T4 GPU** e depois **Runtime > Run all**. O notebook executa 50 épocas sem early stopping, gera `best.pt`, apresenta mAP@0.5, precisão e recall, exporta uma imagem JPG com bounding boxes e baixa `evidencias-aula6-yolo.zip` para ser salvo na pasta externa `evidencias`.
