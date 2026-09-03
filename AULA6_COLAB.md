# Aula 6 — execução no Colab

Abra o notebook pelo Colab:

https://colab.research.google.com/github/gvenancio12/yolo-edge-api/blob/main/notebooks/aula6_mobilenetv2_transfer_learning.ipynb

Antes de executar, selecione **Runtime > Change runtime type > T4 GPU**. Depois use **Runtime > Run all**.

O notebook baixa o repositório, valida a GPU, mostra `train/` e `validation/` com as classes `capacete`, `colete` e `pessoa`, treina MobileNetV2 por exatamente 10 épocas e executa `model.evaluate(val_ds)` no final. Tire as três capturas após essas células: estrutura do dataset, acurácia da época 10 e avaliação.

O dataset de classificação é derivado de `datasets/epi-v1` por `scripts/prepare_aula6_classification_dataset.py`; não é evidência de execução. Resultados e capturas gerados no Colab devem ficar fora do repositório, na pasta externa `evidencias`.
