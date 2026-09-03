# Proveniência do Dataset EPI v1

- Origem: exportação YOLOv8 do projeto Roboflow `epi-detection-rpi5` versão 1.
- Arquivo recebido: `epi-detection-rpi5.v1i.yolov8.zip`.
- SHA-256 do arquivo: `404D97A8D16B4B4C4B830109CC9111FB879A7600D5AB910922C41DD77C94371B`.
- Projeto: <https://universe.roboflow.com/guilherme-venancio/epi-detection-rpi5-6q1f3-gk9hv/dataset/1>.
- Licença informada pela exportação: CC BY 4.0.

A exportação continha 135 pares imagem/rótulo, originalmente divididos em 108/14/13.
Para a inspeção local da atividade, os mesmos pares foram redistribuídos de forma
determinística por nome de arquivo em 95 treino, 20 validação e 20 teste. Os rótulos
YOLO não foram modificados; somente os nomes das classes em `data.yaml` foram
normalizados para `capacete`, `colete` e `pessoa`.
