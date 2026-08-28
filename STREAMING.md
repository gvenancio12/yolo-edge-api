# Pipeline de câmera e MJPEG

Os scripts desta pasta são executados diretamente na Raspberry Pi, onde `Picamera2`
controla a câmera CSI. A inferência continua no container da API, pelo endpoint
`http://127.0.0.1:8000/predict`.

## Pré-requisitos

Na Raspberry, confirme que a câmera e a API estão disponíveis:

```bash
rpicam-hello --list-cameras
curl http://127.0.0.1:8000/health
```

## Versão 1 — diagnóstico síncrono

```bash
cd ~/yolo-edge-api
python3 -m streaming.v1_naive --frames 50
```

Ela mede captura, codificação JPEG, chamada HTTP e inferência para cada frame. Os
entregáveis são `~/yolo-edge-evidencias/v1_naive_report.md` e
`~/yolo-edge-evidencias/v1_naive_diagnostic.json`, com 50 amostras individuais.

## Versão 3 — AVI otimizado

```bash
python3 -m streaming.v3_optimized --frames 100
```

A versão 3 mantém a captura em fluxo contínuo, faz inferência assíncrona apenas no
frame mais recente e reaproveita o último resultado para manter o vídeo fluido. Ela
gera `~/yolo-edge-evidencias/v3_optimized.avi` em MJPEG/AVI, com bounding boxes e OSD de FPS e
latência.

Copie o arquivo para o computador com:

```powershell
scp guilherme@guilherme-raspberry:~/yolo-edge-evidencias/v3_optimized.avi .\evidencias\
```

## Servidor MJPEG

```bash
python3 -m streaming.mjpeg_server --port 8081
```

Abra `http://guilherme-raspberry:8081/` no navegador da rede Tailscale e registre a
captura. O endpoint `/stream.mjpg` contém o MJPEG e `/snapshot.jpg` retorna uma
imagem JPEG anotada para conferência. O endpoint da API exigido na aula permanece:

```bash
curl http://127.0.0.1:8000/health
```
