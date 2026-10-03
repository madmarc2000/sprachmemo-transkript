# sprachmemo-transkript — Worker: drei Whisper-Fassungen einer Aufnahme
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
# Reihenfolge zählt: faster-whisper zieht das normale ctranslate2, CrisperWhisper braucht den Fork
# ctranslate2-crisperwhisper (gleicher Modulname, mehr APIs) — der Fork muss zuletzt kommen.
# Kein torch im Image: das Umwandeln des CrisperWhisper-Modells läuft einmalig (siehe README).
RUN pip install --no-cache-dir faster-whisper "nvidia-cudnn-cu12==9.*" nvidia-cublas-cu12 \
 && pip install --no-cache-dir crisperwhisper \
 && pip install --no-cache-dir --force-reinstall --no-deps "ctranslate2-crisperwhisper>=4.7.1.post3,<5"
# Ohne diesen Pfad findet CTranslate2 cuBLAS/cuDNN nicht
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib
ENV HF_HOME=/cache/hf XDG_CACHE_HOME=/cache PYTHONUNBUFFERED=1
COPY fassungen.py /app/fassungen.py
ENTRYPOINT ["python", "/app/fassungen.py"]
