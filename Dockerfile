FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    OMP_NUM_THREADS=2 \
    TF_NUM_INTRAOP_THREADS=2 \
    TF_NUM_INTEROP_THREADS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# The trained model is NOT baked into the image: mount ./model at runtime (see docker-compose.yml).

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

CMD ["python", "app_gradio.py"]
