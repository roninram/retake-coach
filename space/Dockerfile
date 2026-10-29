# Retake Coach public demo on a Hugging Face Space (Docker, CPU basic is enough).
# The app comes from the public GitHub repo; after a new commit there, use "Factory rebuild" in the Space settings.
# Secrets: HF_TOKEN (required for the DINOv3 obstruction map) and MODEL_API_KEY (optional, Muse Spark coach's note).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1
WORKDIR /home/user/app
RUN git clone --depth 1 https://github.com/roninram/retake-coach.git .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r space/requirements.txt
RUN python space/build_examples.py examples
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=false \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN 2>/dev/null || true) python space/prefetch.py
# a public demo: show the banner, and keep Muse Spark spending small
ENV RC_DEMO=1 MUSE_DAILY_USD=0.10 MUSE_DAILY_REQUESTS=40 MUSE_TOTAL_USD=0.50
EXPOSE 7860
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
