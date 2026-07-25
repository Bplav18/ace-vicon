FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgles2 \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=10000

CMD ["gunicorn", "--workers", "1", "--threads", "1", "--timeout", "300", "--bind", "0.0.0.0:10000", "app:app"]