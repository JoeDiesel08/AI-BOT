FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persistent state directory for Fly.io volume mount
RUN mkdir -p /data/kraken_paper && chmod 777 /data/kraken_paper
VOLUME ["/data"]

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "app.py"]
