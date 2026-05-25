FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/                models/
COPY static/                static/
COPY main.py                .
COPY mcp_server.py          .
COPY *.csv                  .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8001
