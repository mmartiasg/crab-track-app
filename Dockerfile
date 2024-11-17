FROM  --platform=linux/x86_64 python:3.10.12-slim
RUN apt update && apt install -y libgl1 libglib2.0-0
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY models models
COPY config config
COPY src src
COPY app/main.py main.py
