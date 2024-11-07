FROM python:3.10.12
WORKDIR /app
COPY config config
COPY src src
COPY models models
COPY main.py main.py
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

