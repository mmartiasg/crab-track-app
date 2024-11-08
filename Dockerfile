FROM python:3.10.12
RUN apt update && apt install -y libgl1
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY models models
COPY config config
COPY src src
COPY main.py main.py
