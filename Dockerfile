FROM python:3.11.3-slim-bullseye
COPY requirements.txt /
RUN apt update \
    && apt install -y wget git \
    && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
