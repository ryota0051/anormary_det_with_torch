FROM python:3.8-slim

RUN apt-get update && apt-get install -y --no-install-recommends wget build-essential libreadline-dev \
libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev liblzma-dev zlib1g-dev uuid-dev libffi-dev libdb-dev

WORKDIR /work

COPY ./requirements.txt /work/

RUN python -m pip install --upgrade pip && pip install -r requirements.txt
