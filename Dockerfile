FROM python:3.12-slim-bookworm

COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

RUN mkdir -p /app

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libgomp1 libatlas-base-dev liblapack-dev

COPY pyproject.toml .
COPY uv.lock .

RUN uv sync

COPY . . 

EXPOSE 5002

ENV SENTENCE_TRANSFORMERS_HOME="/app/models/"
ENV HOST="0.0.0.0"

CMD ["uv", "run", "app.py"]

# TODO switch to use a WSGI
# ENV TOKENIZERS_PARALLELISM=true
# CMD ["uv" "run" "gunicorn" "-w" "3" "-b" "0.0.0.0:5002" "--preload" "app:app"]


