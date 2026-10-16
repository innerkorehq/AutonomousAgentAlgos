# aaa-server in a container: the library and the server installed from the local source, running as an unprivileged user.
#   docker build -t aaa-server .
#   docker run -p 8000:8000 -e AAA_DATABASE_URL=postgresql://... -e AAA_API_KEYS='{"s3cret":["acme","ann"]}' aaa-server
FROM python:3.12-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY packages ./packages
RUN pip wheel --wheel-dir /wheels . ./packages/aaa-server asyncpg uvicorn

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN useradd --system --uid 10001 --home-dir /app --create-home aaa
COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links /wheels autonomous-agent-algos aaa-server asyncpg uvicorn && rm -rf /wheels
COPY deploy/app.py /app/app.py
WORKDIR /app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"
# 0.0.0.0 INSIDE the container; publish the port only to a reverse proxy that terminates TLS.
CMD ["aaa-server", "serve", "app:build", "--host", "0.0.0.0", "--port", "8000"]
