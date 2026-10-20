FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates make && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
COPY . .
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu && python -m pip install --no-cache-dir '.[train,bench,export,demo,dev]' && python -m pip install --no-cache-dir ./rust
RUN python -m pytest -q
CMD ["python", "scripts/reproduce.py", "--out", "/results/reproduction"]
