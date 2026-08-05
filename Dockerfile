# The repository root IS the `rlrp_smsemoa` package, so it is copied into
# /app/rlrp_smsemoa and /app is placed on the import path.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements first, so the dependency layer survives code changes
COPY requirements.txt /app/rlrp_smsemoa/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/rlrp_smsemoa/requirements.txt

COPY . /app/rlrp_smsemoa/

RUN mkdir -p /app/results

# Short smoke run by default; docker-compose defines the real experiments.
CMD ["python", "rlrp_smsemoa/run_full_experiment.py", \
     "--problems", "dtlz2", "--n_seeds", "3", "--t_max", "1000", "--mu", "30", \
     "--outdir", "/app/results/smoke"]
