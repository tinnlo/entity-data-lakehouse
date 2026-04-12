FROM apache/airflow:2.9.0-python3.12
USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
USER airflow
ARG AIRFLOW_VERSION=2.9.0
ARG PYTHON_VERSION=3.12
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
COPY airflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --constraint "${CONSTRAINT_URL}" -r /tmp/requirements.txt
