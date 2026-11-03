# MLflow 공식 이미지에는 PostgreSQL 드라이버와 boto3가 없어서 추가한다.
# 서버 버전은 pyproject.toml의 mlflow 클라이언트 버전과 맞춘다.
FROM ghcr.io/mlflow/mlflow:v3.17.2

RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.13 \
    boto3==1.43.123
