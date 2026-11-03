#!/bin/sh
# 볼륨이 비어 있을 때(처음 뜰 때) 한 번만 실행된다.
# POSTGRES_DB(MLflow용)는 이미지가 만들고, Dagster용 DB를 하나 더 만든다.
# 실행 권한 없이 clone되면 엔트리포인트가 source로 읽으므로 set 옵션을 바꾸지 않는다.
# 실패는 ON_ERROR_STOP과 엔트리포인트의 set -e가 잡는다.

psql -v ON_ERROR_STOP=1 -v dbname="$DAGSTER_DB" \
    --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'EOSQL'
CREATE DATABASE :"dbname";
EOSQL
