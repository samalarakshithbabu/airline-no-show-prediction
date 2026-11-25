#!/bin/sh
set -eu

psql \
  --username "$POSTGRES_USER" \
  --dbname "$POSTGRES_DB" \
  --set=runtime_user="$DB_RUNTIME_USER" \
  --set=runtime_password="$DB_RUNTIME_PASSWORD" \
  --set=migration_user="$DB_MIGRATION_USER" \
  --set=migration_password="$DB_MIGRATION_PASSWORD" \
  --set=database_name="$POSTGRES_DB" \
  --file=/docker-entrypoint-initdb.d/00-create-roles.sql
