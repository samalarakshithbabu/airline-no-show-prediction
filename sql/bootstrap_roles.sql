CREATE ROLE :"runtime_user"
    WITH LOGIN PASSWORD :'runtime_password'
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
CREATE ROLE :"migration_user"
    WITH LOGIN PASSWORD :'migration_password'
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;

GRANT CONNECT ON DATABASE :"database_name" TO :"runtime_user", :"migration_user";
GRANT USAGE ON SCHEMA public TO :"runtime_user";
GRANT CREATE ON SCHEMA public TO :"migration_user";
GRANT CREATE ON DATABASE :"database_name" TO :"migration_user";

ALTER DEFAULT PRIVILEGES FOR ROLE :"migration_user" IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO :"runtime_user";
ALTER DEFAULT PRIVILEGES FOR ROLE :"migration_user" IN SCHEMA public
    GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO :"runtime_user";
