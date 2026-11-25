"""
Database Connection and Management Module
Supports both SQLite (development) and PostgreSQL (production)
"""

import os
import sys
from pathlib import Path
from typing import Generator, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, event, text, inspect
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool, StaticPool
import logging

# Add backend/app to path for imports
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

logger = logging.getLogger(__name__)

# Import configuration from the local config module
from config import (
    SQLALCHEMY_DATABASE_URI,
    SQLALCHEMY_TRACK_MODIFICATIONS,
    SQLALCHEMY_ENGINE_OPTIONS,
    DB_TYPE
)

# Create engine based on database type
if DB_TYPE == 'sqlite':
    # SQLite for development - no connection pooling needed
    engine = create_engine(
        SQLALCHEMY_DATABASE_URI,
        connect_args={'check_same_thread': False},
        poolclass=StaticPool,
        echo=False,
        logging_name='database'
    )
    logger.info("Using SQLite database for development")
else:
    # PostgreSQL for production with connection pooling
    engine = create_engine(
        SQLALCHEMY_DATABASE_URI,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
        logging_name='database'
    )
    logger.info("Using PostgreSQL database")

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db_url() -> str:
    """Get the current database URL."""
    return SQLALCHEMY_DATABASE_URI


def is_sqlite() -> bool:
    """Check if using SQLite."""
    return DB_TYPE == 'sqlite'


def is_postgresql() -> bool:
    """Check if using PostgreSQL."""
    return DB_TYPE == 'postgresql'


class DatabaseManager:
    """Database connection and management class."""
    
    def __init__(self, db_url: str = None):
        """Initialize database manager."""
        self.db_url = db_url or SQLALCHEMY_DATABASE_URI
        self._engine = None
        self._session_factory = None
    
    @property
    def engine(self):
        """Get database engine."""
        return engine
    
    @property
    def session_factory(self):
        """Get session factory."""
        return SessionLocal
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Context manager for database sessions."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def test_connection(self) -> dict:
        """Test database connection."""
        result = {
            'status': 'unknown',
            'db_type': DB_TYPE,
            'message': ''
        }
        
        try:
            with self.engine.connect() as conn:
                if is_sqlite():
                    conn.execute(text("SELECT 1"))
                else:
                    conn.execute(text("SELECT 1"))
            
            result['status'] = 'healthy'
            result['message'] = f'Connected to {DB_TYPE} database successfully'
            logger.info(result['message'])
            
        except Exception as e:
            result['status'] = 'unhealthy'
            result['message'] = f'Database connection failed: {str(e)}'
            logger.error(result['message'])
        
        return result
    
    def migrate_legacy_primary_keys(self, target_engine=None):
        """Rename legacy ORM `id` columns to the stable SQL schema identifiers."""
        target_engine = target_engine or self.engine
        renames = (
            ('passengers', 'id', 'passenger_id'),
            ('flights', 'id', 'flight_id'),
            ('bookings', 'id', 'booking_id'),
            ('predictions', 'id', 'prediction_id'),
            ('model_performance', 'id', 'performance_id'),
            ('monitoring_logs', 'id', 'log_id'),
        )
        inspector = inspect(target_engine)
        tables = set(inspector.get_table_names())
        for table, old_name, new_name in renames:
            if table not in tables:
                continue
            columns = {column['name'] for column in inspect(target_engine).get_columns(table)}
            if old_name in columns and new_name not in columns:
                with target_engine.begin() as connection:
                    connection.execute(text(
                        f'ALTER TABLE "{table}" RENAME COLUMN "{old_name}" TO "{new_name}"'
                    ))
                logger.info(f'Migrated {table}.{old_name} to {new_name}')

    def init_schema(self):
        """Create missing tables from the SQLAlchemy model metadata."""
        from models.schemas import Base
        self.migrate_legacy_primary_keys()
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database schema initialized from models")
    
    def create_tables(self):
        """Create all tables from models."""
        from models.schemas import Base
        Base.metadata.create_all(bind=self.engine)
        logger.info("All tables created")
    
    def drop_tables(self):
        """Drop all tables (use with caution)."""
        from models.schemas import Base
        Base.metadata.drop_all(bind=self.engine)
        logger.info("All tables dropped")


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database sessions in Flask."""
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


def init_db(app=None):
    """Initialize database with Flask app."""
    if app:
        @app.before_request
        def before_request():
            from flask import g
            if 'db' not in g:
                g.db = db_manager.get_session()
        
        @app.teardown_appcontext
        def teardown(exception=None):
            from flask import g
            db = g.pop('db', None)
            if db is not None:
                db.close()


def create_database_backup():
    """Create a backup of the database (SQLite only)."""
    if is_sqlite():
        import shutil
        from datetime import datetime
        
        db_path = os.getenv('SQLITE_DB_PATH', 'instance/airline_noshow.db')
        backup_path = f"instance/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        if os.path.exists(db_path):
            shutil.copy(db_path, backup_path)
            logger.info(f"Database backup created: {backup_path}")
            return backup_path
    
    return None


# Table creation helper
def create_all_tables():
    """Create all database tables."""
    from models.schemas import Base
    Base.metadata.create_all(bind=engine)
    return True


if __name__ == "__main__":
    # Test database connection
    print("Testing database connection...")
    result = db_manager.test_connection()
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    
    if result['status'] == 'healthy':
        print("\nInitializing database schema...")
        db_manager.init_schema()
        print("Database schema initialized successfully")