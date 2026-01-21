"""
Database Connection Manager for XOR Project

Provides unified interface for both SQLite and MariaDB backends.
Reads credentials from ../.env file for MariaDB connections.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union
from contextlib import contextmanager
import sqlite3

import mysql.connector
from mysql.connector import pooling
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration loaded from environment"""

    def __init__(self, env_path: Optional[Path] = None):
        """
        Load database configuration from .env file.

        Args:
            env_path: Path to .env file (default: ../.env)
        """
        if env_path is None:
            # Look for .env in parent directory
            env_path = Path(__file__).parent.parent.parent / ".env"

        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment from {env_path}")
        else:
            logger.warning(f".env file not found at {env_path}")

        # MariaDB configuration
        self.mariadb_host = os.getenv("MARIADB_HOST", "10.0.0.103")
        self.mariadb_user = os.getenv("MARIADBUSER", "devuser")
        self.mariadb_password = os.getenv("MARIADBDEVPWD", "")
        self.mariadb_database = os.getenv("MARIADB_DATABASE", "xor_project")
        self.mariadb_port = int(os.getenv("MARIADB_PORT", "3306"))

        # Connection pool settings
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "5"))
        self.pool_name = os.getenv("DB_POOL_NAME", "xor_pool")


class DatabaseConnection:
    """
    Database connection manager supporting both SQLite and MariaDB.

    Usage:
        # SQLite (default for local/dev)
        db = DatabaseConnection(backend='sqlite', db_path='features.db')

        # MariaDB (for production)
        db = DatabaseConnection(backend='mariadb')

        # Using context manager
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM features LIMIT 10")
    """

    _pool = None  # Class-level connection pool

    def __init__(self,
                 backend: str = 'sqlite',
                 db_path: Optional[Union[str, Path]] = None,
                 config: Optional[DatabaseConfig] = None):
        """
        Initialize database connection manager.

        Args:
            backend: 'sqlite' or 'mariadb'
            db_path: Path to SQLite database file (ignored for MariaDB)
            config: DatabaseConfig instance (created if None)
        """
        self.backend = backend.lower()

        if self.backend not in ['sqlite', 'mariadb']:
            raise ValueError(f"Unsupported backend: {backend}. Use 'sqlite' or 'mariadb'")

        self.db_path = Path(db_path) if db_path else None
        self.config = config or DatabaseConfig()

        if self.backend == 'mariadb':
            self._init_pool()

        logger.info(f"DatabaseConnection initialized with {self.backend} backend")

    def _init_pool(self):
        """Initialize MariaDB connection pool (class-level)"""
        if DatabaseConnection._pool is None:
            try:
                DatabaseConnection._pool = pooling.MySQLConnectionPool(
                    pool_name=self.config.pool_name,
                    pool_size=self.config.pool_size,
                    pool_reset_session=True,
                    host=self.config.mariadb_host,
                    port=self.config.mariadb_port,
                    user=self.config.mariadb_user,
                    password=self.config.mariadb_password,
                    database=self.config.mariadb_database,
                    ssl_disabled=True,
                    autocommit=False
                )
                logger.info(f"MariaDB connection pool created: {self.config.pool_name}")
            except mysql.connector.Error as e:
                logger.error(f"Failed to create connection pool: {e}")
                raise

    @contextmanager
    def get_connection(self):
        """
        Get database connection as context manager.

        Yields:
            Database connection (sqlite3.Connection or mysql.connector.connection)
        """
        conn = None
        try:
            if self.backend == 'sqlite':
                conn = self._get_sqlite_connection()
            else:
                conn = self._get_mariadb_connection()

            yield conn

            # Commit if no exception
            if conn:
                conn.commit()

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise

        finally:
            if conn:
                conn.close()

    def _get_sqlite_connection(self):
        """Get SQLite connection"""
        if not self.db_path:
            raise ValueError("db_path required for SQLite backend")

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        return conn

    def _get_mariadb_connection(self):
        """Get MariaDB connection from pool"""
        if DatabaseConnection._pool is None:
            raise RuntimeError("Connection pool not initialized")

        return DatabaseConnection._pool.get_connection()

    def test_connection(self) -> bool:
        """
        Test database connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if self.backend == 'sqlite':
                    cursor.execute("SELECT 1")
                else:
                    cursor.execute("SELECT 1")
                result = cursor.fetchone()
                logger.info(f"Connection test successful: {result}")
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_cursor_dict(self, conn):
        """
        Get cursor that returns results as dictionaries.

        Args:
            conn: Database connection

        Returns:
            Cursor configured to return dict results
        """
        if self.backend == 'sqlite':
            conn.row_factory = sqlite3.Row
            return conn.cursor()
        else:
            return conn.cursor(dictionary=True)

    @staticmethod
    def get_placeholder(backend: str) -> str:
        """
        Get SQL placeholder for parameterized queries.

        Args:
            backend: 'sqlite' or 'mariadb'

        Returns:
            '?' for SQLite, '%s' for MariaDB
        """
        return '?' if backend == 'sqlite' else '%s'

    def execute_script(self, sql_script: str):
        """
        Execute SQL script (multiple statements).

        Args:
            sql_script: SQL script with multiple statements
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if self.backend == 'sqlite':
                cursor.executescript(sql_script)
            else:
                # MariaDB requires splitting and executing individually
                for statement in sql_script.split(';'):
                    statement = statement.strip()
                    if statement:
                        cursor.execute(statement)

        logger.info("SQL script executed successfully")


def create_database(backend: str = 'mariadb',
                   db_name: str = 'xor_project',
                   drop_if_exists: bool = False) -> bool:
    """
    Create the XOR project database.

    Args:
        backend: 'sqlite' or 'mariadb'
        db_name: Database name
        drop_if_exists: Drop database if it already exists

    Returns:
        True if successful
    """
    if backend == 'sqlite':
        logger.info("SQLite databases are created automatically")
        return True

    # MariaDB: create database
    config = DatabaseConfig()

    try:
        # Connect without specifying database
        conn = mysql.connector.connect(
            host=config.mariadb_host,
            port=config.mariadb_port,
            user=config.mariadb_user,
            password=config.mariadb_password,
            ssl_disabled=True
        )

        cursor = conn.cursor()

        if drop_if_exists:
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
            logger.info(f"Dropped database {db_name}")

        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} "
                      f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        logger.info(f"Created database {db_name}")

        conn.close()
        return True

    except mysql.connector.Error as e:
        logger.error(f"Failed to create database: {e}")
        return False


# Convenience function for quick connections
def get_db(backend: str = 'sqlite', db_path: Optional[str] = None) -> DatabaseConnection:
    """
    Quick helper to get a DatabaseConnection instance.

    Args:
        backend: 'sqlite' or 'mariadb'
        db_path: Path for SQLite database

    Returns:
        DatabaseConnection instance
    """
    return DatabaseConnection(backend=backend, db_path=db_path)


if __name__ == "__main__":
    # Test the connection
    logging.basicConfig(level=logging.INFO)

    print("Testing MariaDB connection...")
    db = DatabaseConnection(backend='mariadb')

    if db.test_connection():
        print("✅ MariaDB connection successful!")
    else:
        print("❌ MariaDB connection failed")

    print("\nTesting SQLite connection...")
    db_sqlite = DatabaseConnection(backend='sqlite', db_path='test.db')

    if db_sqlite.test_connection():
        print("✅ SQLite connection successful!")
    else:
        print("❌ SQLite connection failed")
