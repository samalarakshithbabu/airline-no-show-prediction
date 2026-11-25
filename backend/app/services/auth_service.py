"""
Authentication Service for Airline No-Show Prediction System
Provides secure user authentication with JWT tokens and password hashing
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
import jwt
import bcrypt
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging

# Add backend/app to path for imports
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from config import (
    JWT_SECRET_KEY,
    JWT_ACCESS_TOKEN_EXPIRY_HOURS,
    SECRET_KEY
)

logger = logging.getLogger(__name__)


class AuthService:
    """Authentication service with JWT and password hashing."""
    
    def __init__(self, db_session: Session = None):
        """Initialize authentication service."""
        self.db_session = db_session
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            plain_password: Plain text password
            hashed_password: Stored password hash
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            return bcrypt.checkpw(
                plain_password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
        except Exception:
            return False
    
    def create_access_token(self, 
                            user_id: str, 
                            username: str, 
                            role: str,
                            expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT access token.
        
        Args:
            user_id: User's unique identifier
            username: Username
            role: User role (admin/user)
            expires_delta: Token expiration time
            
        Returns:
            JWT token string
        """
        if expires_delta is None:
            expires_delta = timedelta(hours=JWT_ACCESS_TOKEN_EXPIRY_HOURS)
        
        expire = datetime.utcnow() + expires_delta
        
        payload = {
            'sub': user_id,
            'username': username,
            'exp': expire,
            'iat': datetime.utcnow(),
            'type': 'access'
        }
        
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')
        return token
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode and validate a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Token payload dict or None if invalid
        """
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def validate_token(self, token: str) -> tuple:
        """
        Validate a token and return user info.
        
        Args:
            token: JWT token
            
        Returns:
            Tuple of (is_valid, user_info or error_message)
        """
        payload = self.decode_token(token)
        
        if payload is None:
            return (False, "Invalid or expired token")
        
        if payload.get('type') != 'access':
            return (False, "Invalid token type")
        
        return (True, payload)
    
    def register_user(self, 
                      username: str, 
                      email: str, 
                      password: str,
                      role: str = 'user') -> Dict[str, Any]:
        """
        Register a new user.
        
        Args:
            username: Username
            email: Email address
            password: Plain text password
            role: User role
            
        Returns:
            Dict with status and user info or error
        """
        if not self.db_session:
            return {'success': False, 'error': 'Database session required'}
        
        try:
            # Check if username or email already exists
            check_query = text("""
                SELECT id, username, email FROM users 
                WHERE username = :username OR email = :email
            """)
            existing = self.db_session.execute(
                check_query, 
                {'username': username, 'email': email}
            ).fetchone()
            
            if existing:
                return {
                    'success': False, 
                    'error': 'Username or email already exists'
                }
            
            # Hash password
            password_hash = self.hash_password(password)
            
            # Insert new user (generate UUID for SQLite compatibility)
            import uuid
            user_id = uuid.uuid4().hex
            
            insert_query = text("""
                INSERT INTO users (id, username, email, password_hash, role, is_active, created_at)
                VALUES (:id, :username, :email, :password_hash, :role, true, :created_at)
            """)
            
            self.db_session.execute(
                insert_query,
                {
                    'id': user_id,
                    'username': username,
                    'email': email,
                    'password_hash': password_hash,
                    'role': role,
                    'created_at': datetime.utcnow()
                }
            )
            
            self.db_session.commit()
            
            logger.info(f"User registered: {username}")
            
            return {
                'success': True,
                'user': {
                    'id': user_id,
                    'username': username,
                    'email': email,
                    'role': role
                }
            }
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Registration error: {e}")
            return {'success': False, 'error': str(e)}
    
    def authenticate_user(self, 
                          username: str, 
                          password: str) -> Dict[str, Any]:
        """
        Authenticate a user with username/email and password.
        
        Args:
            username: Username or email
            password: Plain text password
            
        Returns:
            Dict with status and token/user info or error
        """
        if not self.db_session:
            return {'success': False, 'error': 'Database session required'}
        
        try:
            # Find user by username or email
            query = text("""
                SELECT id, username, email, password_hash, role, is_active, last_login
                FROM users 
                WHERE (username = :username OR email = :username) AND is_active = true
            """)
            
            result = self.db_session.execute(
                query, 
                {'username': username}
            ).fetchone()
            
            if not result:
                return {
                    'success': False, 
                    'error': 'Invalid username or password'
                }
            
            user_id, db_username, email, password_hash, role, is_active, last_login = result
            
            # Verify password
            if not self.verify_password(password, password_hash):
                return {
                    'success': False, 
                    'error': 'Invalid username or password'
                }
            
            # Update last login
            update_query = text("""
                UPDATE users SET last_login = :last_login WHERE id = :user_id
            """)
            self.db_session.execute(
                update_query,
                {'user_id': user_id, 'last_login': datetime.utcnow()}
            )
            self.db_session.commit()
            
            # Create access token
            access_token = self.create_access_token(
                user_id=str(user_id),
                username=db_username,
                role=role
            )
            
            logger.info(f"User authenticated: {db_username}")
            
            return {
                'success': True,
                'token': access_token,
                'expires_in': JWT_ACCESS_TOKEN_EXPIRY_HOURS * 3600,
                'user': {
                    'id': str(user_id),
                    'username': db_username,
                    'email': email,
                    'role': role
                }
            }
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Authentication error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user by ID.
        
        Args:
            user_id: User UUID
            
        Returns:
            User dict or None
        """
        if not self.db_session:
            return None
        
        try:
            query = text("""
                SELECT id, username, email, role, is_active, created_at, last_login
                FROM users WHERE id = :user_id
            """)
            
            result = self.db_session.execute(
                query, 
                {'user_id': user_id}
            ).fetchone()
            
            if not result:
                return None

            def isoformat(value):
                return value.isoformat() if hasattr(value, 'isoformat') else str(value) if value else None
            
            return {
                'id': str(result[0]),
                'username': result[1],
                'email': result[2],
                'role': result[3],
                'is_active': result[4],
                'created_at': isoformat(result[5]),
                'last_login': isoformat(result[6])
            }
            
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def change_password(self, 
                        user_id: str, 
                        old_password: str, 
                        new_password: str) -> Dict[str, Any]:
        """
        Change user password.
        
        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password
            
        Returns:
            Status dict
        """
        if not self.db_session:
            return {'success': False, 'error': 'Database session required'}
        
        try:
            # Get current password hash
            query = text("SELECT password_hash FROM users WHERE id = :user_id")
            result = self.db_session.execute(query, {'user_id': user_id}).fetchone()
            
            if not result:
                return {'success': False, 'error': 'User not found'}
            
            # Verify old password
            if not self.verify_password(old_password, result[0]):
                return {'success': False, 'error': 'Current password is incorrect'}
            
            # Hash new password
            new_hash = self.hash_password(new_password)
            
            # Update password
            update_query = text("""
                UPDATE users SET password_hash = :password_hash WHERE id = :user_id
            """)
            self.db_session.execute(
                update_query,
                {'password_hash': new_hash, 'user_id': user_id}
            )
            self.db_session.commit()
            
            logger.info(f"Password changed for user: {user_id}")
            
            return {'success': True, 'message': 'Password changed successfully'}
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Password change error: {e}")
            return {'success': False, 'error': str(e)}
    
    def admin_create_user(self,
                          username: str,
                          email: str,
                          password: str,
                          role: str = 'user') -> Dict[str, Any]:
        """
        Create a user with admin privileges (for initial setup).
        
        Args:
            username: Username
            email: Email
            password: Password
            role: User role
            
        Returns:
            Status dict
        """
        if not self.db_session:
            return {'success': False, 'error': 'Database session required'}
        
        # Check if any users exist (for initial admin setup)
        count_query = text("SELECT COUNT(*) FROM users")
        count = self.db_session.execute(count_query).scalar()
        
        # First user becomes admin
        if count == 0:
            role = 'admin'
            logger.info("Creating initial admin user")
        
        return self.register_user(username, email, password, role)


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Utility function to verify a token.
    
    Args:
        token: JWT token
        
    Returns:
        Token payload or None
    """
    auth_service = AuthService()
    is_valid, result = auth_service.validate_token(token)
    
    if is_valid:
        return result
    return None


# Singleton instance
auth_service = AuthService()


if __name__ == "__main__":
    # Test authentication
    auth = AuthService()
    
    # Test password hashing
    password = "testpassword123"
    hashed = auth.hash_password(password)
    print(f"Password hashed successfully: {len(hashed)} characters")
    
    # Test password verification
    is_valid = auth.verify_password(password, hashed)
    print(f"Password verification: {is_valid}")
    
    # Test token creation
    token = auth.create_access_token(
        user_id="test-uuid",
        username="testuser",
        role="admin"
    )
    print(f"Token created: {token[:50]}...")
    
    # Test token validation
    payload = auth.decode_token(token)
    print(f"Token payload: {payload}")