"""
SQLAlchemy Models for Airline No-Show Prediction System
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    ForeignKey, Text, JSON, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from database.connection import Base


class Passenger(Base):
    """Passenger information model."""
    __tablename__ = 'passengers'
    
    id = Column('passenger_id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    passenger_code = Column(String(20), unique=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255))
    phone = Column(String(20))
    loyalty_tier = Column(String(20), default='STANDARD')
    total_bookings = Column(Integer, default=0)
    total_noshows = Column(Integer, default=0)
    total_cancellations = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    bookings = relationship("Booking", back_populates="passenger")
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'passenger_code': self.passenger_code,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'email': self.email,
            'loyalty_tier': self.loyalty_tier,
            'total_bookings': self.total_bookings,
            'total_noshows': self.total_noshows
        }


class Flight(Base):
    """Flight information model."""
    __tablename__ = 'flights'
    
    id = Column('flight_id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    flight_code = Column(String(20), unique=True, nullable=False)
    departure_airport = Column(String(10), nullable=False)
    arrival_airport = Column(String(10), nullable=False)
    route = Column(String(20))
    departure_time = Column(DateTime, nullable=False)
    arrival_time = Column(DateTime, nullable=False)
    capacity = Column(Integer, nullable=False, default=180)
    base_price = Column(Float, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    bookings = relationship("Booking", back_populates="flight")
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'flight_code': self.flight_code,
            'route': self.route,
            'departure_airport': self.departure_airport,
            'arrival_airport': self.arrival_airport,
            'capacity': self.capacity,
            'base_price': self.base_price
        }


class Booking(Base):
    """Booking information model."""
    __tablename__ = 'bookings'
    
    id = Column('booking_id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    booking_code = Column(String(20), unique=True, nullable=False)
    passenger_id = Column(UUID(as_uuid=True), ForeignKey('passengers.passenger_id'), nullable=True)
    flight_id = Column(UUID(as_uuid=True), ForeignKey('flights.flight_id'), nullable=True)
    
    booking_date = Column(DateTime, nullable=False)
    booking_channel = Column(String(50), default='WEB')
    booking_lead_time = Column(Integer)
    ticket_price = Column(Float, nullable=False)
    cabin_class = Column(String(20), default='ECONOMY')
    booking_status = Column(String(20), default='CONFIRMED')
    
    noshow_probability = Column(Float)
    noshow_confirmed = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    passenger = relationship("Passenger", back_populates="bookings")
    flight = relationship("Flight", back_populates="bookings")
    predictions = relationship("Prediction", back_populates="booking")
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'booking_code': self.booking_code,
            'booking_channel': self.booking_channel,
            'ticket_price': self.ticket_price,
            'cabin_class': self.cabin_class,
            'booking_status': self.booking_status,
            'noshow_probability': self.noshow_probability
        }


class Prediction(Base):
    """Model prediction storage."""
    __tablename__ = 'predictions'
    
    id = Column('prediction_id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    booking_id = Column(UUID(as_uuid=True), ForeignKey('bookings.booking_id'), nullable=True)
    
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    
    noshow_probability = Column(Float, nullable=False)
    prediction_class = Column(Integer, nullable=False)
    
    feature_values = Column(JSON)
    shap_explanation = Column(JSON)
    risk_category = Column(String(20))
    
    prediction_timestamp = Column(DateTime, default=datetime.utcnow)
    model_run_time_ms = Column(Integer)
    is_active = Column(Boolean, default=True)
    
    booking = relationship("Booking", back_populates="predictions")
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'booking_id': str(self.booking_id) if self.booking_id else None,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'noshow_probability': self.noshow_probability,
            'prediction_class': self.prediction_class,
            'risk_category': self.risk_category,
            'prediction_timestamp': self.prediction_timestamp.isoformat() if self.prediction_timestamp else None
        }


class ModelPerformance(Base):
    """Model performance metrics storage."""
    __tablename__ = 'model_performance'
    
    id = Column('performance_id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    
    evaluation_date = Column(DateTime, default=datetime.utcnow)
    dataset_type = Column(String(20), default='TEST')  # TRAIN, VALIDATION, TEST, PRODUCTION
    
    accuracy = Column(Float)
    precision = Column('precision_score', Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    pr_auc = Column(Float)
    specificity = Column(Float)
    
    confusion_matrix = Column(JSON)
    calibration_data = Column(JSON)
    
    train_samples = Column(Integer)
    test_samples = Column(Integer)
    
    hyperparameters = Column('hyperparams', JSON)
    feature_importance = Column(JSON)
    notes = Column(Text)
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'evaluation_date': self.evaluation_date.isoformat() if self.evaluation_date else None,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'pr_auc': self.pr_auc
        }


class User(Base):
    """User authentication model."""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default='user')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'is_active': self.is_active
        }


class MonitoringLog(Base):
    """Model monitoring and drift detection logs."""
    __tablename__ = 'monitoring_logs'
    
    id = Column('log_id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    log_type = Column(String(50), nullable=False)  # DRIFT, PERFORMANCE, ALERT
    severity = Column(String(20), default='INFO')  # INFO, WARNING, CRITICAL
    
    metric_name = Column(String(100))
    reference_value = Column(Float)
    current_value = Column(Float)
    change_percent = Column(Float)
    drift_detected = Column(Boolean, default=False)
    
    details = Column(JSON)
    recommendations = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'log_type': self.log_type,
            'severity': self.severity,
            'metric_name': self.metric_name,
            'drift_detected': self.drift_detected,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


def create_tables():
    """Create all tables in the database."""
    from backend.app.database.connection import db, engine
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    from backend.app.database.connection import engine
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")