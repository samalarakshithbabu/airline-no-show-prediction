"""
Analytics service - provides business intelligence and analytics.
Supports both SQLite (development) and PostgreSQL (production).
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import text, create_engine
from sqlalchemy.orm import Session
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class AnalyticsService:
    """
    Service for analytics and business intelligence.
    """
    
    def __init__(self, db: Session = None):
        """
        Initialize analytics service.
        
        Args:
            db: Database session (optional for demo mode)
        """
        self.db = db
        self.demo_mode = db is None
        self._db_type = None
    
    def _get_db_type(self) -> str:
        """Detect database type from connection."""
        if self._db_type:
            return self._db_type
        
        if self.db is None:
            return 'none'
        
        try:
            # Check if SQLite by trying to get version
            result = self.db.execute(text("SELECT sqlite_version()")).fetchone()
            if result:
                self._db_type = 'sqlite'
                return 'sqlite'
        except:
            pass
        
        try:
            result = self.db.execute(text("SELECT version()")).fetchone()
            if 'PostgreSQL' in str(result):
                self._db_type = 'postgresql'
                return 'postgresql'
        except:
            pass
        
        return 'unknown'
    
    def get_executive_dashboard(self, days: int = 30) -> Dict[str, Any]:
        """
        Get executive dashboard metrics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dashboard metrics dictionary with actual data or empty state
        """
        if self.demo_mode:
            # Return informative message instead of fabricated data
            return {
                'period_days': days,
                'total_bookings': 0,
                'total_flights': 0,
                'predicted_noshows': 0,
                'actual_noshows': 0,
                'noshow_rate': 0,
                'avg_noshow_probability': 0,
                'total_revenue': 0,
                'avg_booking_lead_days': 0,
                'unique_passengers': 0,
                'status': 'demo_mode',
                'message': 'No database connection. Connect to a database for actual analytics.'
            }
        
        start_date = datetime.utcnow() - timedelta(days=days)
        db_type = self._get_db_type()
        
        try:
            if db_type == 'sqlite':
                # SQLite-compatible query (no FILTER, no ::numeric)
                query = text("""
                    SELECT 
                        COUNT(*) AS total_bookings,
                        COUNT(DISTINCT flight_id) AS total_flights,
                        SUM(CASE WHEN noshow_probability > 0.5 THEN 1 ELSE 0 END) AS predicted_noshows,
                        SUM(CASE WHEN booking_status = 'NOSHOW' THEN 1 ELSE 0 END) AS actual_noshows,
                        AVG(noshow_probability) * 100 AS avg_noshow_probability,
                        SUM(ticket_price) AS total_revenue,
                        AVG(booking_lead_time) AS avg_booking_lead_days,
                        COUNT(DISTINCT passenger_id) AS unique_passengers
                    FROM bookings
                    WHERE booking_date >= :start_date
                """)
            else:
                # PostgreSQL query
                query = text("""
                    SELECT 
                        COUNT(*) AS total_bookings,
                        COUNT(DISTINCT flight_id) AS total_flights,
                        COUNT(*) FILTER (WHERE noshow_probability > 0.5) AS predicted_noshows,
                        COUNT(*) FILTER (WHERE booking_status = 'NOSHOW') AS actual_noshows,
                        ROUND(AVG(noshow_probability)::numeric * 100, 2) AS avg_noshow_probability,
                        ROUND(SUM(ticket_price)::numeric, 2) AS total_revenue,
                        ROUND(AVG(booking_lead_time)::numeric, 1) AS avg_booking_lead_days,
                        COUNT(DISTINCT passenger_id) AS unique_passengers
                    FROM bookings
                    WHERE booking_date >= :start_date
                """)
            
            result = self.db.execute(query, {'start_date': start_date}).fetchone()
            
            if result and result[0] and result[0] > 0:
                total_bookings = result[0] or 0
                actual_noshows = result[3] or 0
                
                return {
                    'period_days': days,
                    'total_bookings': total_bookings,
                    'total_flights': result[1] or 0,
                    'predicted_noshows': result[2] or 0,
                    'actual_noshows': actual_noshows,
                    'noshow_rate': round(actual_noshows / total_bookings * 100, 2) if total_bookings > 0 else 0,
                    'avg_noshow_probability': float(result[4] or 0),
                    'total_revenue': float(result[5] or 0),
                    'avg_booking_lead_days': round(float(result[6] or 0), 1),
                    'unique_passengers': result[7] or 0,
                    'status': 'success',
                    'db_type': db_type
                }
            else:
                # No data in database
                return {
                    'period_days': days,
                    'total_bookings': 0,
                    'total_flights': 0,
                    'predicted_noshows': 0,
                    'actual_noshows': 0,
                    'noshow_rate': 0,
                    'avg_noshow_probability': 0,
                    'total_revenue': 0,
                    'avg_booking_lead_days': 0,
                    'unique_passengers': 0,
                    'status': 'no_data',
                    'message': 'No booking data found in the database. Add data to see analytics.',
                    'db_type': db_type
                }
                
        except Exception as e:
            logger.warning(f"Database query failed: {e}")
            return {
                'period_days': days,
                'total_bookings': 0,
                'total_flights': 0,
                'predicted_noshows': 0,
                'actual_noshows': 0,
                'noshow_rate': 0,
                'avg_noshow_probability': 0,
                'total_revenue': 0,
                'avg_booking_lead_days': 0,
                'unique_passengers': 0,
                'status': 'error',
                'message': str(e),
                'db_type': db_type
            }
    
    def get_route_analysis(self, route: str = None) -> List[Dict[str, Any]]:
        """
        Get route-level analytics.
        
        Args:
            route: Optional route filter
            
        Returns:
            List of route statistics
        """
        if self.demo_mode:
            return []
        
        db_type = self._get_db_type()
        
        try:
            if db_type == 'sqlite':
                if route:
                    query = text("""
                        SELECT 
                            f.route,
                            f.departure_airport,
                            f.arrival_airport,
                            COUNT(DISTINCT f.flight_id) AS total_flights,
                            COUNT(b.booking_id) AS total_bookings,
                            ROUND(COUNT(b.booking_id) * CASE WHEN COUNT(b.booking_id) > 0 THEN 100.0 * SUM(CASE WHEN b.booking_status = 'NOSHOW' THEN 1 ELSE 0 END) / COUNT(b.booking_id) ELSE 0 END, 2) AS noshow_rate,
                            ROUND(AVG(b.booking_lead_time), 1) AS avg_booking_lead,
                            ROUND(AVG(b.ticket_price), 2) AS avg_ticket_price
                        FROM flights f
                        LEFT JOIN bookings b ON f.flight_id = b.flight_id
                        WHERE f.route = :route
                        GROUP BY f.route, f.departure_airport, f.arrival_airport
                    """)
                    results = self.db.execute(query, {'route': route}).fetchall()
                else:
                    query = text("""
                        SELECT 
                            f.route,
                            f.departure_airport,
                            f.arrival_airport,
                            COUNT(DISTINCT f.flight_id) AS total_flights,
                            COUNT(b.booking_id) AS total_bookings,
                            ROUND(COUNT(b.booking_id) * CASE WHEN COUNT(b.booking_id) > 0 THEN 100.0 * SUM(CASE WHEN b.booking_status = 'NOSHOW' THEN 1 ELSE 0 END) / COUNT(b.booking_id) ELSE 0 END, 2) AS noshow_rate,
                            ROUND(AVG(b.booking_lead_time), 1) AS avg_booking_lead,
                            ROUND(AVG(b.ticket_price), 2) AS avg_ticket_price
                        FROM flights f
                        LEFT JOIN bookings b ON f.flight_id = b.flight_id
                        GROUP BY f.route, f.departure_airport, f.arrival_airport
                        ORDER BY total_bookings DESC
                    """)
                    results = self.db.execute(query).fetchall()
            else:
                # PostgreSQL
                if route:
                    query = text("""
                        SELECT 
                            f.route,
                            f.departure_airport,
                            f.arrival_airport,
                            COUNT(DISTINCT f.flight_id) AS total_flights,
                            COUNT(b.booking_id) AS total_bookings,
                            ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::numeric / NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS noshow_rate,
                            ROUND(AVG(b.booking_lead_time)::numeric, 1) AS avg_booking_lead,
                            ROUND(AVG(b.ticket_price)::numeric, 2) AS avg_ticket_price
                        FROM flights f
                        LEFT JOIN bookings b ON f.flight_id = b.flight_id
                        WHERE f.route = :route
                        GROUP BY f.route, f.departure_airport, f.arrival_airport
                    """)
                    results = self.db.execute(query, {'route': route}).fetchall()
                else:
                    query = text("""
                        SELECT 
                            f.route,
                            f.departure_airport,
                            f.arrival_airport,
                            COUNT(DISTINCT f.flight_id) AS total_flights,
                            COUNT(b.booking_id) AS total_bookings,
                            ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::numeric / NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS noshow_rate,
                            ROUND(AVG(b.booking_lead_time)::numeric, 1) AS avg_booking_lead,
                            ROUND(AVG(b.ticket_price)::numeric, 2) AS avg_ticket_price
                        FROM flights f
                        LEFT JOIN bookings b ON f.flight_id = b.flight_id
                        GROUP BY f.route, f.departure_airport, f.arrival_airport
                        ORDER BY total_bookings DESC
                    """)
                    results = self.db.execute(query).fetchall()
            
            return [
                {
                    'route': r[0],
                    'departure_airport': r[1],
                    'arrival_airport': r[2],
                    'total_flights': r[3],
                    'total_bookings': r[4],
                    'noshow_rate': float(r[5] or 0),
                    'avg_booking_lead_days': float(r[6] or 0),
                    'avg_ticket_price': float(r[7] or 0)
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Database query failed: {e}")
            return []
    
    def get_channel_performance(self) -> List[Dict[str, Any]]:
        """
        Get booking channel performance analysis.
        
        Returns:
            List of channel statistics
        """
        if self.demo_mode:
            return []
        
        db_type = self._get_db_type()
        
        try:
            if db_type == 'sqlite':
                query = text("""
                    SELECT 
                        booking_channel,
                        COUNT(*) AS total_bookings,
                        ROUND(AVG(ticket_price), 2) AS avg_ticket_price,
                        ROUND(COUNT(*) * CASE WHEN COUNT(*) > 0 THEN 100.0 * SUM(CASE WHEN booking_status = 'NOSHOW' THEN 1 ELSE 0 END) / COUNT(*) ELSE 0 END, 2) AS noshow_rate
                    FROM bookings
                    GROUP BY booking_channel
                    ORDER BY total_bookings DESC
                """)
            else:
                query = text("""
                    SELECT 
                        booking_channel,
                        COUNT(*) AS total_bookings,
                        ROUND(AVG(ticket_price)::numeric, 2) AS avg_ticket_price,
                        ROUND(COUNT(*) FILTER (WHERE booking_status = 'NOSHOW')::numeric / COUNT(*) * 100, 2) AS noshow_rate
                    FROM bookings
                    GROUP BY booking_channel
                    ORDER BY total_bookings DESC
                """)
            
            results = self.db.execute(query).fetchall()
            
            return [
                {
                    'channel': r[0],
                    'total_bookings': r[1],
                    'avg_ticket_price': float(r[2] or 0),
                    'noshow_rate': float(r[3] or 0)
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Database query failed: {e}")
            return []
    
    def get_time_series_trends(self, 
                                group_by: str = 'day',
                                days: int = 30) -> List[Dict[str, Any]]:
        """
        Get time-series trends.
        
        Args:
            group_by: 'day', 'week', or 'month'
            days: Number of days to analyze
            
        Returns:
            Time series data
        """
        if self.demo_mode:
            return []
        
        db_type = self._get_db_type()
        if group_by not in {'day', 'week', 'month'}:
            raise ValueError("group_by must be 'day', 'week', or 'month'")
        if days < 1:
            raise ValueError("days must be at least 1")
        
        try:
            if db_type == 'sqlite':
                period_expression = {
                    'day': 'DATE(booking_date)',
                    'week': "DATE(booking_date, 'weekday 0', '-6 days')",
                    'month': "DATE(booking_date, 'start of month')"
                }[group_by]
                query = text(f"""
                SELECT 
                    {period_expression} AS period,
                    COUNT(*) AS total_bookings,
                    SUM(CASE WHEN booking_status = 'NOSHOW' THEN 1 ELSE 0 END) AS noshow_count,
                    ROUND(AVG(booking_lead_time), 1) AS avg_lead_time,
                    COUNT(DISTINCT passenger_id) AS unique_passengers
                FROM bookings
                WHERE booking_date >= :start_date
                GROUP BY {period_expression}
                ORDER BY period
            """)
            else:
                query = text("""
                    SELECT 
                        DATE_TRUNC(:group_by, booking_date) AS period,
                        COUNT(*) AS total_bookings,
                        COUNT(*) FILTER (WHERE booking_status = 'NOSHOW') AS noshow_count,
                        ROUND(AVG(booking_lead_time)::numeric, 1) AS avg_lead_time,
                        COUNT(DISTINCT passenger_id) AS unique_passengers
                    FROM bookings
                    WHERE booking_date >= :start_date
                    GROUP BY DATE_TRUNC(:group_by, booking_date)
                    ORDER BY period
                """)

            params = {
                'start_date': datetime.utcnow() - timedelta(days=days),
            }
            if db_type != 'sqlite':
                params['group_by'] = group_by
            results = self.db.execute(query, params).fetchall()
            
            return [
                {
                    'period': r[0],
                    'total_bookings': r[1],
                    'noshow_count': r[2],
                    'noshow_rate': round((r[2] or 0) / max(r[1] or 1, 1) * 100, 2),
                    'avg_booking_lead_days': float(r[3] or 0),
                    'unique_passengers': r[4]
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Database query failed: {e}")
            raise
    
    def get_high_risk_bookings(self, 
                                 min_probability: float = 0.5,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get high-risk bookings for intervention.
        
        Args:
            min_probability: Minimum no-show probability
            limit: Maximum results
            
        Returns:
            List of high-risk bookings
        """
        if self.demo_mode:
            return []
        
        try:
            query = text("""
                SELECT 
                    b.booking_code,
                    p.first_name || ' ' || p.last_name AS passenger_name,
                    f.route,
                    f.departure_time,
                    b.booking_status,
                    pr.noshow_probability,
                    pr.risk_category,
                    b.booking_channel,
                    b.ticket_price
                FROM bookings b
                JOIN passengers p ON b.passenger_id = p.passenger_id
                JOIN flights f ON b.flight_id = f.flight_id
                JOIN (
                    SELECT booking_id, noshow_probability, risk_category,
                           ROW_NUMBER() OVER (
                               PARTITION BY booking_id
                               ORDER BY prediction_timestamp DESC
                           ) AS row_num
                    FROM predictions
                    WHERE is_active = TRUE
                ) pr ON pr.booking_id = b.booking_id AND pr.row_num = 1
                WHERE pr.noshow_probability >= :min_prob
                ORDER BY pr.noshow_probability DESC
                LIMIT :limit
            """)
            
            results = self.db.execute(query, {
                'min_prob': min_probability,
                'limit': limit
            }).fetchall()
            
            return [
                {
                    'booking_code': r[0],
                    'passenger_name': r[1],
                    'route': r[2],
                    'departure_time': r[3].isoformat() if hasattr(r[3], 'isoformat') else r[3],
                    'booking_status': r[4],
                    'noshow_probability': float(r[5]),
                    'risk_category': r[6],
                    'booking_channel': r[7],
                    'ticket_price': float(r[8])
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Database query failed: {e}")
            raise
    
    def get_model_comparison(self) -> List[Dict[str, Any]]:
        """
        Get model performance comparison from model info files.
        
        Returns:
            List of model performance statistics
        """
        try:
            from config import MODEL_DIR
            model_dir = Path(MODEL_DIR)
            models = []
            
            for path in model_dir.glob('*_info.json'):
                with path.open('r', encoding='utf-8') as f:
                    info = json.load(f)
                    
                metrics = info.get('metrics', {})
                models.append({
                    'model_name': info.get('model_name', path.name.split('_')[0]),
                    'model_version': info.get('version'),
                    'accuracy': metrics.get('accuracy'),
                    'precision': metrics.get('precision'),
                    'recall': metrics.get('recall'),
                    'f1_score': metrics.get('f1_score'),
                    'roc_auc': metrics.get('roc_auc'),
                    'pr_auc': metrics.get('pr_auc'),
                    'trained_at': info.get('trained_at')
                })
            
            return sorted(models, key=lambda x: x.get('roc_auc') or 0, reverse=True)
            
        except Exception as e:
            logger.warning(f"Could not load model comparison: {e}")
            return []
    
    def get_passenger_summary(self, 
                               passenger_id: str = None,
                               limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get passenger summary statistics.
        
        Args:
            passenger_id: Optional specific passenger
            limit: Maximum results
            
        Returns:
            List of passenger summaries
        """
        if self.demo_mode:
            return []
        
        return []