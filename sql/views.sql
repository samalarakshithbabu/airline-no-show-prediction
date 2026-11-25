-- Airline No-Show Prediction System - Predefined SQL Views
-- PostgreSQL Database Views for Analytics Dashboards

-- ============================================
-- DASHBOARD VIEWS
-- ============================================

-- View 1: Executive Dashboard KPI Summary
CREATE OR REPLACE VIEW v_executive_kpis AS
SELECT 
    CURRENT_DATE AS report_date,
    (SELECT COUNT(*) FROM bookings WHERE booking_date >= CURRENT_DATE - 30) AS bookings_last_30_days,
    (SELECT COUNT(DISTINCT flight_id) FROM flights WHERE departure_time >= CURRENT_DATE - 30) AS flights_last_30_days,
    (SELECT COUNT(*) FROM bookings WHERE booking_status = 'NOSHOW' AND booking_date >= CURRENT_DATE - 30) AS noshows_last_30_days,
    (SELECT COUNT(*) FROM bookings WHERE booking_date >= CURRENT_DATE - 30) AS total_bookings,
    ROUND(
        (SELECT COUNT(*) FROM bookings WHERE booking_status = 'NOSHOW' AND booking_date >= CURRENT_DATE - 30)::DECIMAL /
        NULLIF((SELECT COUNT(*) FROM bookings WHERE booking_date >= CURRENT_DATE - 30), 0) * 100, 2
    ) AS noshow_rate,
    (SELECT ROUND(AVG(noshow_probability) * 100, 2) FROM bookings WHERE booking_date >= CURRENT_DATE - 30) AS avg_predicted_prob,
    (SELECT ROUND(SUM(ticket_price), 2) FROM bookings WHERE booking_date >= CURRENT_DATE - 30) AS total_revenue,
    (SELECT ROUND(COUNT(*)::DECIMAL * 0.18, 0) FROM bookings WHERE noshow_probability > 0.5) AS predicted_high_risk
FROM bookings
LIMIT 1;

-- View 2: Real-Time Flight Dashboard
CREATE OR REPLACE VIEW v_flight_dashboard AS
SELECT 
    f.flight_id,
    f.flight_code,
    f.route,
    f.departure_time,
    f.flight_date,
    f.flight_dayofweek,
    f.flight_hour,
    f.capacity,
    COUNT(b.booking_id) AS current_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'CONFIRMED') AS confirmed_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW') AS noshow_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'CANCELLED') AS cancelled_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.check_in_status = TRUE) AS checked_in,
    ROUND(COUNT(b.booking_id)::DECIMAL / f.capacity * 100, 1) AS load_factor,
    ROUND(AVG(b.noshow_probability)::DECIMAL * 100, 2) AS avg_noshow_prob,
    SUM(b.ticket_price) AS projected_revenue,
    CASE 
        WHEN COUNT(b.booking_id) > f.capacity THEN 'OVERBOOKED'
        WHEN COUNT(b.booking_id) >= f.capacity * 0.95 THEN 'SOLD_OUT'
        WHEN COUNT(b.booking_id) >= f.capacity * 0.8 THEN 'HIGH_DEMAND'
        ELSE 'AVAILABLE'
    END AS sales_status,
    COUNT(b.booking_id) FILTER (WHERE b.noshow_probability > 0.5) AS high_risk_bookings
FROM flights f
LEFT JOIN bookings b ON f.flight_id = b.flight_id
GROUP BY f.flight_id;

-- View 3: Passenger Risk Scoring
CREATE OR REPLACE VIEW v_passenger_risk AS
SELECT 
    p.passenger_id,
    p.passenger_code,
    p.first_name,
    p.last_name,
    p.email,
    p.loyalty_tier,
    p.total_bookings,
    p.total_noshows,
    p.total_cancellations,
    ROUND(p.total_noshows::DECIMAL / NULLIF(p.total_bookings, 0) * 100, 2) AS historical_noshow_rate,
    (SELECT COUNT(*) FROM bookings WHERE passenger_id = p.passenger_id AND noshow_probability > 0.5) AS recent_high_risk_predictions,
    (SELECT MAX(booking_date) FROM bookings WHERE passenger_id = p.passenger_id) AS last_booking_date,
    (SELECT COUNT(*) FROM bookings WHERE passenger_id = p.passenger_id AND booking_date > CURRENT_DATE - 90) AS bookings_90_days,
    CASE 
        WHEN p.total_bookings = 0 THEN 'UNKNOWN'
        WHEN p.total_noshows::DECIMAL / p.total_bookings > 0.25 THEN 'CRITICAL'
        WHEN p.total_noshows::DECIMAL / p.total_bookings > 0.15 THEN 'HIGH'
        WHEN p.total_noshows::DECIMAL / p.total_bookings > 0.08 THEN 'MODERATE'
        ELSE 'LOW'
    END AS risk_level,
    (SELECT booking_id FROM bookings WHERE passenger_id = p.passenger_id ORDER BY booking_date DESC LIMIT 1) AS last_booking_id
FROM passengers p;

-- View 4: Revenue Impact Analysis
CREATE OR REPLACE VIEW v_revenue_impact AS
SELECT 
    f.flight_code,
    f.route,
    f.departure_time,
    COUNT(b.booking_id) AS total_bookings,
    SUM(b.ticket_price) AS total_revenue,
    SUM(b.ticket_price) FILTER (WHERE b.booking_status = 'NOSHOW') AS lost_revenue_noshows,
    SUM(b.ticket_price) FILTER (WHERE b.booking_status = 'CANCELLED') AS lost_revenue_cancellations,
    SUM(b.ticket_price) FILTER (WHERE b.noshow_probability > 0.5) AS at_risk_revenue,
    ROUND(SUM(b.ticket_price) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
          NULLIF(SUM(b.ticket_price), 0) * 100, 2) AS revenue_loss_pct,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW') AS noshow_count,
    COUNT(b.booking_id) FILTER (WHERE b.noshow_probability > 0.5) AS predicted_noshows,
    ROUND(AVG(b.noshow_probability)::DECIMAL * 100, 2) AS avg_noshow_prob,
    f.capacity * (SELECT base_price FROM flights WHERE flight_id = f.flight_id LIMIT 1) AS max_revenue_potential,
    ROUND((SUM(b.ticket_price) / (f.capacity * (SELECT base_price FROM flights WHERE flight_id = f.flight_id LIMIT 1))) * 100, 1) AS revenue_realization_pct
FROM flights f
LEFT JOIN bookings b ON f.flight_id = b.flight_id
GROUP BY f.flight_id, f.flight_code, f.route, f.departure_time, f.capacity;

-- ============================================
-- PREDICTION VIEWS
-- ============================================

-- View 5: Active Predictions Summary
CREATE OR REPLACE VIEW v_active_predictions AS
SELECT 
    pr.prediction_id,
    pr.booking_id,
    pr.model_name,
    pr.model_version,
    pr.noshow_probability,
    pr.prediction_class,
    pr.risk_category,
    pr.prediction_timestamp,
    b.booking_code,
    b.booking_status,
    b.flight_id,
    f.flight_code,
    f.route,
    f.departure_time,
    p.passenger_id,
    p.passenger_code,
    p.first_name || ' ' || p.last_name AS passenger_name,
    b.booking_channel,
    b.cabin_class,
    b.ticket_price,
    b.booking_lead_time
FROM predictions pr
JOIN bookings b ON pr.booking_id = b.booking_id
JOIN flights f ON b.flight_id = f.flight_id
JOIN passengers p ON b.passenger_id = p.passenger_id
WHERE pr.is_active = TRUE
ORDER BY pr.noshow_probability DESC;

-- View 6: Prediction Distribution by Risk Category
CREATE OR REPLACE VIEW v_prediction_distribution AS
SELECT 
    risk_category,
    COUNT(*) AS prediction_count,
    ROUND(COUNT(*)::DECIMAL / NULLIF(SUM(COUNT(*)) OVER(), 0) * 100, 2) AS percentage,
    ROUND(AVG(noshow_probability)::DECIMAL * 100, 2) AS avg_probability,
    ROUND(MIN(noshow_probability)::DECIMAL * 100, 2) AS min_probability,
    ROUND(MAX(noshow_probability)::DECIMAL * 100, 2) AS max_probability,
    SUM(CASE WHEN booking_status = 'NOSHOW' THEN 1 ELSE 0 END) AS actual_noshows,
    ROUND(SUM(CASE WHEN booking_status = 'NOSHOW' THEN 1 ELSE 0 END)::DECIMAL / NULLIF(COUNT(*), 0) * 100, 2) AS actual_noshow_rate
FROM v_active_predictions
GROUP BY risk_category
ORDER BY 
    CASE risk_category 
        WHEN 'CRITICAL' THEN 1 
        WHEN 'HIGH' THEN 2 
        WHEN 'ELEVATED' THEN 3 
        WHEN 'MODERATE' THEN 4 
        ELSE 5 
    END;

-- View 7: Model Comparison
CREATE OR REPLACE VIEW v_model_comparison AS
SELECT 
    model_name,
    model_version,
    MAX(evaluation_date) AS latest_evaluation,
    MAX(accuracy) AS best_accuracy,
    MAX(precision_score) AS best_precision,
    MAX(recall_score) AS best_recall,
    MAX(f1_score) AS best_f1,
    MAX(roc_auc) AS best_roc_auc,
    COUNT(*) AS total_evaluations,
    ROUND(AVG(train_samples)::DECIMAL, 0) AS avg_train_samples,
    ROUND(AVG(test_samples)::DECIMAL, 0) AS avg_test_samples
FROM model_performance
GROUP BY model_name, model_version
ORDER BY best_f1 DESC;

-- View 8: Feature Importance Tracking
CREATE OR REPLACE VIEW v_feature_importance_tracking AS
SELECT 
    model_name,
    model_version,
    evaluation_date,
    feature_importance->>'feature1' AS top_feature_1,
    (feature_importance->>'importance1')::DECIMAL AS importance_1,
    feature_importance->>'feature2' AS top_feature_2,
    (feature_importance->>'importance2')::DECIMAL AS importance_2,
    feature_importance->>'feature3' AS top_feature_3,
    (feature_importance->>'importance3')::DECIMAL AS importance_3
FROM model_performance
WHERE feature_importance IS NOT NULL
ORDER BY evaluation_date DESC;

-- ============================================
-- OPERATIONAL VIEWS
-- ============================================

-- View 9: Daily Operations Summary
CREATE OR REPLACE VIEW v_daily_operations AS
SELECT 
    flight_date AS operation_date,
    COUNT(*) AS total_flights,
    SUM(capacity) AS total_capacity,
    SUM(current_bookings) AS total_bookings,
    SUM(high_risk_bookings) AS high_risk_count,
    SUM(checked_in) AS checked_in_count,
    ROUND(SUM(current_bookings)::DECIMAL / SUM(capacity) * 100, 1) AS avg_load_factor,
    ROUND(AVG(avg_noshow_prob)::DECIMAL, 2) AS avg_noshow_prob,
    SUM(CASE WHEN sales_status = 'OVERBOOKED' THEN 1 ELSE 0 END) AS overbooked_flights,
    SUM(CASE WHEN sales_status = 'SOLD_OUT' THEN 1 ELSE 0 END) AS sold_out_flights
FROM v_flight_dashboard
GROUP BY flight_date
ORDER BY operation_date;

-- View 10: Overbooking Risk Assessment
CREATE OR REPLACE VIEW v_overbooking_risk AS
SELECT 
    f.flight_id,
    f.flight_code,
    f.route,
    f.departure_time,
    f.capacity,
    COUNT(b.booking_id) AS current_bookings,
    COUNT(b.booking_id) - f.capacity AS overbooking_count,
    ROUND(AVG(b.noshow_probability)::DECIMAL * 100, 2) AS avg_noshow_prob,
    ROUND(COUNT(b.booking_id) * AVG(b.noshow_probability), 1) AS expected_noshows,
    COUNT(b.booking_id) - ROUND(COUNT(b.booking_id) * AVG(b.noshow_probability), 0) AS expected_shows,
    f.capacity - (COUNT(b.booking_id) - ROUND(COUNT(b.booking_id) * AVG(b.noshow_probability), 0)) AS expected_empty_seats,
    CASE 
        WHEN COUNT(b.booking_id) - ROUND(COUNT(b.booking_id) * AVG(b.noshow_probability), 0) > f.capacity 
        THEN 'DENIED_BOARDING_RISK'
        WHEN COUNT(b.booking_id) - f.capacity > 0 
        THEN 'OVERBOOKED'
        ELSE 'CAPACITY_OK'
    END AS capacity_status,
    ROUND(
        (COUNT(b.booking_id) - f.capacity + COUNT(b.booking_id) * AVG(b.noshow_probability))::DECIMAL / NULLIF(COUNT(b.booking_id), 0) * 100, 
        2
    ) AS denied_boarding_risk_pct
FROM flights f
LEFT JOIN bookings b ON f.flight_id = b.flight_id
GROUP BY f.flight_id, f.flight_code, f.route, f.departure_time, f.capacity
HAVING COUNT(b.booking_id) > f.capacity OR AVG(b.noshow_probability) > 0.2
ORDER BY denied_boarding_risk_pct DESC;

-- View 11: Channel Performance Comparison
CREATE OR REPLACE VIEW v_channel_comparison AS
SELECT 
    booking_channel,
    COUNT(*) AS total_bookings,
    COUNT(DISTINCT passenger_id) AS unique_passengers,
    ROUND(AVG(booking_lead_time)::DECIMAL, 1) AS avg_booking_lead,
    ROUND(AVG(ticket_price)::DECIMAL, 2) AS avg_ticket_price,
    SUM(ticket_price) AS total_revenue,
    ROUND(COUNT(*) FILTER (WHERE booking_status = 'NOSHOW')::DECIMAL / COUNT(*) * 100, 2) AS noshow_rate,
    ROUND(COUNT(*) FILTER (WHERE noshow_probability > 0.5)::DECIMAL / COUNT(*) * 100, 2) AS predicted_noshow_pct,
    COUNT(*) FILTER (WHERE booking_status = 'COMPLETED') AS completed_trips,
    COUNT(*) FILTER (WHERE booking_status = 'CANCELLED') AS cancellations
FROM bookings
GROUP BY booking_channel
ORDER BY noshow_rate;

-- View 12: Time-Based Pattern Analysis
CREATE OR REPLACE VIEW v_time_patterns AS
SELECT 
    EXTRACT(HOUR FROM departure_time)::INTEGER AS hour_of_day,
    EXTRACT(DOW FROM departure_time)::INTEGER AS day_of_week,
    TO_CHAR(departure_time, 'Day') AS day_name,
    COUNT(DISTINCT flight_id) AS flights_scheduled,
    COUNT(*) AS total_bookings,
    ROUND(COUNT(*) FILTER (WHERE booking_status = 'NOSHOW')::DECIMAL / COUNT(*) * 100, 2) AS noshow_rate,
    ROUND(AVG(booking_lead_time)::DECIMAL, 1) AS avg_booking_lead,
    ROUND(AVG(noshow_probability)::DECIMAL * 100, 2) AS avg_predicted_prob
FROM bookings b
JOIN flights f ON b.flight_id = f.flight_id
GROUP BY EXTRACT(HOUR FROM departure_time), EXTRACT(DOW FROM departure_time), TO_CHAR(departure_time, 'Day')
ORDER BY hour_of_day, day_of_week;

-- ============================================
-- DRIFT MONITORING VIEWS
-- ============================================

-- View 13: Data Drift Indicators
CREATE OR REPLACE VIEW v_drift_indicators AS
SELECT 
    'Prediction Probability' AS metric_type,
    AVG(noshow_probability) AS current_mean,
    STDDEV(noshow_probability) AS current_std,
    (SELECT AVG(noshow_probability) FROM predictions WHERE prediction_timestamp < CURRENT_DATE - 7) AS reference_mean,
    (SELECT STDDEV(noshow_probability) FROM predictions WHERE prediction_timestamp < CURRENT_DATE - 7) AS reference_std,
    AVG(noshow_probability) - (SELECT AVG(noshow_probability) FROM predictions WHERE prediction_timestamp < CURRENT_DATE - 7) AS mean_shift,
    COUNT(*) AS sample_count,
    CURRENT_DATE AS check_date
FROM predictions
WHERE prediction_timestamp >= CURRENT_DATE - 7;

-- View 14: Performance Degradation Alert
CREATE OR REPLACE VIEW v_performance_alerts AS
SELECT 
    model_name,
    model_version,
    MAX(evaluation_date) AS last_evaluation,
    MAX(accuracy) AS reported_accuracy,
    MAX(f1_score) AS reported_f1,
    (SELECT MAX(accuracy) FROM model_performance mp2 
     WHERE mp2.model_name = mp.model_name AND mp2.model_version = mp.model_version
     AND mp2.evaluation_date < mp.evaluation_date - INTERVAL '7 days') AS accuracy_7_days_ago,
    (SELECT MAX(f1_score) FROM model_performance mp2 
     WHERE mp2.model_name = mp.model_name AND mp2.model_version = mp.model_version
     AND mp2.evaluation_date < mp.evaluation_date - INTERVAL '7 days') AS f1_7_days_ago,
    CASE 
        WHEN (SELECT MAX(accuracy) FROM model_performance mp2 
              WHERE mp2.model_name = mp.model_name AND mp2.model_version = mp.model_version
              AND mp2.evaluation_date < mp.evaluation_date - INTERVAL '7 days') IS NULL THEN 'NEW_MODEL'
        WHEN MAX(accuracy) < (SELECT MAX(accuracy) FROM model_performance mp2 
              WHERE mp2.model_name = mp.model_name AND mp2.model_version = mp.model_version
              AND mp2.evaluation_date < mp.evaluation_date - INTERVAL '7 days') * 0.95 THEN 'ACCURACY_DROP'
        WHEN MAX(f1_score) < (SELECT MAX(f1_score) FROM model_performance mp2 
              WHERE mp2.model_name = mp.model_name AND mp2.model_version = mp.model_version
              AND mp2.evaluation_date < mp.evaluation_date - INTERVAL '7 days') * 0.90 THEN 'F1_DROP'
        ELSE 'STABLE'
    END AS alert_status
FROM model_performance mp
GROUP BY model_name, model_version;

-- ============================================
-- REPORTING VIEWS
-- ============================================

-- View 15: Monthly Summary Report
CREATE OR REPLACE VIEW v_monthly_report AS
SELECT 
    DATE_TRUNC('month', b.booking_date)::DATE AS report_month,
    COUNT(DISTINCT f.flight_id) AS total_flights,
    COUNT(b.booking_id) AS total_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW') AS noshows,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'CANCELLED') AS cancellations,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'COMPLETED') AS completed,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / COUNT(b.booking_id) * 100, 2) AS noshow_rate,
    ROUND(AVG(b.ticket_price)::DECIMAL, 2) AS avg_ticket_price,
    ROUND(SUM(b.ticket_price)::DECIMAL, 2) AS total_revenue,
    ROUND(SUM(b.ticket_price) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL, 2) AS lost_revenue_noshows,
    ROUND(COUNT(DISTINCT b.passenger_id)::DECIMAL, 0) AS unique_passengers,
    ROUND(AVG(b.booking_lead_time)::DECIMAL, 1) AS avg_booking_lead_days,
    COUNT(b.booking_id) FILTER (WHERE b.booking_channel = 'MOBILE') AS mobile_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.booking_channel = 'WEB') AS web_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.booking_channel = 'AGENT') AS agent_bookings
FROM bookings b
JOIN flights f ON b.flight_id = f.flight_id
GROUP BY DATE_TRUNC('month', b.booking_date)::DATE
ORDER BY report_month DESC;

-- View 16: Route Performance Leaderboard
CREATE OR REPLACE VIEW v_route_leaderboard AS
SELECT 
    f.route,
    f.departure_airport,
    f.arrival_airport,
    COUNT(DISTINCT f.flight_id) AS flights_operated,
    COUNT(b.booking_id) AS bookings_processed,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / COUNT(b.booking_id) * 100, 2) AS noshow_rate,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL, 1) AS total_noshows,
    ROUND(AVG(b.ticket_price)::DECIMAL, 2) AS avg_ticket_price,
    ROUND(SUM(b.ticket_price)::DECIMAL, 2) AS route_revenue,
    ROUND(AVG(b.noshow_probability)::DECIMAL * 100, 2) AS avg_predicted_prob,
    COUNT(DISTINCT b.passenger_id) AS unique_passengers_served,
    COUNT(DISTINCT b.booking_channel) AS channels_used,
    RANK() OVER (ORDER BY 
        COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / COUNT(b.booking_id) * 100 ASC
    ) AS noshow_rate_rank
FROM flights f
LEFT JOIN bookings b ON f.flight_id = b.flight_id
GROUP BY f.route, f.departure_airport, f.arrival_airport
ORDER BY noshow_rate_rank;

-- View 17: Cabin Class Profitability
CREATE OR REPLACE VIEW v_cabin_profitability AS
SELECT 
    cabin_class,
    COUNT(*) AS bookings,
    ROUND(COUNT(*)::DECIMAL / SUM(COUNT(*)) OVER() * 100, 2) AS revenue_share,
    ROUND(AVG(ticket_price)::DECIMAL, 2) AS avg_price,
    MIN(ticket_price) AS min_price,
    MAX(ticket_price) AS max_price,
    ROUND(SUM(ticket_price)::DECIMAL, 2) AS total_revenue,
    COUNT(*) FILTER (WHERE booking_status = 'NOSHOW') AS noshows,
    ROUND(COUNT(*) FILTER (WHERE booking_status = 'NOSHOW')::DECIMAL / COUNT(*) * 100, 2) AS noshow_rate,
    ROUND(SUM(ticket_price) FILTER (WHERE booking_status = 'NOSHOW')::DECIMAL, 2) AS lost_revenue,
    ROUND(SUM(ticket_price) FILTER (WHERE booking_status = 'NOSHOW')::DECIMAL / SUM(ticket_price) * 100, 2) AS loss_percentage
FROM bookings
GROUP BY cabin_class
ORDER BY total_revenue DESC;

-- View 18: Customer Lifetime Value Estimation
CREATE OR REPLACE VIEW v_customer_ltv AS
SELECT 
    p.passenger_id,
    p.passenger_code,
    p.first_name,
    p.last_name,
    p.loyalty_tier,
    COUNT(b.booking_id) AS total_trips,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'COMPLETED') AS completed_trips,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW') AS noshow_trips,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'COMPLETED')::DECIMAL / NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS completion_rate,
    ROUND(SUM(b.ticket_price)::DECIMAL, 2) AS lifetime_value,
    ROUND(AVG(b.ticket_price)::DECIMAL, 2) AS avg_ticket_value,
    ROUND(SUM(b.ticket_price) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL, 2) AS lost_value_noshows,
    MAX(b.booking_date) AS last_activity,
    ROUND(AVG(b.noshow_probability)::DECIMAL * 100, 2) AS avg_noshow_prob,
    CASE 
        WHEN SUM(b.ticket_price) > 5000 THEN 'HIGH_VALUE'
        WHEN SUM(b.ticket_price) > 2000 THEN 'MEDIUM_VALUE'
        ELSE 'STANDARD'
    END AS customer_tier,
    CASE 
        WHEN COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / NULLIF(COUNT(b.booking_id), 0) > 0.2 
        THEN 'AT_RISK'
        ELSE 'HEALTHY'
    END AS retention_status
FROM passengers p
LEFT JOIN bookings b ON p.passenger_id = b.passenger_id
GROUP BY p.passenger_id
ORDER BY lifetime_value DESC NULLS LAST;