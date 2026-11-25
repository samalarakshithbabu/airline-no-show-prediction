-- Airline No-Show Prediction System - Analytics Queries
-- PostgreSQL Database Analytics

-- ============================================
-- EXECUTIVE DASHBOARD QUERIES
-- ============================================

-- Query 1: Executive Dashboard Summary
-- Returns key metrics for the executive dashboard
CREATE OR REPLACE FUNCTION fn_executive_dashboard_summary(p_days INTEGER DEFAULT 30)
RETURNS TABLE (
    total_bookings BIGINT,
    total_flights INTEGER,
    predicted_noshows BIGINT,
    actual_noshows BIGINT,
    noshow_rate DECIMAL,
    avg_noshow_probability DECIMAL,
    total_revenue DECIMAL,
    at_risk_revenue DECIMAL
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(b.booking_id)::BIGINT AS total_bookings,
        COUNT(DISTINCT b.flight_id)::INTEGER AS total_flights,
        COUNT(b.booking_id) FILTER (WHERE b.noshow_probability > 0.5)::BIGINT AS predicted_noshows,
        COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::BIGINT AS actual_noshows,
        ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
              NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS noshow_rate,
        ROUND(AVG(b.noshow_probability)::DECIMAL * 100, 2) AS avg_noshow_probability,
        ROUND(SUM(b.ticket_price)::DECIMAL, 2) AS total_revenue,
        ROUND(SUM(b.ticket_price) FILTER (WHERE b.noshow_probability > 0.5)::DECIMAL, 2) AS at_risk_revenue
    FROM bookings b
    WHERE b.booking_date >= CURRENT_DATE - p_days;
END;
$$;

-- ============================================
-- PASSENGER ANALYTICS
-- ============================================

-- Query 2: Passenger No-Show History Analysis
-- Analyzes individual passenger booking patterns
CREATE OR REPLACE VIEW v_passenger_noshow_analysis AS
SELECT 
    p.passenger_id,
    p.passenger_code,
    p.first_name,
    p.last_name,
    p.loyalty_tier,
    COUNT(b.booking_id) AS total_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW') AS noshow_count,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'CANCELLED') AS cancel_count,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
          NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS noshow_percentage,
    ROUND(AVG(b.booking_lead_time)::DECIMAL, 1) AS avg_booking_lead_days,
    STRING_AGG(DISTINCT b.cabin_class, ', ') AS cabin_classes_booked,
    COUNT(DISTINCT b.booking_channel) AS channels_used,
    MAX(b.booking_date) AS last_booking,
    CASE 
        WHEN COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
             NULLIF(COUNT(b.booking_id), 0) > 0.25 THEN 'HIGH_RISK'
        WHEN COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
             NULLIF(COUNT(b.booking_id), 0) > 0.15 THEN 'MEDIUM_RISK'
        ELSE 'LOW_RISK'
    END AS risk_classification
FROM passengers p
LEFT JOIN bookings b ON p.passenger_id = b.passenger_id
GROUP BY p.passenger_id;

-- Query 3: Frequent No-Show Passengers
-- Identifies passengers with highest no-show rates
CREATE OR REPLACE VIEW v_frequent_noshows AS
SELECT 
    passenger_code,
    first_name,
    last_name,
    loyalty_tier,
    total_bookings,
    noshow_count,
    noshow_percentage
FROM v_passenger_noshow_analysis
WHERE total_bookings >= 5 AND noshow_percentage >= 20
ORDER BY noshow_percentage DESC, noshow_count DESC
LIMIT 50;

-- ============================================
-- FLIGHT AND ROUTE ANALYTICS
-- ============================================

-- Query 4: Flight-Level No-Show Analysis
CREATE OR REPLACE VIEW v_flight_noshow_analysis AS
SELECT 
    f.flight_id,
    f.flight_code,
    f.route,
    f.departure_time,
    f.capacity,
    COUNT(b.booking_id) AS bookings_count,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW') AS noshow_count,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'CANCELLED') AS cancel_count,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'COMPLETED') AS completed_count,
    COUNT(b.booking_id) FILTER (WHERE b.check_in_status = TRUE) AS checked_in_count,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
          NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS noshow_rate,
    ROUND(AVG(b.noshow_probability)::DECIMAL * 100, 2) AS avg_predicted_prob,
    ROUND(AVG(b.ticket_price)::DECIMAL, 2) AS avg_ticket_price,
    SUM(b.ticket_price) AS total_revenue,
    CASE 
        WHEN COUNT(b.booking_id) > f.capacity THEN 'OVERBOOKED'
        WHEN COUNT(b.booking_id) >= f.capacity * 0.9 THEN 'NEAR_CAPACITY'
        ELSE 'AVAILABLE'
    END AS capacity_status
FROM flights f
LEFT JOIN bookings b ON f.flight_id = b.flight_id
GROUP BY f.flight_id;

-- Query 5: Route Deep Dive Analysis
CREATE OR REPLACE FUNCTION fn_route_analysis(p_route VARCHAR(20))
RETURNS TABLE (
    route VARCHAR(20),
    total_flights INTEGER,
    total_bookings BIGINT,
    noshow_rate DECIMAL,
    avg_booking_lead DECIMAL,
    avg_ticket_price DECIMAL,
    peak_noshow_day INTEGER,
    peak_noshow_hour INTEGER,
    best_channel VARCHAR(50),
    worst_channel VARCHAR(50),
    revenue_at_risk DECIMAL
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        f.route,
        COUNT(DISTINCT f.flight_id)::INTEGER AS total_flights,
        COUNT(b.booking_id)::BIGINT AS total_bookings,
        ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
              NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS noshow_rate,
        ROUND(AVG(b.booking_lead_time)::DECIMAL, 1) AS avg_booking_lead,
        ROUND(AVG(b.ticket_price)::DECIMAL, 2) AS avg_ticket_price,
        EXTRACT(DOW FROM f.departure_time)::INTEGER AS peak_noshow_day,
        EXTRACT(HOUR FROM f.departure_time)::INTEGER AS peak_noshow_hour,
        (
            SELECT b2.booking_channel 
            FROM bookings b2 
            JOIN flights f2 ON b2.flight_id = f2.flight_id
            WHERE f2.route = p_route
            GROUP BY b2.booking_channel
            ORDER BY COUNT(*) DESC
            LIMIT 1
        ) AS best_channel,
        (
            SELECT b2.booking_channel 
            FROM bookings b2 
            JOIN flights f2 ON b2.flight_id = f2.flight_id
            WHERE f2.route = p_route
            GROUP BY b2.booking_channel
            ORDER BY COUNT(b2.booking_id) FILTER (WHERE b2.booking_status = 'NOSHOW')::DECIMAL / 
                     NULLIF(COUNT(b2.booking_id), 0) DESC
            LIMIT 1
        ) AS worst_channel,
        ROUND(SUM(b.ticket_price) FILTER (WHERE b.noshow_probability > 0.5)::DECIMAL, 2) AS revenue_at_risk
    FROM flights f
    LEFT JOIN bookings b ON f.flight_id = b.flight_id
    WHERE f.route = p_route
    GROUP BY f.route, peak_noshow_day, peak_noshow_hour;
END;
$$;

-- ============================================
-- TEMPORAL ANALYTICS
-- ============================================

-- Query 6: Time-Series No-Show Trends
CREATE OR REPLACE VIEW v_noshow_timeseries AS
SELECT 
    DATE_TRUNC('day', b.booking_date)::DATE AS date,
    EXTRACT(DOW FROM b.booking_date)::INTEGER AS day_of_week,
    TO_CHAR(b.booking_date, 'Day') AS day_name,
    COUNT(b.booking_id) AS total_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW') AS noshow_count,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
          NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS noshow_rate,
    ROUND(AVG(b.booking_lead_time)::DECIMAL, 1) AS avg_lead_time,
    ROUND(AVG(b.noshow_probability)::DECIMAL * 100, 2) AS avg_predicted_prob,
    COUNT(DISTINCT b.passenger_id) AS unique_passengers,
    COUNT(DISTINCT b.booking_channel) AS channels_active
FROM bookings b
GROUP BY DATE_TRUNC('day', b.booking_date)::DATE, 
         EXTRACT(DOW FROM b.booking_date), 
         TO_CHAR(b.booking_date, 'Day')
ORDER BY date;

-- Query 7: Weekly No-Show Pattern
CREATE OR REPLACE VIEW v_weekly_pattern AS
SELECT 
    day_of_week,
    day_name,
    COUNT(*) AS total_flights,
    ROUND(AVG(noshow_rate)::DECIMAL, 2) AS avg_noshow_rate,
    ROUND(AVG(avg_ticket_price)::DECIMAL, 2) AS avg_ticket_price,
    COUNT(*) FILTER (WHERE noshow_rate > 15) AS high_noshow_days
FROM v_dow_analysis
GROUP BY day_of_week, day_name
ORDER BY day_of_week;

-- Query 8: Hourly No-Show Distribution
CREATE OR REPLACE VIEW v_hourly_analysis AS
SELECT 
    EXTRACT(HOUR FROM f.departure_time)::INTEGER AS departure_hour,
    COUNT(DISTINCT f.flight_id) AS total_flights,
    COUNT(b.booking_id) AS total_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW') AS noshow_count,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
          NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS noshow_rate,
    ROUND(AVG(b.ticket_price)::DECIMAL, 2) AS avg_ticket_price
FROM flights f
LEFT JOIN bookings b ON f.flight_id = b.flight_id
GROUP BY EXTRACT(HOUR FROM f.departure_time)::INTEGER
ORDER BY departure_hour;

-- ============================================
-- CHANNEL AND SEGMENT ANALYTICS
-- ============================================

-- Query 9: Booking Channel Deep Dive
CREATE OR REPLACE VIEW v_channel_deep_dive AS
SELECT 
    b.booking_channel,
    COUNT(b.booking_id) AS total_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW') AS noshow_count,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'CANCELLED') AS cancel_count,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'COMPLETED') AS completed_count,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
          NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS noshow_rate,
    ROUND(AVG(b.ticket_price)::DECIMAL, 2) AS avg_ticket_price,
    ROUND(AVG(b.booking_lead_time)::DECIMAL, 1) AS avg_lead_days,
    ROUND(AVG(b.noshow_probability)::DECIMAL * 100, 2) AS avg_predicted_prob,
    COUNT(DISTINCT b.passenger_id) AS unique_passengers,
    ROUND(COUNT(b.booking_id)::DECIMAL / 
          NULLIF(SUM(COUNT(b.booking_id)) OVER(), 0) * 100, 2) AS market_share
FROM bookings b
GROUP BY b.booking_channel;

-- Query 10: Cabin Class Analysis
CREATE OR REPLACE VIEW v_cabin_class_analysis AS
SELECT 
    b.cabin_class,
    COUNT(b.booking_id) AS total_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW') AS noshow_count,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
          NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS noshow_rate,
    ROUND(AVG(b.ticket_price)::DECIMAL, 2) AS avg_price,
    ROUND(AVG(b.booking_lead_time)::DECIMAL, 1) AS avg_lead_days,
    ROUND(AVG(b.noshow_probability)::DECIMAL * 100, 2) AS avg_predicted_prob,
    SUM(b.ticket_price) AS total_revenue
FROM bookings b
GROUP BY b.cabin_class;

-- ============================================
-- MODEL PERFORMANCE ANALYTICS
-- ============================================

-- Query 11: Model Performance Over Time
CREATE OR REPLACE VIEW v_model_performance_trends AS
SELECT 
    model_name,
    model_version,
    DATE_TRUNC('day', evaluation_date)::DATE AS eval_date,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    roc_auc,
    evaluation_date
FROM model_performance
ORDER BY model_name, model_version, evaluation_date DESC;

-- Query 12: Prediction Accuracy vs Actual
CREATE OR REPLACE FUNCTION fn_prediction_accuracy(days INTEGER DEFAULT 7)
RETURNS TABLE (
    total_predictions BIGINT,
    correct_predictions BIGINT,
    accuracy DECIMAL,
    true_positives BIGINT,
    false_positives BIGINT,
    true_negatives BIGINT,
    false_negatives BIGINT,
    precision_score DECIMAL,
    recall_score DECIMAL,
    f1_score DECIMAL
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH predictions_with_outcome AS (
        SELECT 
            pr.prediction_id,
            pr.noshow_probability,
            pr.prediction_class,
            CASE WHEN b.booking_status = 'NOSHOW' THEN 1 ELSE 0 END AS actual_noshow,
            CASE 
                WHEN pr.prediction_class = 1 AND b.booking_status = 'NOSHOW' THEN 'TP'
                WHEN pr.prediction_class = 1 AND b.booking_status != 'NOSHOW' THEN 'FP'
                WHEN pr.prediction_class = 0 AND b.booking_status != 'NOSHOW' THEN 'TN'
                ELSE 'FN'
            END AS prediction_type
        FROM predictions pr
        JOIN bookings b ON pr.booking_id = b.booking_id
        WHERE pr.prediction_timestamp >= CURRENT_DATE - days
    )
    SELECT 
        COUNT(*)::BIGINT AS total_predictions,
        COUNT(*) FILTER (WHERE prediction_type IN ('TP', 'TN'))::BIGINT AS correct_predictions,
        ROUND(COUNT(*) FILTER (WHERE prediction_type IN ('TP', 'TN'))::DECIMAL / 
              NULLIF(COUNT(*), 0) * 100, 2) AS accuracy,
        COUNT(*) FILTER (WHERE prediction_type = 'TP')::BIGINT AS true_positives,
        COUNT(*) FILTER (WHERE prediction_type = 'FP')::BIGINT AS false_positives,
        COUNT(*) FILTER (WHERE prediction_type = 'TN')::BIGINT AS true_negatives,
        COUNT(*) FILTER (WHERE prediction_type = 'FN')::BIGINT AS false_negatives,
        ROUND(COUNT(*) FILTER (WHERE prediction_type = 'TP')::DECIMAL / 
              NULLIF(COUNT(*) FILTER (WHERE prediction_type IN ('TP', 'FP')), 0) * 100, 2) AS precision_score,
        ROUND(COUNT(*) FILTER (WHERE prediction_type = 'TP')::DECIMAL / 
              NULLIF(COUNT(*) FILTER (WHERE prediction_type IN ('TP', 'FN')), 0) * 100, 2) AS recall_score,
        ROUND(
            2.0 * (COUNT(*) FILTER (WHERE prediction_type = 'TP')::DECIMAL / 
                   NULLIF(COUNT(*) FILTER (WHERE prediction_type IN ('TP', 'FP')), 0)) *
            (COUNT(*) FILTER (WHERE prediction_type = 'TP')::DECIMAL / 
             NULLIF(COUNT(*) FILTER (WHERE prediction_type IN ('TP', 'FN')), 0)) /
            NULLIF(
                (COUNT(*) FILTER (WHERE prediction_type = 'TP')::DECIMAL / 
                 NULLIF(COUNT(*) FILTER (WHERE prediction_type IN ('TP', 'FP')), 0)) +
                (COUNT(*) FILTER (WHERE prediction_type = 'TP')::DECIMAL / 
                 NULLIF(COUNT(*) FILTER (WHERE prediction_type IN ('TP', 'FN')), 0))
            , 0) * 100, 2
        ) AS f1_score
    FROM predictions_with_outcome;
END;
$$;

-- ============================================
-- OVERBOOKING SIMULATION QUERIES
-- ============================================

-- Query 13: Overbooking Strategy Analysis
CREATE OR REPLACE FUNCTION fn_overbooking_analysis(
    p_flight_id UUID,
    p_max_overbooking INTEGER DEFAULT 10
)
RETURNS TABLE (
    overbooking_level INTEGER,
    base_capacity INTEGER,
    total_bookings INTEGER,
    expected_noshows DECIMAL,
    expected_shows DECIMAL,
    denied_boardings DECIMAL,
    expected_empty_seats DECIMAL,
    risk_score DECIMAL,
    recommended_overbooking INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_base_capacity INTEGER;
    v_total_bookings INTEGER;
    v_avg_noshow_rate DECIMAL;
BEGIN
    -- Get flight details
    SELECT f.capacity, COUNT(b.booking_id), AVG(b.noshow_probability)
    INTO v_base_capacity, v_total_bookings, v_avg_noshow_rate
    FROM flights f
    LEFT JOIN bookings b ON f.flight_id = b.flight_id
    WHERE f.flight_id = p_flight_id
    GROUP BY f.capacity;
    
    RETURN QUERY
    WITH RECURSIVE overbooking_simulation AS (
        SELECT 
            0 AS overbooking_level,
            v_base_capacity AS base_capacity,
            v_total_bookings AS total_bookings,
            v_total_bookings * v_avg_noshow_rate AS expected_noshows,
            v_total_bookings * (1 - v_avg_noshow_rate) AS expected_shows,
            GREATEST(0, v_total_bookings * (1 - v_avg_noshow_rate) - v_base_capacity) AS denied_boardings,
            GREATEST(0, v_base_capacity - v_total_bookings * (1 - v_avg_noshow_rate)) AS expected_empty_seats,
            0.5 AS risk_score
        UNION ALL
        SELECT 
            ob.overbooking_level + 1,
            v_base_capacity,
            v_total_bookings,
            v_total_bookings * v_avg_noshow_rate,
            v_total_bookings * (1 - v_avg_noshow_rate),
            GREATEST(0, v_total_bookings * (1 - v_avg_noshow_rate) - v_base_capacity + (ob.overbooking_level + 1)),
            GREATEST(0, v_base_capacity - v_total_bookings * (1 - v_avg_noshow_rate) + (ob.overbooking_level + 1)),
            CASE 
                WHEN (v_total_bookings * (1 - v_avg_noshow_rate) - v_base_capacity + (ob.overbooking_level + 1)) > 0 
                THEN LEAST(1.0, (v_total_bookings * (1 - v_avg_noshow_rate) - v_base_capacity + (ob.overbooking_level + 1)) / 10.0)
                ELSE 0.0
            END
        FROM overbooking_simulation ob
        WHERE ob.overbooking_level < p_max_overbooking
    )
    SELECT 
        overbooking_level,
        base_capacity,
        total_bookings,
        ROUND(expected_noshows::DECIMAL, 2),
        ROUND(expected_shows::DECIMAL, 2),
        ROUND(denied_boardings::DECIMAL, 2),
        ROUND(expected_empty_seats::DECIMAL, 2),
        ROUND(risk_score::DECIMAL, 4),
        (SELECT MIN(overbooking_level) FROM overbooking_simulation 
         WHERE denied_boardings <= 1 AND expected_empty_seats < 5) AS recommended_overbooking
    FROM overbooking_simulation;
END;
$$;

-- ============================================
-- DRIFT MONITORING QUERIES
-- ============================================

-- Query 14: Feature Distribution Drift Detection
CREATE OR REPLACE FUNCTION fn_detect_feature_drift(
    p_reference_table VARCHAR,
    p_current_table VARCHAR,
    p_feature VARCHAR,
    p_threshold DECIMAL DEFAULT 0.1
)
RETURNS TABLE (
    feature_name VARCHAR,
    reference_mean DECIMAL,
    current_mean DECIMAL,
    mean_change DECIMAL,
    reference_std DECIMAL,
    current_std DECIMAL,
    std_change DECIMAL,
    distribution_shift DECIMAL,
    drift_detected BOOLEAN
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    EXECUTE format('
        WITH reference_stats AS (
            SELECT 
                AVG(%I) AS ref_mean,
                STDDEV(%I) AS ref_std
            FROM %I
        ),
        current_stats AS (
            SELECT 
                AVG(%I) AS cur_mean,
                STDDEV(%I) AS cur_std
            FROM %I
        )
        SELECT 
            %L AS feature_name,
            rs.ref_mean,
            cs.cur_mean,
            COALESCE(cs.cur_mean - rs.ref_mean, 0) AS mean_change,
            rs.ref_std,
            cs.cur_std,
            COALESCE(cs.cur_std - rs.ref_std, 0) AS std_change,
            ABS(COALESCE(cs.cur_mean - rs.ref_mean, 0) / NULLIF(rs.ref_mean, 0)) AS distribution_shift,
            ABS(COALESCE(cs.cur_mean - rs.ref_mean, 0) / NULLIF(rs.ref_mean, 0)) > %L AS drift_detected
        FROM reference_stats rs, current_stats cs
    ', p_feature, p_feature, p_reference_table, 
       p_feature, p_feature, p_current_table,
       p_feature, p_threshold);
END;
$$;

-- Query 15: Prediction Distribution Drift
CREATE OR REPLACE FUNCTION fn_prediction_drift(p_days INTEGER DEFAULT 7)
RETURNS TABLE (
    metric_name VARCHAR,
    previous_period DECIMAL,
    current_period DECIMAL,
    change_percent DECIMAL,
    drift_status VARCHAR
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH previous_stats AS (
        SELECT 
            AVG(noshow_probability) AS avg_prob,
            STDDEV(noshow_probability) AS std_prob,
            COUNT(*) AS prediction_count
        FROM predictions
        WHERE prediction_timestamp >= CURRENT_DATE - (2 * p_days)
          AND prediction_timestamp < CURRENT_DATE - p_days
    ),
    current_stats AS (
        SELECT 
            AVG(noshow_probability) AS avg_prob,
            STDDEV(noshow_probability) AS std_prob,
            COUNT(*) AS prediction_count
        FROM predictions
        WHERE prediction_timestamp >= CURRENT_DATE - p_days
    )
    SELECT 
        'Average Probability' AS metric_name,
        ps.avg_prob AS previous_period,
        cs.avg_prob AS current_period,
        ROUND((cs.avg_prob - ps.avg_prob) / NULLIF(ps.avg_prob, 0) * 100, 2) AS change_percent,
        CASE 
            WHEN ABS((cs.avg_prob - ps.avg_prob) / NULLIF(ps.avg_prob, 0)) > 0.1 THEN 'DRIFT_DETECTED'
            ELSE 'STABLE'
        END AS drift_status
    FROM previous_stats ps, current_stats cs
    
    UNION ALL
    
    SELECT 
        'Prediction Count' AS metric_name,
        ps.prediction_count AS previous_period,
        cs.prediction_count AS current_period,
        ROUND((cs.prediction_count - ps.prediction_count) / NULLIF(ps.prediction_count::DECIMAL, 0) * 100, 2) AS change_percent,
        CASE 
            WHEN ABS((cs.prediction_count - ps.prediction_count) / NULLIF(ps.prediction_count::DECIMAL, 0)) > 0.2 THEN 'DRIFT_DETECTED'
            ELSE 'STABLE'
        END AS drift_status
    FROM previous_stats ps, current_stats cs;
END;
$$;

-- ============================================
-- SAMPLE DATA GENERATION FOR TESTING
-- ============================================

-- Generate sample data for testing the analytics queries
CREATE OR REPLACE PROCEDURE sp_generate_sample_data(num_bookings INTEGER DEFAULT 1000)
LANGUAGE plpgsql
AS $$
DECLARE
    v_passenger_id UUID;
    v_flight_id UUID;
    v_booking_code VARCHAR(20);
    v_booking_date TIMESTAMP;
    v_noshow_probability DECIMAL(5, 4);
    v_booking_status VARCHAR(20);
    v_passenger_count INTEGER;
    v_flight_count INTEGER;
BEGIN
    -- Get counts
    SELECT COUNT(*) INTO v_passenger_count FROM passengers;
    SELECT COUNT(*) INTO v_flight_count FROM flights;
    
    IF v_passenger_count = 0 OR v_flight_count = 0 THEN
        RAISE NOTICE 'Cannot generate bookings - passengers or flights table is empty';
        RETURN;
    END IF;
    
    -- Generate bookings
    FOR i IN 1..num_bookings LOOP
        -- Random passenger and flight
        v_passenger_id := (SELECT passenger_id FROM passengers ORDER BY RANDOM() LIMIT 1);
        v_flight_id := (SELECT flight_id FROM flights ORDER BY RANDOM() LIMIT 1);
        v_booking_code := 'BK' || upper(to_char(CURRENT_TIMESTAMP, 'YYMMDDHH24MISS')) || lpad(i::TEXT, 6, '0');
        v_booking_date := NOW() - (random() * 60 + 1) * INTERVAL '1 day';
        
        -- Generate probability based on various factors
        v_noshow_probability := random() * 0.4;
        
        -- Adjust based on booking lead time
        v_noshow_probability := v_noshow_probability + 
            (EXTRACT(DAY FROM (SELECT departure_time FROM flights WHERE flight_id = v_flight_id) - v_booking_date) / 100.0);
        
        -- Cap probability
        v_noshow_probability := LEAST(0.95, GREATEST(0.05, v_noshow_probability));
        
        -- Determine actual status based on probability (for historical data)
        v_booking_status := CASE WHEN random() < v_noshow_probability THEN 'NOSHOW' ELSE 'COMPLETED' END;
        
        -- Insert booking
        INSERT INTO bookings (
            booking_code, passenger_id, flight_id, booking_date, booking_channel,
            booking_lead_time, ticket_price, cabin_class, booking_status, noshow_probability
        )
        SELECT 
            v_booking_code, v_passenger_id, v_flight_id, v_booking_date,
            (ARRAY['WEB', 'MOBILE', 'AGENT', 'CALL_CENTER'])[floor(random() * 4 + 1)::INTEGER],
            EXTRACT(DAY FROM (SELECT departure_time FROM flights WHERE flight_id = v_flight_id) - v_booking_date),
            (SELECT base_price FROM flights WHERE flight_id = v_flight_id) * (0.8 + random() * 0.4),
            (ARRAY['ECONOMY', 'PREMIUM_ECONOMY', 'BUSINESS', 'FIRST'])[floor(random() * 4 + 1)::INTEGER],
            v_booking_status,
            v_noshow_probability;
    END LOOP;
    
    -- Update passenger stats
    FOR v_passenger_id IN SELECT passenger_id FROM passengers LOOP
        CALL sp_update_passenger_stats(v_passenger_id);
    END LOOP;
    
    RAISE NOTICE 'Generated % sample bookings', num_bookings;
END;
$$;

-- Grant execute on all functions
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA PUBLIC TO airline_user;
GRANT SELECT ON ALL TABLES IN SCHEMA PUBLIC TO airline_user;