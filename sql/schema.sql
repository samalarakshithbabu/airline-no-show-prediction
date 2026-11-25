-- Airline No-Show Prediction System Database Schema
-- PostgreSQL Database

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- PASSENGERS TABLE
-- ============================================
CREATE TABLE passengers (
    passenger_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    passenger_code VARCHAR(20) UNIQUE NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(20),
    date_of_birth DATE,
    loyalty_tier VARCHAR(20) DEFAULT 'STANDARD',
    total_bookings INTEGER DEFAULT 0,
    total_noshows INTEGER DEFAULT 0,
    total_cancellations INTEGER DEFAULT 0,
    total_flights INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- FLIGHTS TABLE
-- ============================================
CREATE TABLE flights (
    flight_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    flight_code VARCHAR(20) UNIQUE NOT NULL,
    departure_airport VARCHAR(10) NOT NULL,
    arrival_airport VARCHAR(10) NOT NULL,
    route VARCHAR(20),
    departure_time TIMESTAMP NOT NULL,
    arrival_time TIMESTAMP NOT NULL,
    aircraft_type VARCHAR(50),
    capacity INTEGER NOT NULL DEFAULT 180,
    base_price DECIMAL(10, 2) NOT NULL,
    flight_date DATE GENERATED ALWAYS AS (departure_time::DATE) STORED,
    flight_dayofweek INTEGER GENERATED ALWAYS AS (EXTRACT(DOW FROM departure_time)::INTEGER) STORED,
    flight_hour INTEGER GENERATED ALWAYS AS (EXTRACT(HOUR FROM departure_time)::INTEGER) STORED,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for flight queries
CREATE INDEX idx_flights_route ON flights(route);
CREATE INDEX idx_flights_date ON flights(flight_date);
CREATE INDEX idx_flights_departure ON flights(departure_time);

-- ============================================
-- BOOKINGS TABLE
-- ============================================
CREATE TABLE bookings (
    booking_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    booking_code VARCHAR(20) UNIQUE NOT NULL,
    passenger_id UUID REFERENCES passengers(passenger_id) ON DELETE SET NULL,
    flight_id UUID REFERENCES flights(flight_id) ON DELETE SET NULL,
    booking_date TIMESTAMP NOT NULL,
    booking_channel VARCHAR(50) NOT NULL DEFAULT 'WEB',
    booking_lead_time INTEGER,  -- Days between booking and flight
    ticket_price DECIMAL(10, 2) NOT NULL,
    cabin_class VARCHAR(20) DEFAULT 'ECONOMY',
    baggage_included BOOLEAN DEFAULT TRUE,
    seat_number VARCHAR(5),
    check_in_status BOOLEAN DEFAULT FALSE,
    check_in_time TIMESTAMP,
    booking_status VARCHAR(20) DEFAULT 'CONFIRMED',  -- CONFIRMED, CANCELLED, COMPLETED, NOSHOW
    noshow_confirmed BOOLEAN DEFAULT FALSE,
    noshow_probability DECIMAL(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for booking queries
CREATE INDEX idx_bookings_passenger ON bookings(passenger_id);
CREATE INDEX idx_bookings_flight ON bookings(flight_id);
CREATE INDEX idx_bookings_status ON bookings(booking_status);
CREATE INDEX idx_bookings_date ON bookings(booking_date);
CREATE INDEX idx_bookings_noshow_probability ON bookings(noshow_probability);

-- ============================================
-- BOOKING_HISTORY TABLE (Audit Trail)
-- ============================================
CREATE TABLE booking_history (
    history_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    booking_id UUID REFERENCES bookings(booking_id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    old_status VARCHAR(20),
    new_status VARCHAR(20),
    event_details JSONB,
    created_by VARCHAR(100) DEFAULT 'SYSTEM'
);

-- Indexes for history queries
CREATE INDEX idx_booking_history_booking ON booking_history(booking_id);
CREATE INDEX idx_booking_history_timestamp ON booking_history(event_timestamp);

-- ============================================
-- PREDICTIONS TABLE
-- ============================================
CREATE TABLE predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    booking_id UUID REFERENCES bookings(booking_id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    noshow_probability DECIMAL(5, 4) NOT NULL,
    prediction_class INTEGER NOT NULL,  -- 0 or 1
    feature_values JSONB NOT NULL,
    shap_explanation JSONB,
    risk_category VARCHAR(20),
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_run_time_ms INTEGER,
    is_active BOOLEAN DEFAULT TRUE
);

-- Indexes for prediction queries
CREATE INDEX idx_predictions_booking ON predictions(booking_id);
CREATE INDEX idx_predictions_model ON predictions(model_name, model_version);
CREATE INDEX idx_predictions_timestamp ON predictions(prediction_timestamp);
CREATE INDEX idx_predictions_risk ON predictions(risk_category);

-- ============================================
-- MODEL_PERFORMANCE TABLE
-- ============================================
CREATE TABLE model_performance (
    performance_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dataset_type VARCHAR(20) NOT NULL,  -- TRAIN, VALIDATION, TEST, PRODUCTION
    accuracy DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall_score DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    roc_auc DECIMAL(5, 4),
    pr_auc DECIMAL(5, 4),
    specificity DECIMAL(5, 4),
    confusion_matrix JSONB,
    calibration_data JSONB,
    train_samples INTEGER,
    test_samples INTEGER,
    hyperparams JSONB,
    feature_importance JSONB,
    notes TEXT
);

-- Indexes for performance queries
CREATE INDEX idx_model_perf_model ON model_performance(model_name, model_version);
CREATE INDEX idx_model_perf_date ON model_performance(evaluation_date);

-- User accounts used by the API authentication service.
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- ============================================
-- ANALYTICAL VIEWS
-- ============================================

-- View: Daily Flight Statistics
CREATE OR REPLACE VIEW v_daily_flight_stats AS
SELECT 
    f.flight_date,
    f.route,
    f.flight_code,
    f.departure_time,
    f.capacity,
    COUNT(b.booking_id) AS total_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW') AS noshow_count,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'CANCELLED') AS cancellation_count,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'COMPLETED') AS completed_count,
    COUNT(b.booking_id) FILTER (WHERE b.check_in_status = TRUE) AS checked_in_count,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
          NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS noshow_rate,
    ROUND(AVG(b.noshow_probability)::DECIMAL * 100, 2) AS avg_predicted_noshow_prob,
    ROUND(AVG(b.ticket_price)::DECIMAL, 2) AS avg_ticket_price,
    SUM(b.ticket_price) AS total_revenue
FROM flights f
LEFT JOIN bookings b ON f.flight_id = b.flight_id
GROUP BY f.flight_id, f.flight_date, f.route, f.flight_code, 
         f.departure_time, f.capacity;

-- View: Passenger Booking Summary
CREATE OR REPLACE VIEW v_passenger_summary AS
SELECT 
    p.passenger_id,
    p.passenger_code,
    p.first_name,
    p.last_name,
    p.loyalty_tier,
    p.total_bookings,
    p.total_noshows,
    p.total_cancellations,
    p.total_flights,
    ROUND(p.total_noshows::DECIMAL / NULLIF(p.total_bookings, 0) * 100, 2) AS noshow_rate,
    ROUND(p.total_cancellations::DECIMAL / NULLIF(p.total_bookings, 0) * 100, 2) AS cancellation_rate,
    MAX(b.booking_date) AS last_booking_date,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW') AS recent_noshows,
    COUNT(b.booking_id) FILTER (WHERE b.booking_date >= CURRENT_DATE - INTERVAL '90 days') AS bookings_last_90_days
FROM passengers p
LEFT JOIN bookings b ON p.passenger_id = b.passenger_id
GROUP BY p.passenger_id;

-- View: Route Performance Analysis
CREATE OR REPLACE VIEW v_route_performance AS
SELECT 
    f.route,
    f.departure_airport,
    f.arrival_airport,
    COUNT(DISTINCT f.flight_id) AS total_flights,
    COUNT(DISTINCT b.booking_id) AS total_bookings,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
          NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS noshow_rate,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'CANCELLED')::DECIMAL / 
          NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS cancellation_rate,
    ROUND(AVG(b.booking_lead_time)::DECIMAL, 1) AS avg_booking_lead_time,
    ROUND(AVG(b.ticket_price)::DECIMAL, 2) AS avg_ticket_price,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_channel = 'MOBILE')::DECIMAL / 
          NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS mobile_booking_pct,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_channel = 'WEB')::DECIMAL / 
          NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS web_booking_pct
FROM flights f
LEFT JOIN bookings b ON f.flight_id = b.flight_id
GROUP BY f.route, f.departure_airport, f.arrival_airport;

-- View: Monthly No-Show Trends
CREATE OR REPLACE VIEW v_monthly_noshow_trends AS
SELECT 
    DATE_TRUNC('month', b.booking_date) AS month,
    COUNT(b.booking_id) AS total_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW') AS noshow_count,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
          NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS noshow_rate,
    ROUND(AVG(b.booking_lead_time)::DECIMAL, 1) AS avg_booking_lead_time,
    COUNT(DISTINCT b.passenger_id) AS unique_passengers
FROM bookings b
GROUP BY DATE_TRUNC('month', b.booking_date)
ORDER BY month;

-- View: Channel Performance
CREATE OR REPLACE VIEW v_channel_performance AS
SELECT 
    b.booking_channel,
    COUNT(b.booking_id) AS total_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW') AS noshow_count,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
          NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS noshow_rate,
    ROUND(AVG(b.ticket_price)::DECIMAL, 2) AS avg_ticket_price,
    ROUND(AVG(b.booking_lead_time)::DECIMAL, 1) AS avg_lead_time
FROM bookings b
GROUP BY b.booking_channel;

-- View: Day of Week Analysis
CREATE OR REPLACE VIEW v_dow_analysis AS
SELECT 
    EXTRACT(DOW FROM f.departure_time)::INTEGER AS day_of_week,
    TO_CHAR(f.departure_time, 'Day') AS day_name,
    COUNT(DISTINCT f.flight_id) AS total_flights,
    COUNT(b.booking_id) AS total_bookings,
    COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW') AS noshow_count,
    ROUND(COUNT(b.booking_id) FILTER (WHERE b.booking_status = 'NOSHOW')::DECIMAL / 
          NULLIF(COUNT(b.booking_id), 0) * 100, 2) AS noshow_rate,
    ROUND(AVG(b.ticket_price)::DECIMAL, 2) AS avg_ticket_price
FROM flights f
LEFT JOIN bookings b ON f.flight_id = b.flight_id
GROUP BY EXTRACT(DOW FROM f.departure_time), TO_CHAR(f.departure_time, 'Day')
ORDER BY day_of_week;

-- View: High-Risk Bookings (Prediction Summary)
CREATE OR REPLACE VIEW v_high_risk_bookings AS
SELECT 
    b.booking_code,
    p.passenger_code,
    p.first_name || ' ' || p.last_name AS passenger_name,
    f.flight_code,
    f.route,
    f.departure_time,
    b.booking_status,
    pr.noshow_probability,
    pr.risk_category,
    pr.prediction_timestamp,
    b.booking_channel,
    b.cabin_class,
    b.booking_lead_time
FROM bookings b
JOIN passengers p ON b.passenger_id = p.passenger_id
JOIN flights f ON b.flight_id = f.flight_id
JOIN LATERAL (
    SELECT booking_id, noshow_probability, risk_category, prediction_timestamp
    FROM predictions
    WHERE booking_id = b.booking_id AND is_active = TRUE
    ORDER BY prediction_timestamp DESC
    LIMIT 1
) pr ON TRUE
WHERE pr.noshow_probability > 0.5
ORDER BY pr.noshow_probability DESC;

-- ============================================
-- STORED PROCEDURES
-- ============================================

-- Procedure: Update Passenger Statistics
CREATE OR REPLACE PROCEDURE sp_update_passenger_stats(passenger_uuid UUID)
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE passengers p
    SET 
        total_bookings = (
            SELECT COUNT(*) FROM bookings WHERE passenger_id = p.passenger_id
        ),
        total_noshows = (
            SELECT COUNT(*) FROM bookings 
            WHERE passenger_id = p.passenger_id AND booking_status = 'NOSHOW'
        ),
        total_cancellations = (
            SELECT COUNT(*) FROM bookings 
            WHERE passenger_id = p.passenger_id AND booking_status = 'CANCELLED'
        ),
        total_flights = (
            SELECT COUNT(*) FROM bookings 
            WHERE passenger_id = p.passenger_id AND booking_status = 'COMPLETED'
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE p.passenger_id = passenger_uuid;
END;
$$;

-- Procedure: Calculate Flight Statistics
CREATE OR REPLACE PROCEDURE sp_calculate_flight_stats(flight_uuid UUID)
LANGUAGE plpgsql
AS $$
DECLARE
    v_total_bookings INTEGER;
    v_noshow_count INTEGER;
    v_actual_noshow_rate DECIMAL(5, 2);
BEGIN
    SELECT 
        COUNT(*) INTO v_total_bookings,
        COUNT(*) FILTER (WHERE booking_status = 'NOSHOW') INTO v_noshow_count
    FROM bookings
    WHERE flight_id = flight_uuid;
    
    IF v_total_bookings > 0 THEN
        v_actual_noshow_rate := (v_noshow_count::DECIMAL / v_total_bookings) * 100;
    ELSE
        v_actual_noshow_rate := 0;
    END IF;
    
    -- Update any flight-level statistics if needed
    RAISE NOTICE 'Flight % has % bookings with % noshows (%.2f%%)', 
        flight_uuid, v_total_bookings, v_noshow_count, v_actual_noshow_rate;
END;
$$;

-- Procedure: Archive Old Predictions
CREATE OR REPLACE PROCEDURE sp_archive_predictions(days_to_keep INTEGER DEFAULT 90)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Mark old predictions as inactive
    UPDATE predictions
    SET is_active = FALSE
    WHERE prediction_timestamp < CURRENT_TIMESTAMP - (days_to_keep || ' days')::INTERVAL
    AND is_active = TRUE;
    
    RAISE NOTICE 'Archived predictions older than % days', days_to_keep;
END;
$$;

-- ============================================
-- FUNCTIONS
-- ============================================

-- Function: Calculate No-Show Probability Category
CREATE OR REPLACE FUNCTION fn_get_risk_category(probability DECIMAL)
RETURNS VARCHAR(20)
LANGUAGE plpgsql
AS $$
BEGIN
    IF probability < 0.2 THEN
        RETURN 'LOW';
    ELSIF probability < 0.4 THEN
        RETURN 'MODERATE';
    ELSIF probability < 0.6 THEN
        return 'ELEVATED';
    ELSIF probability < 0.8 THEN
        return 'HIGH';
    ELSE
        return 'CRITICAL';
    END IF;
END;
$$;

-- Function: Generate Booking Code
CREATE OR REPLACE FUNCTION fn_generate_booking_code()
RETURNS VARCHAR(20)
LANGUAGE plpgsql
AS $$
DECLARE
    v_code VARCHAR(20);
BEGIN
    LOOP
        v_code := 'BK' || upper(to_char(CURRENT_TIMESTAMP, 'YYMMDD')) || 
                  lpad(floor(random() * 1000000)::TEXT, 6, '0');
        EXIT WHEN NOT EXISTS (SELECT 1 FROM bookings WHERE booking_code = v_code);
    END LOOP;
    RETURN v_code;
END;
$$;

-- ============================================
-- TRIGGERS
-- ============================================

-- Trigger: Update updated_at timestamp
CREATE OR REPLACE FUNCTION fn_update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers to relevant tables
CREATE TRIGGER trigger_passengers_update
    BEFORE UPDATE ON passengers
    FOR EACH ROW EXECUTE FUNCTION fn_update_timestamp();

CREATE TRIGGER trigger_flights_update
    BEFORE UPDATE ON flights
    FOR EACH ROW EXECUTE FUNCTION fn_update_timestamp();

CREATE TRIGGER trigger_bookings_update
    BEFORE UPDATE ON bookings
    FOR EACH ROW EXECUTE FUNCTION fn_update_timestamp();

-- ============================================
COMMENT ON SCHEMA public IS 'Airline No-Show Prediction System Database Schema';