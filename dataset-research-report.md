# Airline No-Show Dataset and Methodology Research

## Executive conclusion

The supplied training file is not suitable evidence for an operational airline no-show model. Its source, license, target definition, and label timing are undocumented. It contains only 3,000 bookings, repeated passengers, questionable aggregate-history fields, and strong timestamp-construction patterns. No compatible, license-verified public passenger no-show dataset was found in the focused catalog searches below, so no external data was merged.

Passenger-grouped validation and a forward booking-time experiment both produced near-chance ranking and weak probability quality. No candidate passed the existing production approval gate; no model promotion or business benefit is justified. The frontend and simulator were clarified to identify scenario outputs as assumption-based comparisons, not operational recommendations.

## Dataset audit

Source file: [`data/training_data.csv`](./data/training_data.csv)

| Audit item | Observed result |
|---|---|
| Size | 3,000 rows, 15 columns, 334,365 bytes |
| Provenance / license | No source, license, collection procedure, or data dictionary found in the repository |
| Target | `no_show`, binary values 0/1: 1,086 rows with value 1 (36.2%), 1,914 with value 0. The code treats 1 as the positive class, but no authoritative definition or polarity documentation was found |
| Missing data | No missing cells |
| Exact duplicate rows | 0 |
| Booking IDs | 3,000 unique IDs; sequential values 1–3,000 |
| Passengers | 1,872 unique; 807 repeat passengers across 1,935 rows. Group sizes range from 1 to 6 (median 1). 425 repeat-passenger groups contain both target labels |
| Booking dates | 2024-01-01 through 2024-01-03; exactly 1,440 records on each of the first two days and 120 on the third |
| Flight dates | 2024-01-02 through 2024-03-02; 2,975 distinct timestamps |
| Lead time | 1–59 days. `booking_lead_time` agrees exactly with the date difference in all rows |
| Feature values requiring owner review | 374 rows (12.5%) have the same departure and arrival airport; 304 rows (10.1%) have `total_cancellations > total_bookings`. These may be definition/aggregation issues, but cannot be resolved from the file |

### Provenance and representativeness

No source or target documentation exists alongside the CSV, and repository searches found no attribution. The sequential booking IDs, booking timestamp increments of exactly one minute, and the fact that every flight timestamp inherits the booking timestamp's exact time of day are consistent with constructed or heavily transformed data. These patterns do **not** prove that the file is synthetic, but its provenance must be treated as unknown.

The 36.2% label prevalence is a property of this file only. It is not a defensible estimate for any airline, route, carrier, season, or customer population. The file's date span and route/airport coverage are too narrow to establish industry representativeness.

### Target and data-quality concerns

`no_show` is the only target-like field. Its label definition is not documented: the repository cannot establish whether a no-show means failure to board, failure to check in, cancellation after ticketing, or another event, nor when the label is finalized. Training assumes value 1 is a no-show; that assumption needs confirmation from the data owner before any operational use.

The history columns have no as-of timestamp or event-level construction:

- `total_noshows` has a positive point-biserial association with the current target (r=0.186, unadjusted p≈1.0e-24); `total_bookings` is also associated (r=0.114, p≈3.2e-10). Either may include the current booking or later outcomes.
- Among current target-1 rows, 306 have `total_noshows == 1`; this is not proof of leakage by itself, but is consistent with the aggregate being able to include the current outcome.
- `total_cancellations > total_bookings` occurs 304 times. The column definitions and aggregation windows are unknown.

These columns, and features derived from them, must remain excluded until built from timestamped events strictly preceding the prediction time. The current modeling feature selector excludes the passenger aggregates.

## Feature availability and leakage matrix

“Present” means present in the CSV, not independently verified as correct or available in the live booking system. Prediction-time availability is conditional on a booking-time scoring use case.

| Field(s) | Present? | Booking-time availability | Leakage / quality assessment | Current treatment |
|---|---|---|---|---|
| `booking_id` | Yes | At booking | Identifier, not a behavior feature; sequential values could encode file order | Excluded |
| `passenger_id` | Yes | At booking | Useful to keep passenger groups together in validation; identity memorization is unsafe | Excluded from model features |
| `booking_date` | Yes | At booking if the timestamp is authentic | May generate booking-time calendar features. Its minute-by-minute sequence is a major provenance warning | Used to derive temporal features; needs source validation |
| `flight_date` | Yes | Normally known at booking | Flight calendar features are available in principle. Time-of-day is not credible here: it matches booking time-of-day in all 3,000 rows | Calendar features used; no departure-hour signal |
| `booking_lead_time` | Yes | Derivable at booking | Exact date-difference duplicate; not an independent field | Recomputed from dates |
| `booking_channel` | Yes | Normally known at booking | Provenance and categorical semantics are undocumented; observed unadjusted association is weak (chi-square p≈0.713) | Encoded in current pipeline; validate with source system |
| `cabin_class` | Yes | Normally known at booking | Unadjusted target association (chi-square p≈0.019) is exploratory only; multiple comparisons and data provenance are unresolved | Encoded |
| `ticket_price` | Yes | Usually known at booking | Currency, tax/refund basis, and pricing timestamp are unknown | Used by current engineering; validate definitions |
| `departure_airport`, `arrival_airport` | Yes | Normally known at booking | Same-airport pairs in 374 rows; only eight values observed per column | Encoded; needs route validation |
| `loyalty_tier` | Yes | Only if captured as-of booking | Snapshot timing and tier update rules are unknown; unadjusted association is weak (p≈0.262) | Encoded; validate as-of availability |
| `total_bookings`, `total_noshows`, `total_cancellations` | Yes | Unknown | Snapshot window and event ordering are undocumented; possible target/post-outcome leakage | Excluded |
| `no_show` | Yes | Only after the flight/outcome is resolved | Target, not a prediction-time feature; definition and finalization timing unknown | Target only; value 1 treated as positive |
| Airline/carrier | No | Could be known at booking | Potentially useful but unavailable; do not infer from route | Not used |
| Actual departure time | No | Could be known at booking | Source `flight_date` has the same clock time as `booking_date` in every row, so extracting its hour would encode a timestamp artifact | Not used |
| Ticket type/fare rules, check-in, passenger event history | No | Depends on the scoring cutoff | Check-in is post-booking and could leak for an earlier scoring use case; event history requires strict as-of joins | Not used |

The active feature engineering uses supported fields such as lead time, flight calendar indicators, booking channel, cabin class, price, and airports. No evidence supports adding new fields to this dataset. In particular, do not manufacture airline or departure-time values.

## Public dataset research

Focused searches were made against the [Kaggle catalog for airline passenger no-show](https://www.kaggle.com/api/v1/datasets/list?search=airline%20passenger%20no-show), the [Kaggle catalog for flight-booking no-show](https://www.kaggle.com/api/v1/datasets/list?search=flight%20booking%20no%20show), and the [Hugging Face datasets API](https://huggingface.co/api/datasets?search=airline%20no-show&limit=20). No directly compatible passenger-level airline no-show dataset with verifiable label semantics and suitable licensing was identified. Catalog search is not exhaustive; this is not proof that no such dataset exists.

| Public source examined | Published content / license evidence | Suitability |
|---|---|---|
| [Kaggle Airline Passenger Satisfaction (120k+ copy)](https://www.kaggle.com/datasets/mysarahmadbhat/airline-passenger-satisfaction) | Kaggle metadata describes satisfaction scores and reports CC0 | Not suitable: satisfaction is not whether a passenger boarded; no-show labels cannot be derived. Do not merge |
| [Kaggle OpenFlights Airline Database](https://www.kaggle.com/datasets/open-flights/airline-database) | Kaggle metadata describes a static airline master list and states it was downloaded from OpenFlights under the Open Database License (ODbL) | No bookings or passenger outcomes. Could only support airline-name/code reference if a legitimate carrier field is later obtained; the current CSV has none |
| [U.S. BTS Airline On-Time Statistics and Delay Causes](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp) | BTS describes public flight-level on-time, delay, cancellation, and diversion reporting | Not suitable as the target source: operational flight outcomes are not passenger no-show labels. Joining would need a defensible flight key and time alignment; no such key exists here. The page reviewed did not establish a reuse license for a merged passenger-level product |

No external dataset was downloaded or merged. A future candidate must have a documented source and reuse license, booking-level unit, target definition and polarity, event timestamps, prediction-time feature availability, and a valid key/time policy before use.

## Model experiments and validation

### Method

The repository's production-oriented experiment uses a passenger-grouped 80/20 holdout with random seed 42, then five-fold `StratifiedGroupKFold` on training passengers. Preprocessing/reference statistics and candidate selection are training-only; passenger aggregate histories are excluded. Candidates compared were Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM, using the existing small deterministic parameter grids and class weighting where supported. No resampling was used. Isotonic calibration and threshold choice use training out-of-fold predictions, not held-out labels.

Because `booking_date` exists, a separate forward booking-time diagnostic was also run. The cutoff was 2024-01-02 15:59:12. It held out the 600 later bookings and removed 440 earlier training rows for passengers appearing in the future period, leaving 1,960 training rows, 1,330 train passengers, and 542 test passengers with zero passenger overlap. Train/test prevalence was 35.97%/37.33%. The timestamp construction artifacts make this only a sensitivity analysis, not a trustworthy real-world temporal validation.

Passenger-grouped validation is appropriate because repeat passengers exist. Ordinary row-stratified folds would leak passenger identity/history across folds and are not an appropriate primary estimate. The results below use the same folds/split for all models within each experiment. Metrics from different split designs are not directly comparable.

### Prior grouped holdout

The earlier row-random Logistic Regression result (ROC-AUC 0.6312, PR-AUC 0.4754, F1 0.4167) is retained only as historical context. It allowed repeat passengers to cross the split and used older feature handling; it is leakage-prone and must not be interpreted as a performance improvement over the grouped results.

| Model | Grouped CV ROC-AUC | CV PR-AUC | CV Brier | Held-out ROC-AUC | Held-out PR-AUC | Held-out Brier |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost (selected by CV PR-AUC) | 0.5346 | 0.3814 | 0.2495 | 0.5036 | 0.3632 | 0.2322 |
| Random Forest | 0.5343 | 0.3812 | 0.2472 | 0.4824 | 0.3534 | 0.2339 |
| Gradient Boosting | 0.5304 | 0.3811 | 0.2515 | 0.5188 | 0.3771 | 0.2308 |
| Logistic Regression | 0.5302 | 0.3758 | 0.2518 | 0.4991 | 0.3563 | 0.2322 |
| LightGBM | 0.5256 | 0.3745 | 0.2547 | 0.4881 | 0.3622 | 0.2321 |

Grouped test prevalence was 0.3617. The selected XGBoost's held-out PR-AUC (0.3632) is effectively at prevalence; its ROC-AUC is near chance. Constant-prevalence Brier is 0.2309, better than the selected model's 0.2322. At threshold 0.50 it predicted no positive cases. Its training-OOF F1 threshold was 0.3179; on test it gave precision 0.3647, recall 0.9816, F1 0.5318, and specificity 0.0313 (371 false positives, 4 false negatives, 12 true negatives). The F1 score is achieved by flagging nearly everyone, not by useful ranking.

### Forward booking-time diagnostic

| Model | Training-only grouped-CV ROC-AUC | CV PR-AUC | CV Brier | Future holdout ROC-AUC | Holdout PR-AUC | Holdout Brier |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest (selected by CV PR-AUC) | 0.5194 | 0.3804 | 0.2484 | 0.5433 | 0.3983 | 0.2359 |
| Gradient Boosting | 0.5151 | 0.3734 | 0.2710 | 0.5412 | 0.3929 | 0.2329 |
| XGBoost | 0.5141 | 0.3715 | 0.2623 | 0.5381 | 0.4017 | 0.2335 |
| LightGBM | 0.5066 | 0.3704 | 0.2658 | 0.5432 | 0.3969 | 0.2330 |
| Logistic Regression | 0.4986 | 0.3556 | 0.2578 | 0.5519 | 0.4039 | 0.2333 |

Temporal holdout prevalence was 0.3733; constant-prevalence Brier was 0.2340. The CV-selected Random Forest's Brier was worse at 0.2359. At threshold 0.50 it had 0.20 precision, 0.0089 recall, and 0.0171 F1 (2 true positives out of 224). Its training-OOF F1 threshold collapsed to 0.0; applying that threshold flagged every test row, yielding 0.3733 precision, 1.0 recall, and 0.5437 F1. This is a degenerate threshold and not an operational choice.

The highest temporal test PR-AUC in the table belongs to Logistic Regression, but selecting by that test result would leak test information; the model selection remains based on training-only grouped CV, which selected Random Forest. Overall discrimination remains weak across candidates and split designs.

### Calibration, imbalance, and threshold findings

For the CV-selected Random Forest on the temporal holdout, the calibrated reliability bins were:

| Test rows | Mean predicted no-show probability | Observed no-show rate |
|---:|---:|---:|
| 347 | 0.343 | 0.340 |
| 203 | 0.373 | 0.414 |
| 50 | 0.460 | 0.440 |

This coarse curve is not evidence of robust calibration; predicted probabilities cluster near the base rate. Its Brier score is worse than a constant-prevalence prediction. Class weighting did not solve the ranking problem. Thresholds optimized on training OOF scores were applied only after selection and still produced unusable operating points. No model satisfies the existing approval gate (ROC-AUC >0.70, F1 >0.50, precision >0.50, recall >0.50, and PR-AUC above held-out prevalence).

## Business validation and simulator boundaries

The simulator currently receives a user-entered average no-show probability; it does not consume the model's per-passenger predictions. Its default parameters are assumptions: $300 per empty seat, $1,500 compensation per denied passenger, and $5,000 denied-boarding cost applied per denied passenger. It models ticket revenue as `bookings × ticket_price`, including no-shows. These values and revenue treatment are not validated airline economics. The “risk_adjusted” comparison also subtracts a probability-based denial penalty from modeled profit, so it is a scenario-ranking heuristic rather than an actuarial expected-cost measure.

Illustration only: with capacity 180, 150 current bookings, ticket price $350, the file's 36.2% no-show prevalence as the entered probability, default costs, seed 42, and 10,000 Monte Carlo runs, the simulator reported $27,235.74 modeled profit and 84.21 expected empty seats at +0 bookings; at +20 it reported $38,058.34 modeled profit and 71.47 expected empty seats. Neither scenario had a denied-boarding event in those simulations. The modeled ranking therefore increases with the maximum additional bookings in this low-load setup; it does not demonstrate a real benefit or justify a recommendation.

The UI now calls this the “highest-scoring simulated scenario,” says the probability is user-entered, and explicitly says the result is not operational advice. The API comparison likewise marks `operational_recommendation_available` false. Actual cost, refund, compensation, customer-protection, connection, and risk-limit assumptions must be validated by an airline before any business decision.

## Integration and remaining work

No model or external data was promoted. Existing passenger-grouped feature engineering, approval gating, and application functionality were retained. The only application behavior change from this research pass is clearer labeling/disclaimers for the overbooking scenario comparison, with a contract assertion and updated browser test.

To establish research-grade model validity, obtain an authorized booking-level dataset with documented labels and prediction timestamps; reconstruct passenger history using strictly prior events; verify route/count semantics and source/license; then define a temporal evaluation cohort and business costs with the data owner. Re-run grouped and time-based validation, calibration, threshold, and subgroup analyses. Until then, this application remains a research prototype and should not be used to approve overbooking or passenger interventions.

### Sources

- [Kaggle airline passenger no-show catalog search](https://www.kaggle.com/api/v1/datasets/list?search=airline%20passenger%20no-show)
- [Kaggle flight-booking no-show catalog search](https://www.kaggle.com/api/v1/datasets/list?search=flight%20booking%20no%20show)
- [Hugging Face airline no-show dataset search](https://huggingface.co/api/datasets?search=airline%20no-show&limit=20)
- [Kaggle Airline Passenger Satisfaction metadata](https://www.kaggle.com/datasets/mysarahmadbhat/airline-passenger-satisfaction)
- [Kaggle OpenFlights Airline Database metadata](https://www.kaggle.com/api/v1/datasets/view/open-flights/airline-database)
- [U.S. BTS Airline On-Time Statistics and Delay Causes](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp)
