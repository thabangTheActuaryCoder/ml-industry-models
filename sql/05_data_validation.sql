-- ============================================================
-- Data validation queries
-- Run after loading CSVs to verify row counts and class balance
-- ============================================================

USE CATALOG ml_models;

-- Banking
SELECT
  'banking' AS industry,
  COUNT(*)  AS total_rows,
  SUM(target) AS positive_count,
  ROUND(SUM(target) / COUNT(*) * 100, 1) AS positive_rate_pct
FROM banking.credit_data

UNION ALL

-- Insurance
SELECT
  'insurance',
  COUNT(*),
  SUM(target),
  ROUND(SUM(target) / COUNT(*) * 100, 1)
FROM insurance.claim_data

UNION ALL

-- Retail
SELECT
  'retail',
  COUNT(*),
  SUM(target),
  ROUND(SUM(target) / COUNT(*) * 100, 1)
FROM retail.customer_data

UNION ALL

-- Mining
SELECT
  'mining',
  COUNT(*),
  SUM(target),
  ROUND(SUM(target) / COUNT(*) * 100, 1)
FROM mining.equipment_data

ORDER BY industry;


-- ============================================================
-- Feature distributions (run per industry as needed)
-- ============================================================

-- Banking: credit score distribution by default status
-- SELECT
--   target,
--   COUNT(*)                          AS n,
--   ROUND(AVG(credit_score), 0)       AS avg_credit_score,
--   ROUND(AVG(annual_income), 0)      AS avg_annual_income,
--   ROUND(AVG(debt_to_income_ratio), 3) AS avg_dti,
--   ROUND(AVG(num_late_payments), 1)  AS avg_late_payments
-- FROM banking.credit_data
-- GROUP BY target;

-- Insurance: fraud indicators
-- SELECT
--   target,
--   COUNT(*)                          AS n,
--   ROUND(AVG(claim_amount), 2)       AS avg_claim_amount,
--   ROUND(AVG(days_to_report), 1)     AS avg_days_to_report,
--   ROUND(AVG(witness_present), 2)    AS witness_rate,
--   ROUND(AVG(police_report_filed), 2) AS police_report_rate
-- FROM insurance.claim_data
-- GROUP BY target;

-- Retail: churn indicators
-- SELECT
--   target,
--   COUNT(*)                              AS n,
--   ROUND(AVG(monthly_spend_zar), 2)      AS avg_monthly_spend,
--   ROUND(AVG(days_since_last_purchase), 1) AS avg_days_since_purchase,
--   ROUND(AVG(num_support_tickets), 1)    AS avg_support_tickets,
--   ROUND(AVG(num_orders_last_6m), 1)     AS avg_orders_6m
-- FROM retail.customer_data
-- GROUP BY target;

-- Mining: failure indicators
-- SELECT
--   target,
--   COUNT(*)                              AS n,
--   ROUND(AVG(temperature_celsius), 1)    AS avg_temperature,
--   ROUND(AVG(vibration_mm_s), 2)         AS avg_vibration,
--   ROUND(AVG(days_since_maintenance), 0) AS avg_days_since_maint,
--   ROUND(AVG(operating_hours), 0)        AS avg_operating_hours
-- FROM mining.equipment_data
-- GROUP BY target;
