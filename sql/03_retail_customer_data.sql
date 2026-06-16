-- ============================================================
-- Retail: Customer Churn Prediction Dataset
-- 12,000 synthetic rows, ~20-24% churn rate
-- ============================================================

USE CATALOG ml_models;
USE SCHEMA retail;

CREATE OR REPLACE TABLE customer_data (
  monthly_spend_zar         DOUBLE    COMMENT 'Average monthly spend in ZAR',
  days_since_last_purchase  INT       COMMENT 'Days since last purchase (0-364)',
  num_support_tickets       INT       COMMENT 'Number of support tickets raised',
  loyalty_points            INT       COMMENT 'Accumulated loyalty points (0-49,999)',
  account_age_months        INT       COMMENT 'Months since account creation (1-119)',
  num_returns_last_year     INT       COMMENT 'Number of product returns in the last year',
  avg_order_value_zar       DOUBLE    COMMENT 'Average order value in ZAR',
  num_orders_last_6m        INT       COMMENT 'Number of orders in the last 6 months (0-29)',
  discount_usage_rate       DOUBLE    COMMENT 'Proportion of orders using a discount (0-1)',
  membership_tier           STRING    COMMENT 'Loyalty tier: Bronze, Silver, Gold, Platinum',
  preferred_channel         STRING    COMMENT 'Preferred shopping channel: Online, In-store, Both',
  province                  STRING    COMMENT 'South African province',
  target                    INT       COMMENT 'Churn flag: 0 = retained, 1 = churned'
)
USING DELTA
COMMENT 'Synthetic customer churn data for the retail industry'
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true'
);

-- ============================================================
-- Load data from CSV uploaded to a Databricks volume
-- ============================================================
-- COPY INTO customer_data
-- FROM '/Volumes/ml_models/retail/landing/customer_data.csv'
-- FILEFORMAT = CSV
-- FORMAT_OPTIONS (
--   'header'    = 'true',
--   'inferSchema' = 'true'
-- )
-- COPY_OPTIONS ('mergeSchema' = 'true');
