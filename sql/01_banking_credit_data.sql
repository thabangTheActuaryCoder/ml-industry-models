-- ============================================================
-- Banking: Credit Scoring Dataset
-- 10,000 synthetic rows, ~15-18% default rate
-- ============================================================

USE CATALOG ml_models;
USE SCHEMA banking;

CREATE OR REPLACE TABLE credit_data (
  age                           INT       COMMENT 'Customer age in years (21-64)',
  annual_income                 BIGINT    COMMENT 'Annual income in ZAR',
  employment_years              INT       COMMENT 'Years of employment (0-29)',
  loan_amount                   BIGINT    COMMENT 'Requested loan amount in ZAR',
  credit_score                  INT       COMMENT 'Credit bureau score (300-849)',
  num_late_payments             INT       COMMENT 'Number of late payments on record',
  debt_to_income_ratio          DOUBLE    COMMENT 'Debt-to-income ratio (0.05-0.80)',
  num_open_accounts             INT       COMMENT 'Number of open credit accounts (1-14)',
  months_since_last_delinquency INT       COMMENT 'Months since last delinquency (0-119)',
  education_level               STRING    COMMENT 'Highest qualification: Matric, Diploma, Bachelors, Honours, Masters, Doctorate',
  employment_type               STRING    COMMENT 'Employment type: Permanent, Contract, Self-employed, Part-time',
  province                      STRING    COMMENT 'South African province',
  target                        INT       COMMENT 'Default flag: 0 = no default, 1 = default'
)
USING DELTA
COMMENT 'Synthetic credit scoring data for the banking industry'
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true'
);

-- ============================================================
-- Load data from CSV uploaded to a Databricks volume
-- Adjust the volume path to match your workspace
-- ============================================================
-- COPY INTO credit_data
-- FROM '/Volumes/ml_models/banking/landing/credit_data.csv'
-- FILEFORMAT = CSV
-- FORMAT_OPTIONS (
--   'header'    = 'true',
--   'inferSchema' = 'true'
-- )
-- COPY_OPTIONS ('mergeSchema' = 'true');
