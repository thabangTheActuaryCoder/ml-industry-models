-- ============================================================
-- Insurance: Claim Fraud Detection Dataset
-- 8,000 synthetic rows, ~8-10% fraud rate (imbalanced)
-- ============================================================

USE CATALOG ml_models;
USE SCHEMA insurance;

CREATE OR REPLACE TABLE claim_data (
  claim_amount          DOUBLE    COMMENT 'Claim amount in ZAR',
  policy_tenure_months  INT       COMMENT 'Months the policy has been active (1-239)',
  customer_age          INT       COMMENT 'Customer age in years (18-74)',
  num_prior_claims      INT       COMMENT 'Number of prior claims filed',
  premium_amount        DOUBLE    COMMENT 'Monthly premium in ZAR',
  days_to_report        INT       COMMENT 'Days between incident and claim report (0-89)',
  witness_present       INT       COMMENT 'Whether a witness was present: 0 = no, 1 = yes',
  police_report_filed   INT       COMMENT 'Whether a police report was filed: 0 = no, 1 = yes',
  incident_type         STRING    COMMENT 'Type of incident: Collision, Theft, Fire, Vandalism, Weather Damage, Other',
  vehicle_age_years     INT       COMMENT 'Age of the insured vehicle in years (0-24)',
  province              STRING    COMMENT 'South African province',
  target                INT       COMMENT 'Fraud flag: 0 = legitimate, 1 = fraud'
)
USING DELTA
COMMENT 'Synthetic claim fraud detection data for the insurance industry'
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true'
);

-- ============================================================
-- Load data from CSV uploaded to a Databricks volume
-- ============================================================
-- COPY INTO claim_data
-- FROM '/Volumes/ml_models/insurance/landing/claim_data.csv'
-- FILEFORMAT = CSV
-- FORMAT_OPTIONS (
--   'header'    = 'true',
--   'inferSchema' = 'true'
-- )
-- COPY_OPTIONS ('mergeSchema' = 'true');
