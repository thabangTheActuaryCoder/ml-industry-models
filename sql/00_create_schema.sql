-- ============================================================
-- Create the catalog and schema for ML industry models
-- Run this once before creating any tables
-- ============================================================

CREATE CATALOG IF NOT EXISTS ml_models;

USE CATALOG ml_models;

CREATE SCHEMA IF NOT EXISTS banking
  COMMENT 'Credit scoring model data for the banking industry';

CREATE SCHEMA IF NOT EXISTS insurance
  COMMENT 'Claim fraud detection model data for the insurance industry';

CREATE SCHEMA IF NOT EXISTS retail
  COMMENT 'Customer churn prediction model data for the retail industry';

CREATE SCHEMA IF NOT EXISTS mining
  COMMENT 'Equipment failure prediction model data for the mining industry';
