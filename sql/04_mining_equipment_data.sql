-- ============================================================
-- Mining: Equipment Failure Prediction Dataset
-- 15,000 synthetic rows, ~10-11% failure rate
-- ============================================================

USE CATALOG ml_models;
USE SCHEMA mining;

CREATE OR REPLACE TABLE equipment_data (
  temperature_celsius           DOUBLE    COMMENT 'Equipment operating temperature in Celsius',
  vibration_mm_s                DOUBLE    COMMENT 'Vibration level in mm/s',
  oil_pressure_kpa              DOUBLE    COMMENT 'Oil pressure in kPa',
  rpm                           INT       COMMENT 'Revolutions per minute (500-2999)',
  operating_hours               INT       COMMENT 'Total operating hours (100-49,999)',
  days_since_maintenance        INT       COMMENT 'Days since last scheduled maintenance (0-364)',
  load_percentage               DOUBLE    COMMENT 'Current load as percentage of capacity (0-100)',
  ambient_temperature_celsius   DOUBLE    COMMENT 'Ambient temperature at the mine site in Celsius',
  hydraulic_pressure_kpa        DOUBLE    COMMENT 'Hydraulic system pressure in kPa',
  num_previous_failures         INT       COMMENT 'Number of previously recorded failures',
  equipment_type                STRING    COMMENT 'Equipment type: Haul Truck, Excavator, Drill Rig, Conveyor Belt, Crusher, Load Haul Dumper, Grader, Bulldozer',
  mine_type                     STRING    COMMENT 'Mine type: Open Pit, Underground, Alluvial',
  shift                         STRING    COMMENT 'Shift type: Day, Night, Extended',
  province                      STRING    COMMENT 'South African province',
  target                        INT       COMMENT 'Failure flag: 0 = operational, 1 = failure'
)
USING DELTA
COMMENT 'Synthetic equipment failure data for the mining industry'
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true'
);

-- ============================================================
-- Load data from CSV uploaded to a Databricks volume
-- ============================================================
-- COPY INTO equipment_data
-- FROM '/Volumes/ml_models/mining/landing/equipment_data.csv'
-- FILEFORMAT = CSV
-- FORMAT_OPTIONS (
--   'header'    = 'true',
--   'inferSchema' = 'true'
-- )
-- COPY_OPTIONS ('mergeSchema' = 'true');
