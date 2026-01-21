-- Substance Vocabulary Schema Migration
-- Refactors continuous_experiments to use single substance field per channel
-- instead of redundant source + expected_class fields
--
-- Author: Claude Code
-- Date: 2026-01-04

-- ==============================================================================
-- Step 1: Create substance vocabulary table
-- ==============================================================================

CREATE TABLE IF NOT EXISTS substance_vocabulary (
    id INT AUTO_INCREMENT PRIMARY KEY,
    substance_name VARCHAR(100) NOT NULL UNIQUE,
    class_label INT NOT NULL,
    is_canonical BOOLEAN DEFAULT TRUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_substance_name (substance_name),
    INDEX idx_class_label (class_label)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ==============================================================================
-- Step 2: Populate substance vocabulary with initial values
-- ==============================================================================

INSERT INTO substance_vocabulary (substance_name, class_label, is_canonical, description) VALUES
    -- Class 0: Controls/empty
    ('empty', 0, TRUE, 'Empty beaker - control'),
    ('control', 0, TRUE, 'Control sample'),
    ('water', 0, TRUE, 'Water only'),
    ('air', 0, TRUE, 'Air/nothing'),
    ('blank', 0, TRUE, 'Blank sample'),
    ('nothing', 0, FALSE, 'Alias for empty'),

    -- Class 1: Lavender (including common misspellings)
    ('lavender', 1, TRUE, 'Lavender essential oil'),
    ('lavendar', 1, FALSE, 'Common misspelling of lavender'),
    ('lavander', 1, FALSE, 'Common misspelling of lavender'),

    -- Class 2: Peppermint
    ('peppermint', 2, TRUE, 'Peppermint essential oil'),

    -- Class 3: Eucalyptus
    ('eucalyptus', 3, TRUE, 'Eucalyptus essential oil'),

    -- Class 4: Rosemary
    ('rosemary', 4, TRUE, 'Rosemary essential oil'),

    -- Class 5: Tea Tree
    ('tea_tree', 5, TRUE, 'Tea tree essential oil'),

    -- Class 6: Lemon
    ('lemon', 6, TRUE, 'Lemon essential oil'),

    -- Class 7: Orange
    ('orange', 7, TRUE, 'Orange essential oil')
ON DUPLICATE KEY UPDATE
    class_label = VALUES(class_label),
    description = VALUES(description);

-- ==============================================================================
-- Step 3: Add new substance columns to continuous_experiments
-- ==============================================================================

ALTER TABLE continuous_experiments
    ADD COLUMN channel_1_substance VARCHAR(100) AFTER channel_1_expected_class,
    ADD COLUMN channel_2_substance VARCHAR(100) AFTER channel_2_expected_class;

-- ==============================================================================
-- Step 4: Migrate existing data
-- ==============================================================================

-- Migrate channel 1 data (map source to substance, using lowercase)
UPDATE continuous_experiments
SET channel_1_substance = LOWER(TRIM(REPLACE(channel_1_source, ' ', '_')))
WHERE channel_1_source IS NOT NULL;

-- Migrate channel 2 data
UPDATE continuous_experiments
SET channel_2_substance = LOWER(TRIM(REPLACE(channel_2_source, ' ', '_')))
WHERE channel_2_source IS NOT NULL;

-- ==============================================================================
-- Step 5: Drop old redundant columns
-- ==============================================================================

ALTER TABLE continuous_experiments
    DROP COLUMN channel_1_source,
    DROP COLUMN channel_1_expected_class,
    DROP COLUMN channel_2_source,
    DROP COLUMN channel_2_expected_class;

-- ==============================================================================
-- Step 6: Update Faraday cage default
-- ==============================================================================

ALTER TABLE continuous_experiments
    MODIFY COLUMN faraday_cage_used BOOLEAN DEFAULT TRUE;

-- Update existing records where faraday_cage_used is NULL
UPDATE continuous_experiments
SET faraday_cage_used = TRUE
WHERE faraday_cage_used IS NULL;

-- ==============================================================================
-- Step 7: Add foreign key constraints (optional, for data integrity)
-- ==============================================================================

-- Note: We use VARCHAR instead of FK because the vocabulary can be extended
-- at runtime and we want to allow case-insensitive matching.
-- The application layer enforces vocabulary validation.

-- Add indexes for performance
ALTER TABLE continuous_experiments
    ADD INDEX idx_channel_1_substance (channel_1_substance),
    ADD INDEX idx_channel_2_substance (channel_2_substance);

-- ==============================================================================
-- Step 8: Update experiment_summary view
-- ==============================================================================

DROP VIEW IF EXISTS experiment_summary;

CREATE VIEW experiment_summary AS
SELECT
    e.experiment_id,
    e.experiment_name,
    e.status,
    e.current_cycle,
    e.total_cycles_expected,
    e.channel_1_substance,
    e.channel_2_substance,
    e.current_accuracy,
    e.start_time,
    e.updated_at,
    COUNT(DISTINCT rc.cycle_number) as completed_cycles,
    SUM(CASE WHEN rc.qc_passed = TRUE THEN 1 ELSE 0 END) as qc_passes,
    SUM(CASE WHEN rc.qc_passed = FALSE THEN 1 ELSE 0 END) as qc_fails,
    AVG(rc.qc_separation_score) as avg_separation,
    MAX(rc.model_accuracy) as best_accuracy
FROM continuous_experiments e
LEFT JOIN recording_cycles rc ON e.experiment_id = rc.experiment_id
GROUP BY e.experiment_id, e.experiment_name, e.status, e.current_cycle,
         e.total_cycles_expected, e.channel_1_substance, e.channel_2_substance,
         e.current_accuracy, e.start_time, e.updated_at;

-- ==============================================================================
-- Step 9: Create helper functions (stored procedures)
-- ==============================================================================

DELIMITER $$

-- Function to get class label for a substance
DROP FUNCTION IF EXISTS get_substance_class$$
CREATE FUNCTION get_substance_class(substance_name_input VARCHAR(100))
RETURNS INT
DETERMINISTIC
READS SQL DATA
BEGIN
    DECLARE class_result INT;

    SELECT class_label INTO class_result
    FROM substance_vocabulary
    WHERE LOWER(TRIM(substance_name)) = LOWER(TRIM(substance_name_input))
    LIMIT 1;

    RETURN class_result;
END$$

-- Function to validate substance name
DROP FUNCTION IF EXISTS is_valid_substance$$
CREATE FUNCTION is_valid_substance(substance_name_input VARCHAR(100))
RETURNS BOOLEAN
DETERMINISTIC
READS SQL DATA
BEGIN
    DECLARE count_result INT;

    SELECT COUNT(*) INTO count_result
    FROM substance_vocabulary
    WHERE LOWER(TRIM(substance_name)) = LOWER(TRIM(substance_name_input));

    RETURN count_result > 0;
END$$

DELIMITER ;

-- ==============================================================================
-- Migration Complete
-- ==============================================================================

-- Verify migration
SELECT
    'Substance vocabulary entries' as metric,
    COUNT(*) as value
FROM substance_vocabulary
UNION ALL
SELECT
    'Experiments migrated' as metric,
    COUNT(*) as value
FROM continuous_experiments
WHERE channel_1_substance IS NOT NULL OR channel_2_substance IS NOT NULL
UNION ALL
SELECT
    'Faraday cage default set' as metric,
    COUNT(*) as value
FROM continuous_experiments
WHERE faraday_cage_used = TRUE;

-- Show sample of migrated data
SELECT
    experiment_id,
    experiment_name,
    channel_1_substance,
    channel_2_substance,
    faraday_cage_used,
    status
FROM continuous_experiments
LIMIT 5;
