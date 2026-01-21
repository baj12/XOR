#!/usr/bin/env python3
"""
Add metadata column to experiments table and populate with random metadata.
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from db_connection import DatabaseConnection

# Random metadata generators
USERS = ['alice', 'bob', 'charlie', 'diana', 'eve', 'frank']
TAGS = ['baseline', 'optimization', 'hyperparameter-search', 'ablation', 'validation',
        'production', 'debug', 'test', 'comparison', 'final-model']
PURPOSES = ['research', 'production', 'testing', 'benchmarking', 'validation', 'debugging']
PRIORITIES = ['low', 'medium', 'high', 'critical']
ENVIRONMENTS = ['dev', 'staging', 'production', 'test']
GPU_TYPES = ['CPU', 'Tesla V100', 'RTX 3090', 'A100', 'Apple M1', 'Apple M2', 'Apple M3']

def generate_random_metadata():
    """Generate random metadata JSON"""
    metadata = {
        'user': random.choice(USERS),
        'tags': random.sample(TAGS, k=random.randint(1, 3)),
        'purpose': random.choice(PURPOSES),
        'priority': random.choice(PRIORITIES),
        'environment': random.choice(ENVIRONMENTS),
        'gpu_type': random.choice(GPU_TYPES),
        'notes': f"Automated run {random.randint(1000, 9999)}",
        'seed': random.randint(1, 100000),
        'version': f"v{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,20)}"
    }

    # Add some optional fields randomly
    if random.random() > 0.5:
        metadata['cost_estimate'] = round(random.uniform(0.1, 100.0), 2)

    if random.random() > 0.5:
        metadata['dataset_version'] = f"dataset_v{random.randint(1,10)}"

    if random.random() > 0.7:
        metadata['branch'] = random.choice(['main', 'develop', 'feature/new-model', 'hotfix/bug-123'])

    if random.random() > 0.6:
        metadata['reviewer'] = random.choice(USERS)

    return json.dumps(metadata, indent=2)

def add_metadata_column(db):
    """Add metadata column if it doesn't exist"""
    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Check if column exists
        cursor.execute("""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'xor_project'
            AND TABLE_NAME = 'experiments'
            AND COLUMN_NAME = 'metadata'
        """)

        if cursor.fetchone()[0] > 0:
            print("✓ 'metadata' column already exists")
            return True

        # Add the column
        print("Adding 'metadata' column to experiments table...")
        cursor.execute("""
            ALTER TABLE experiments
            ADD COLUMN metadata TEXT COMMENT 'JSON metadata for experiment'
        """)
        print("✓ Column added successfully")
        return True

def populate_metadata(db, force=False):
    """Populate metadata for experiments that don't have it"""
    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Get experiments without metadata
        if force:
            cursor.execute("SELECT id, experiment_id FROM experiments")
            print("Populating ALL experiments with new random metadata...")
        else:
            cursor.execute("""
                SELECT id, experiment_id FROM experiments
                WHERE metadata IS NULL OR metadata = ''
            """)
            print("Populating experiments with missing metadata...")

        experiments = cursor.fetchall()

        if not experiments:
            print("No experiments need metadata updates")
            return 0

        print(f"Found {len(experiments)} experiments to update")

        # Update each experiment
        updated = 0
        for exp_id, exp_name in experiments:
            metadata = generate_random_metadata()

            cursor.execute(
                "UPDATE experiments SET metadata = %s WHERE id = %s",
                (metadata, exp_id)
            )
            updated += 1

            if updated % 100 == 0:
                print(f"  Updated {updated}/{len(experiments)} experiments...")

        print(f"✓ Successfully updated {updated} experiments")
        return updated

def show_sample_metadata(db, n=5):
    """Show sample metadata from updated experiments"""
    with db.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT experiment_id, experiment_type, metadata
            FROM experiments
            WHERE metadata IS NOT NULL
            LIMIT %s
        """, (n,))

        results = cursor.fetchall()

        if not results:
            print("No experiments with metadata found")
            return

        print(f"\nSample metadata from {len(results)} experiments:")
        print("=" * 80)

        for exp_id, exp_type, metadata_json in results:
            print(f"\nExperiment: {exp_id} (type: {exp_type})")
            print("-" * 80)

            # Pretty print the JSON
            try:
                metadata = json.loads(metadata_json)
                print(json.dumps(metadata, indent=2))
            except json.JSONDecodeError:
                print(f"  Raw: {metadata_json}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Add and populate experiment metadata')
    parser.add_argument('--force', action='store_true',
                       help='Force update ALL experiments with new random metadata')
    parser.add_argument('--show-only', action='store_true',
                       help='Only show sample metadata, do not update')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of sample experiments to show (default: 5)')

    args = parser.parse_args()

    db = DatabaseConnection(backend='mariadb')

    if args.show_only:
        show_sample_metadata(db, n=args.samples)
        return

    # Add column if needed
    add_metadata_column(db)

    # Populate metadata
    updated = populate_metadata(db, force=args.force)

    # Show samples
    if updated > 0:
        show_sample_metadata(db, n=min(args.samples, updated))

if __name__ == "__main__":
    main()
