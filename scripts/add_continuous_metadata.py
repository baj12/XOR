#!/usr/bin/env python3
"""
Add metadata column to continuous_experiments table and populate with random metadata.
"""

import sys
import json
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from db_connection import DatabaseConnection

# Random metadata generators
USERS = ['alice', 'bob', 'charlie', 'diana', 'eve', 'frank', 'bernd', 'researcher1', 'researcher2']
TAGS = ['baseline', 'optimization', 'validation', 'production', 'long-run', 'test',
        'noise-analysis', 'stereo-test', 'rubix44', 'continuous-learning']
PURPOSES = ['research', 'production', 'testing', 'benchmarking', 'validation', 'long-term-study']
PRIORITIES = ['low', 'medium', 'high', 'critical']
ENVIRONMENTS = ['dev', 'staging', 'production', 'test', 'lab']
LOCATIONS = ['lab-bench-1', 'lab-bench-2', 'faraday-cage', 'quiet-room', 'main-lab']
CONDITIONS = ['normal', 'noisy', 'quiet', 'controlled', 'variable']
WEATHER_CONDITIONS = ['sunny', 'cloudy', 'rainy', 'humid', 'dry']

def generate_random_metadata():
    """Generate random metadata JSON"""
    metadata = {
        'user': random.choice(USERS),
        'tags': random.sample(TAGS, k=random.randint(1, 3)),
        'purpose': random.choice(PURPOSES),
        'priority': random.choice(PRIORITIES),
        'environment': random.choice(ENVIRONMENTS),
        'location': random.choice(LOCATIONS),
        'lab_conditions': random.choice(CONDITIONS),
        'run_id': f"run_{random.randint(10000, 99999)}",
        'version': f"v{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,20)}"
    }

    # Add some optional fields randomly
    if random.random() > 0.5:
        metadata['cost_estimate'] = round(random.uniform(1.0, 500.0), 2)
        metadata['cost_currency'] = 'USD'

    if random.random() > 0.5:
        metadata['temperature_celsius'] = round(random.uniform(18.0, 25.0), 1)
        metadata['humidity_percent'] = round(random.uniform(30.0, 70.0), 1)

    if random.random() > 0.6:
        metadata['weather'] = random.choice(WEATHER_CONDITIONS)

    if random.random() > 0.7:
        metadata['collaborators'] = random.sample(USERS, k=random.randint(1, 3))

    if random.random() > 0.5:
        metadata['funding_source'] = random.choice(['grant-A', 'grant-B', 'internal', 'collaboration'])

    if random.random() > 0.6:
        metadata['notes'] = random.choice([
            'Successful run with good separation',
            'Some noise issues detected',
            'Excellent baseline established',
            'Testing new configuration',
            'Long-term stability test',
            'Validation of previous results'
        ])

    if random.random() > 0.5:
        metadata['equipment_version'] = f"rubix44_v{random.randint(1,3)}.{random.randint(0,5)}"

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
            AND TABLE_NAME = 'continuous_experiments'
            AND COLUMN_NAME = 'metadata'
        """)

        if cursor.fetchone()[0] > 0:
            print("✓ 'metadata' column already exists in continuous_experiments")
            return True

        # Add the column
        print("Adding 'metadata' column to continuous_experiments table...")
        cursor.execute("""
            ALTER TABLE continuous_experiments
            ADD COLUMN metadata TEXT COMMENT 'JSON metadata for continuous learning experiment'
        """)
        print("✓ Column added successfully")
        return True

def populate_metadata(db, force=False):
    """Populate metadata for continuous_experiments that don't have it"""
    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Get experiments without metadata
        if force:
            cursor.execute("""
                SELECT experiment_id, experiment_name
                FROM continuous_experiments
            """)
            print("Populating ALL continuous_experiments with new random metadata...")
        else:
            cursor.execute("""
                SELECT experiment_id, experiment_name
                FROM continuous_experiments
                WHERE metadata IS NULL OR metadata = ''
            """)
            print("Populating continuous_experiments with missing metadata...")

        experiments = cursor.fetchall()

        if not experiments:
            print("No continuous_experiments need metadata updates")
            return 0

        print(f"Found {len(experiments)} experiments to update")

        # Update each experiment
        updated = 0
        for exp_id, exp_name in experiments:
            metadata = generate_random_metadata()

            cursor.execute(
                "UPDATE continuous_experiments SET metadata = %s WHERE experiment_id = %s",
                (metadata, exp_id)
            )
            updated += 1
            print(f"  ✓ Updated '{exp_name}' ({exp_id})")

        print(f"\n✓ Successfully updated {updated} continuous_experiments")
        return updated

def show_sample_metadata(db, n=5):
    """Show sample metadata from updated continuous_experiments"""
    with db.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT experiment_id, experiment_name, status, metadata
            FROM continuous_experiments
            WHERE metadata IS NOT NULL
            LIMIT %s
        """, (n,))

        results = cursor.fetchall()

        if not results:
            print("No continuous_experiments with metadata found")
            return

        print(f"\nSample metadata from {len(results)} continuous_experiments:")
        print("=" * 80)

        for exp_id, exp_name, status, metadata_json in results:
            print(f"\nExperiment: {exp_name}")
            print(f"  ID: {exp_id}")
            print(f"  Status: {status}")
            print("-" * 80)

            # Pretty print the JSON
            try:
                metadata = json.loads(metadata_json)
                print(json.dumps(metadata, indent=2))
            except json.JSONDecodeError:
                print(f"  Raw: {metadata_json}")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Add and populate metadata for continuous_experiments table'
    )
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
