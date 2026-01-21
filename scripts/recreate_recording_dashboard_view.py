#!/usr/bin/env python3
"""
Recreate the recording_dashboard view with metadata column included.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from db_connection import DatabaseConnection

def main():
    db = DatabaseConnection(backend='mariadb')

    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Try to show current view definition
        try:
            print("Current view definition:")
            print("=" * 80)
            cursor.execute("SHOW CREATE VIEW recording_dashboard")
            result = cursor.fetchone()
            if result:
                print(result[1])
            print("\n")
        except Exception as e:
            print(f"No existing view found (this is OK): {e}\n")

        # Drop the old view if it exists
        print("Dropping old view if it exists...")
        cursor.execute("DROP VIEW IF EXISTS recording_dashboard")
        print("✓ Old view dropped (or didn't exist)\n")

        # Create new view with metadata column (using actual column names from recording_sessions)
        print("Creating new view with metadata column...")
        cursor.execute("""
            CREATE VIEW recording_dashboard AS
            SELECT
                rs.session_id,
                rs.recording_date,
                rs.channel_1_source,
                rs.channel_2_source,
                rs.experiment_id,
                rs.researcher_name,
                rs.faraday_cage_used,
                rs.weather_temperature_c,
                rs.weather_conditions,
                rs.metadata_complete,
                rs.quality_approved,
                rs.imported_to_features_db,
                rs.processed_for_training,
                rs.metadata,
                COUNT(qv.id) AS qc_visualizations_count,
                rs.created_at,
                rs.updated_at
            FROM recording_sessions rs
            LEFT JOIN qc_visualizations qv ON rs.session_id = qv.session_id
            GROUP BY rs.session_id
            ORDER BY rs.recording_date DESC
        """)

        print("✓ New view created successfully\n")

        # Test the new view
        print("Testing new view...")
        cursor.execute("""
            SELECT session_id, metadata_complete, metadata
            FROM recording_dashboard
            LIMIT 3
        """)

        results = cursor.fetchall()
        print(f"✓ View working - found {len(results)} rows")

        for row in results:
            session_id, metadata_complete, metadata_json = row
            has_metadata = "YES" if metadata_json else "NO"
            print(f"  {session_id}: metadata_complete={metadata_complete}, has_metadata={has_metadata}")

        print("\n✓ recording_dashboard view successfully recreated with metadata column!")

if __name__ == "__main__":
    main()
