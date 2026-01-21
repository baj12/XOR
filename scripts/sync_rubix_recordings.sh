#!/bin/bash
#
# Sync Rubix44 Recordings to CIH Storage
#
# This script transfers recordings from rubix44-recorder (10.0.0.58)
# to local CIH storage (/Volumes/CIH/mora/moraWav/rubix/) and updates
# the MariaDB database with metadata.
#
# Usage:
#   ./scripts/sync_rubix_recordings.sh              # Transfer new recordings
#   ./scripts/sync_rubix_recordings.sh --delete     # Transfer and delete from server
#   ./scripts/sync_rubix_recordings.sh --dry-run    # Preview what would transfer
#   ./scripts/sync_rubix_recordings.sh --help       # Show help
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Python executable
PYTHON_ENV="/Users/bernd/Library/r-miniconda/envs/xorProject/bin/python"

# Check if Python environment exists
if [ ! -f "$PYTHON_ENV" ]; then
    echo -e "${RED}Error: Python environment not found at $PYTHON_ENV${NC}"
    echo "Please activate the xorProject conda environment first."
    exit 1
fi

# Check if CIH volume is mounted
if [ ! -d "/Volumes/CIH/mora/moraWav/rubix" ]; then
    echo -e "${RED}Error: CIH volume not mounted${NC}"
    echo "Expected directory: /Volumes/CIH/mora/moraWav/rubix"
    echo "Please mount the network share first."
    exit 1
fi

# Check API connectivity
echo -e "${BLUE}Checking rubix44-recorder API...${NC}"
if ! curl -s -f http://10.0.0.58:5000/api/v1/health > /dev/null 2>&1; then
    echo -e "${RED}Error: Cannot connect to rubix44-recorder API at 10.0.0.58:5000${NC}"
    echo "Please check that the rubix44-recorder server is running."
    exit 1
fi
echo -e "${GREEN}✓ API is healthy${NC}"

# Parse arguments
DELETE_FLAG=""
DRY_RUN_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --delete|--delete-after-transfer)
            DELETE_FLAG="--delete-after-transfer"
            echo -e "${YELLOW}⚠ Delete mode enabled - will remove recordings from server after transfer${NC}"
            shift
            ;;
        --dry-run)
            DRY_RUN_FLAG="--dry-run"
            echo -e "${YELLOW}Dry run mode - no actual transfers will occur${NC}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --delete             Delete recordings from server after successful transfer"
            echo "  --dry-run            Preview what would be transferred without doing it"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                   Transfer new recordings"
            echo "  $0 --delete          Transfer and delete from server"
            echo "  $0 --dry-run         Preview transfers"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run transfer script
echo -e "${BLUE}Starting transfer...${NC}"
echo ""

cd "$PROJECT_DIR"

if $PYTHON_ENV scripts/transfer_rubix_recordings.py $DELETE_FLAG $DRY_RUN_FLAG; then
    echo ""
    echo -e "${GREEN}✓ Transfer completed successfully${NC}"

    # Show disk usage
    echo ""
    echo -e "${BLUE}Storage usage:${NC}"
    du -sh /Volumes/CIH/mora/moraWav/rubix/ 2>/dev/null || echo "Could not determine disk usage"

    # Show recording count
    echo ""
    echo -e "${BLUE}Checking database...${NC}"
    $PYTHON_ENV << 'EOF'
import sys
sys.path.insert(0, 'src')
from db_connection import DatabaseConnection

try:
    db = DatabaseConnection(backend='mariadb')
    with db.get_connection() as conn:
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT COUNT(*) as total,
                   SUM(duration_seconds) / 3600 as total_hours,
                   SUM(file_size_bytes) / 1024 / 1024 / 1024 as total_gb
            FROM recording_sessions
        """)
        stats = cursor.fetchone()

        print(f"  Recordings: {stats['total']}")
        print(f"  Total duration: {stats['total_hours']:.1f} hours")
        print(f"  Total size: {stats['total_gb']:.2f} GB")
except Exception as e:
    print(f"  Could not query database: {e}")
EOF

    exit 0
else
    echo ""
    echo -e "${RED}✗ Transfer failed${NC}"
    echo "Check the error messages above for details."
    exit 1
fi
