#!/bin/bash
# Test script for rubix44 server health and logging endpoints
# Usage: ./scripts/test_rubix44_health.sh [server_url]

SERVER=${1:-"http://10.0.0.58:5000"}
echo "Testing rubix44 server at: $SERVER"
echo "=" * 60

# Test 1: Basic connectivity
echo -e "\n1. Testing basic connectivity..."
if curl -s --max-time 5 "$SERVER/" > /dev/null 2>&1; then
    echo "✅ Server is reachable"
else
    echo "❌ Server is not reachable"
    exit 1
fi

# Test 2: Health endpoint
echo -e "\n2. Checking API health..."
curl -s "$SERVER/api/v1/health" | python3 -m json.tool || echo "❌ Health endpoint failed"

# Test 3: System health with crash history
echo -e "\n3. Checking system health and crash history..."
curl -s "$SERVER/api/v1/system/health" | python3 -m json.tool || echo "❌ System health endpoint failed"

# Test 4: List log files
echo -e "\n4. Listing available log files..."
curl -s "$SERVER/api/v1/logs" | python3 -m json.tool || echo "❌ Logs list endpoint failed"

# Test 5: Read last 50 lines of main log
echo -e "\n5. Reading last 50 lines of app.log..."
curl -s "$SERVER/api/v1/logs/app.log?lines=50" || echo "❌ Log read failed"

# Test 6: Read error log
echo -e "\n6. Reading last 20 lines of errors.log..."
curl -s "$SERVER/api/v1/logs/errors.log?lines=20" || echo "❌ Error log read failed"

# Test 7: Check if recording can be started
echo -e "\n7. Testing recording start endpoint (will not start, just check availability)..."
response=$(curl -s -o /dev/null -w "%{http_code}" "$SERVER/api/v1/recordings/status")
if [ "$response" == "200" ]; then
    echo "✅ Recording status endpoint working (HTTP $response)"
elif [ "$response" == "404" ]; then
    echo "⚠️  Recording status endpoint not found (HTTP $response)"
else
    echo "❌ Recording status endpoint error (HTTP $response)"
fi

echo -e "\n=" * 60
echo "Health check complete!"
