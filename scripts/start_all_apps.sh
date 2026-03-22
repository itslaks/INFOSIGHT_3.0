#!/usr/bin/env bash
# Start all INFOSIGHT 3.0 app modules in standalone mode (Linux/Mac)
# Usage: ./scripts/start_all_apps.sh

set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

declare -A appMap=(
  [infocrypt]=5001
  [cybersentry_ai]=5002
  [donna]=5003
  [enscan]=5004
  [filescanner]=5005
  [infosight_ai]=5006
  [inkwell_ai]=5007
  [nova_ai]=5008
  [osint]=5009
  [portscanner]=5010
  [snapspeak_ai]=5011
  [trueshot_ai]=5012
  [webseeker]=5013
)

mkdir -p logs

for app in "${!appMap[@]}"; do
  port=${appMap[$app]}
  echo "Starting $app on 127.0.0.1:$port"
  APP_HOST=127.0.0.1 APP_PORT=$port nohup python "app/$app.py" > "logs/$app.log" 2>&1 &
  sleep 0.25
done

# Start main gateway server on 5000 by default
gateway_host="${SERVER_HOST:-127.0.0.1}"
gateway_port="${SERVER_PORT:-5000}"

echo "Starting main server gateway on $gateway_host:$gateway_port"
nohup python server.py --mode distributed --host "$gateway_host" --port "$gateway_port" > logs/server.log 2>&1 &

echo "All INFOSIGHT processes launched."

echo "Use 'ps -ef | grep python' or 'pkill -f app/<name>.py' to inspect/stop."