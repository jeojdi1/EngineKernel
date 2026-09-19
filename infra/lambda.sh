#!/usr/bin/env bash
# Thin wrapper over the Lambda Cloud API. Key is read from ~/.lambda_key.
set -euo pipefail

KEY_FILE="${LAMBDA_KEY_FILE:-$HOME/.lambda_key}"
[ -n "${LAMBDA_API_KEY:-}" ] || LAMBDA_API_KEY="$(cat "$KEY_FILE")"
API="https://cloud.lambda.ai/api/v1"
auth=(-H "Authorization: Bearer ${LAMBDA_API_KEY}")

case "${1:-help}" in
  types)   # which H100 capacity is actually available, and where
    curl -sS "${auth[@]}" "$API/instance-types" \
      | python3 -c '
import json,sys
d=json.load(sys.stdin)["data"]
for k,v in sorted(d.items()):
    regions=[r["name"] for r in v["regions_with_capacity_available"]]
    if regions:
        price=v["instance_type"]["price_cents_per_hour"]/100
        print(f"{k:34s} ${price:6.2f}/hr  {regions}")'
    ;;
  keys)
    curl -sS "${auth[@]}" "$API/ssh-keys" | python3 -m json.tool
    ;;
  ls)
    curl -sS "${auth[@]}" "$API/instances" | python3 -m json.tool
    ;;
  launch)  # launch <instance-type> <region> <ssh-key-name>
    curl -sS "${auth[@]}" -H 'Content-Type: application/json' \
      -d "{\"region_name\":\"$3\",\"instance_type_name\":\"$2\",\"ssh_key_names\":[\"$4\"],\"quantity\":1,\"name\":\"enginekernel\"}" \
      "$API/instance-operations/launch" | python3 -m json.tool
    ;;
  ip)
    curl -sS "${auth[@]}" "$API/instances" \
      | python3 -c 'import json,sys;[print(i["id"],i["status"],i.get("ip")) for i in json.load(sys.stdin)["data"]]'
    ;;
  kill)    # kill <instance-id>
    curl -sS "${auth[@]}" -H 'Content-Type: application/json' \
      -d "{\"instance_ids\":[\"$2\"]}" "$API/instance-operations/terminate" | python3 -m json.tool
    ;;
  *) sed -n '2,12p' "$0" ; echo "usage: $0 {types|keys|ls|ip|launch|kill}" ;;
esac
