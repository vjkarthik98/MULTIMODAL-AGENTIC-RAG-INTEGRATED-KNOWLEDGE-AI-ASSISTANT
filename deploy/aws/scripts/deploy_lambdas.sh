#!/usr/bin/env bash
# Deploy (or update) the MAGIK wake-gateway and idle-stop Lambdas.
#
# Idempotent: safe to re-run. Creates what is missing, updates what exists.
#
# Run from AWS CloudShell (already authenticated, no local setup) or any shell
# with AWS credentials that can manage IAM/Lambda/EventBridge:
#
#     cd deploy/aws/scripts && bash deploy_lambdas.sh
#
# Requires APP_URL to be set once the app endpoint is known. Before a domain
# exists that is the instance's Elastic IP; afterwards it is the HTTPS
# subdomain. Re-run this script after changing it — the value is baked into the
# Lambda's environment, not read at request time.

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
ACCOUNT_ID="${ACCOUNT_ID:-537557168406}"
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_ID="${INSTANCE_ID:-i-02efa81c8876a014e}"
INSTANCE_TAG="${INSTANCE_TAG:-magik-prod}"

# Where the wake gateway redirects to once /health answers.
# Pre-domain:  http://<elastic-ip>:8000
# Post-domain: https://magik-app.yourdomain.com
APP_URL="${APP_URL:-http://3.208.159.124:8000}"

IDLE_MINUTES="${IDLE_MINUTES:-20}"
MIN_UPTIME_MINUTES="${MIN_UPTIME_MINUTES:-15}"

WAKE_FN="magik-wake-gateway"
IDLE_FN="magik-idle-stop"
WAKE_ROLE="magik-wake-gateway-role"
IDLE_ROLE="magik-idle-stop-role"
SCHEDULE_RULE="magik-idle-stop-schedule"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "${BUILD_DIR}"' EXIT

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()  { printf '    \033[0;32mok\033[0m %s\n' "$*"; }

LAMBDA_TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

# ── Helpers ──────────────────────────────────────────────────────────────────
ensure_role() {
  local role="$1" policy_file="$2" policy_name="$3"
  if aws iam get-role --role-name "$role" >/dev/null 2>&1; then
    ok "role $role exists"
  else
    aws iam create-role --role-name "$role" \
      --assume-role-policy-document "$LAMBDA_TRUST" \
      --description "MAGIK Phase 30 scale-to-zero" >/dev/null
    ok "created role $role"
    # IAM is eventually consistent; Lambda creation fails if the role is not
    # yet visible to the Lambda service.
    sleep 10
  fi

  aws iam attach-role-policy --role-name "$role" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole >/dev/null
  ok "attached AWSLambdaBasicExecutionRole"

  # Strip the JSON "_comment" key — valid in our source files for readability,
  # rejected by IAM.
  python3 - "$policy_file" > "${BUILD_DIR}/${policy_name}.json" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
doc.pop("_comment", None)
json.dump(doc, sys.stdout)
PY

  aws iam put-role-policy --role-name "$role" \
    --policy-name "$policy_name" \
    --policy-document "file://${BUILD_DIR}/${policy_name}.json" >/dev/null
  ok "inline policy $policy_name applied"
}

deploy_fn() {
  local fn="$1" src_dir="$2" role="$3" env_vars="$4" timeout="$5"
  local zip="${BUILD_DIR}/${fn}.zip"

  ( cd "$src_dir" && zip -qr "$zip" handler.py )

  local role_arn="arn:aws:iam::${ACCOUNT_ID}:role/${role}"

  if aws lambda get-function --function-name "$fn" --region "$REGION" >/dev/null 2>&1; then
    aws lambda update-function-code --function-name "$fn" \
      --zip-file "fileb://${zip}" --region "$REGION" >/dev/null
    aws lambda wait function-updated --function-name "$fn" --region "$REGION"
    aws lambda update-function-configuration --function-name "$fn" \
      --environment "$env_vars" --timeout "$timeout" --memory-size 256 \
      --region "$REGION" >/dev/null
    aws lambda wait function-updated --function-name "$fn" --region "$REGION"
    ok "updated $fn"
  else
    aws lambda create-function --function-name "$fn" \
      --runtime python3.12 --handler handler.handler \
      --role "$role_arn" --zip-file "fileb://${zip}" \
      --environment "$env_vars" --timeout "$timeout" --memory-size 256 \
      --region "$REGION" >/dev/null
    aws lambda wait function-active --function-name "$fn" --region "$REGION"
    ok "created $fn"
  fi
}

# ── 1. Wake gateway ──────────────────────────────────────────────────────────
say "Wake gateway — IAM role"
ensure_role "$WAKE_ROLE" "${AWS_DIR}/iam/lambda-wake-gateway-permissions.json" "magik-wake-gateway-permissions"

say "Wake gateway — Lambda"
deploy_fn "$WAKE_FN" "${AWS_DIR}/lambda/wake_gateway" "$WAKE_ROLE" \
  "Variables={EC2_INSTANCE_TAG=${INSTANCE_TAG},APP_URL=${APP_URL},HEALTH_TIMEOUT_S=3,REFRESH_SECONDS=7}" \
  15

say "Wake gateway — public Function URL"
if ! aws lambda get-function-url-config --function-name "$WAKE_FN" --region "$REGION" >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name "$WAKE_FN" \
    --auth-type NONE --region "$REGION" >/dev/null
  # A Function URL with auth NONE still needs an explicit resource policy
  # allowing public invoke — without it every request returns 403.
  aws lambda add-permission --function-name "$WAKE_FN" \
    --statement-id FunctionURLAllowPublicAccess \
    --action lambda:InvokeFunctionUrl --principal '*' \
    --function-url-auth-type NONE --region "$REGION" >/dev/null 2>&1 || true
  ok "created public Function URL"
else
  ok "Function URL already configured"
fi

WAKE_URL=$(aws lambda get-function-url-config --function-name "$WAKE_FN" \
  --region "$REGION" --query FunctionUrl --output text)

# ── 2. Idle stop ─────────────────────────────────────────────────────────────
say "Idle stop — IAM role"
ensure_role "$IDLE_ROLE" "${AWS_DIR}/iam/lambda-idle-stop-permissions.json" "magik-idle-stop-permissions"

say "Idle stop — Lambda"
deploy_fn "$IDLE_FN" "${AWS_DIR}/lambda/idle_stop" "$IDLE_ROLE" \
  "Variables={EC2_INSTANCE_TAG=${INSTANCE_TAG},IDLE_MINUTES=${IDLE_MINUTES},MIN_UPTIME_MINUTES=${MIN_UPTIME_MINUTES},NETWORK_IN_THRESHOLD_BYTES=1000000,DRY_RUN=false}" \
  60

say "Idle stop — EventBridge schedule (every 5 minutes)"
aws events put-rule --name "$SCHEDULE_RULE" \
  --schedule-expression "rate(5 minutes)" \
  --description "Poll MAGIK GPU instance for idleness and stop it" \
  --region "$REGION" >/dev/null
ok "rule $SCHEDULE_RULE"

aws lambda add-permission --function-name "$IDLE_FN" \
  --statement-id "${SCHEDULE_RULE}-invoke" \
  --action lambda:InvokeFunction --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${SCHEDULE_RULE}" \
  --region "$REGION" >/dev/null 2>&1 || true
ok "EventBridge may invoke $IDLE_FN"

aws events put-targets --rule "$SCHEDULE_RULE" \
  --targets "Id=1,Arn=arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${IDLE_FN}" \
  --region "$REGION" >/dev/null
ok "target wired"

# ── Summary ──────────────────────────────────────────────────────────────────
cat <<SUMMARY

────────────────────────────────────────────────────────────────────────
 Deployed.

   Wake gateway URL : ${WAKE_URL}
   Redirects to     : ${APP_URL}
   Idle stop        : every 5 min; stops after ${IDLE_MINUTES}m idle,
                      never before ${MIN_UPTIME_MINUTES}m uptime

 Verify (safe to run now):

   1. Stop the instance:
        aws ec2 stop-instances --instance-ids ${INSTANCE_ID}
   2. Open ${WAKE_URL} in a browser
        -> "Waking up MAGIK", instance transitions to running
        -> redirects to ${APP_URL} once /health answers
   3. Leave it idle and confirm it stops itself within ~${IDLE_MINUTES}-25 min:
        aws ec2 describe-instances --instance-ids ${INSTANCE_ID} \\
          --query 'Reservations[0].Instances[0].State.Name' --output text

 Logs:
   aws logs tail /aws/lambda/${WAKE_FN} --follow
   aws logs tail /aws/lambda/${IDLE_FN} --follow

 NOTE: the wake gateway health-checks APP_URL from outside AWS, so the app
 port must be reachable. Pre-domain that means 8000 open in the security
 group; once Caddy + the domain are live, switch APP_URL to the HTTPS
 subdomain, re-run this script, and close 8000.
────────────────────────────────────────────────────────────────────────
SUMMARY
