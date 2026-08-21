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
# Requires APP_URL to be set to the app endpoint. Re-run this script after
# changing it — the value is baked into the Lambda's environment, not read at
# request time.

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
ACCOUNT_ID="${ACCOUNT_ID:-857194222592}"
REGION="${AWS_REGION:-us-east-1}"
# Rebuilt 2026-08-21 in a new AWS account (was 537557168406) after the prior
# EC2 fleet + EBS volumes were deleted. New magik-prod instance from
# deploy/aws/terraform/'s production_instance_id output. Only used for this
# script's own printed verification commands (the Lambdas resolve the
# instance by tag at runtime, not this default).
INSTANCE_ID="${INSTANCE_ID:-i-09831ac06b063d36f}"
INSTANCE_TAG="${INSTANCE_TAG:-magik-prod}"

# Where the wake gateway redirects to once /health answers. HTTPS via Caddy
# on magik.vk-ai.online — NOT the bare Elastic IP. The Lambda health-checks
# this URL from outside AWS (it runs outside a VPC), so the domain must
# resolve and the cert must be valid, both true since Caddy/Let's Encrypt were
# set up in deploy/aws/README.md's Caddy section.
APP_URL="${APP_URL:-https://magik.vk-ai.online}"

IDLE_MINUTES="${IDLE_MINUTES:-20}"
MIN_UPTIME_MINUTES="${MIN_UPTIME_MINUTES:-15}"

# Minutes the instance can be "running" without /health going green before
# the wake gateway stops repeating the generic "starting up" copy and shows a
# distinct "taking longer than usual" page instead. Model loading normally
# finishes well under this.
STUCK_MINUTES="${STUCK_MINUTES:-6}"

# Optional — empty by default, which is a strict no-op in both handlers (see
# their KUMA_PUSH_URL guards). Set this only once the Phase F Uptime Kuma
# host (monitoring/uptime-kuma/) exists and you have its push-monitor URL —
# never wire this up speculatively, since a set-but-wrong value just means
# silently-failing pushes (harmless, but worth getting right).
KUMA_PUSH_URL="${KUMA_PUSH_URL:-}"

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
  "Variables={EC2_INSTANCE_TAG=${INSTANCE_TAG},APP_URL=${APP_URL},HEALTH_TIMEOUT_S=3,REFRESH_SECONDS=7,STUCK_MINUTES=${STUCK_MINUTES},KUMA_PUSH_URL=${KUMA_PUSH_URL}}" \
  15

say "Wake gateway — public API Gateway HTTP API"
# NOT a Lambda Function URL. Verified in this account (2026-07-29): a
# Function URL with auth-type NONE plus the documented public-invoke resource
# policy statement was confirmed correct via `get-function-url-config` and
# `get-policy` on TWO independently created URLs, and both still returned 403
# Forbidden / AccessDeniedException. Root cause never conclusively identified.
# API Gateway's invoke-permission model (principal apigateway.amazonaws.com,
# action lambda:InvokeFunction) is older, more battle-tested, and worked
# immediately once the permission was attached. handler.py needs no changes —
# both integration types return the same {statusCode, headers, body} shape.
API_NAME="magik-wake-gateway-api"
WAKE_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${WAKE_FN}"

API_ID=$(aws apigatewayv2 get-apis --region "$REGION" \
  --query "Items[?Name=='${API_NAME}'].ApiId" --output text)

if [ -z "$API_ID" ] || [ "$API_ID" = "None" ]; then
  API_ID=$(aws apigatewayv2 create-api --name "$API_NAME" --protocol-type HTTP \
    --target "$WAKE_ARN" --region "$REGION" --query ApiId --output text)
  ok "created API $API_ID"
else
  ok "API $API_ID already exists"
fi

# create-api --target is a "quick create" shortcut; it did not reliably attach
# the Lambda invoke permission in testing. Adding it explicitly and
# idempotently (statement-id collision is treated as already-applied, not an
# error) rather than trusting the shortcut.
aws lambda add-permission --function-name "$WAKE_FN" \
  --statement-id apigateway-invoke --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*/*" \
  --region "$REGION" >/dev/null 2>&1 \
  && ok "apigateway invoke permission attached" \
  || ok "apigateway invoke permission already present"

WAKE_URL=$(aws apigatewayv2 get-apis --region "$REGION" \
  --query "Items[?ApiId=='${API_ID}'].ApiEndpoint" --output text)

# ── 2. Idle stop ─────────────────────────────────────────────────────────────
say "Idle stop — IAM role"
ensure_role "$IDLE_ROLE" "${AWS_DIR}/iam/lambda-idle-stop-permissions.json" "magik-idle-stop-permissions"

GITHUB_REPO="${GITHUB_REPO:-vjkarthik98/MULTIMODAL-AGENTIC-RAG-INTEGRATED-KNOWLEDGE-AI-ASSISTANT}"
GITHUB_RUNNER_LABEL="${GITHUB_RUNNER_LABEL:-gpu}"
GITHUB_TOKEN_PARAM="${GITHUB_TOKEN_PARAM:-/magik/github_actions_pat}"

if ! aws ssm get-parameter --name "$GITHUB_TOKEN_PARAM" --region "$REGION" >/dev/null 2>&1; then
  printf '    \033[0;33mwarn\033[0m %s\n' \
    "${GITHUB_TOKEN_PARAM} not found in SSM — idle-stop will fail SAFE (treat the" \
    "          runner as busy, never stop) until it exists. See README.md:" \
    "          'Runner busy-check token' for how to create and store it."
fi

say "Idle stop — Lambda"
deploy_fn "$IDLE_FN" "${AWS_DIR}/lambda/idle_stop" "$IDLE_ROLE" \
  "Variables={EC2_INSTANCE_TAG=${INSTANCE_TAG},IDLE_MINUTES=${IDLE_MINUTES},MIN_UPTIME_MINUTES=${MIN_UPTIME_MINUTES},NETWORK_IN_THRESHOLD_BYTES=1000000,DRY_RUN=false,GITHUB_REPO=${GITHUB_REPO},GITHUB_RUNNER_LABEL=${GITHUB_RUNNER_LABEL},GITHUB_TOKEN_PARAM=${GITHUB_TOKEN_PARAM},APP_URL=${APP_URL},KUMA_PUSH_URL=${KUMA_PUSH_URL}}" \
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

 NOTE: the wake gateway health-checks APP_URL from outside AWS. Now that
 APP_URL is the HTTPS domain, that check goes through Caddy on 443, not
 port 8000 directly — 8000 can stay closed to the internet, reachable only
 from Caddy on localhost.
────────────────────────────────────────────────────────────────────────
SUMMARY
