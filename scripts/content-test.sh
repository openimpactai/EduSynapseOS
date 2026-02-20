#!/usr/bin/env bash
# =============================================================================
# EduSynapse Content Creation E2E Test Helper
# =============================================================================
# Usage:
#   ./scripts/content-test.sh auth [uk|usa|rwanda|malawi] [teacher|student|parent]
#   ./scripts/content-test.sh chat "your message here"
#   ./scripts/content-test.sh chat "your message here" <session_id>
#   ./scripts/content-test.sh h5p <content_id>          # read content.json from H5P
#   ./scripts/content-test.sh h5p-ls <content_id>       # list files in H5P content dir
#   ./scripts/content-test.sh status                    # show current auth state
#
# First run:  ./scripts/content-test.sh auth
# Then:       ./scripts/content-test.sh chat "Create 3 MC questions about fractions for Year 5"
#             ./scripts/content-test.sh chat "approve" <session_id_from_above>
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CREDS_FILE="${EDUSYNAPSE_CREDS_FILE:-$PROJECT_DIR/scripts/.playground-result.json}"
AUTH_CACHE="$SCRIPT_DIR/.content-test-auth.json"
API_BASE="${EDUSYNAPSE_API_URL:-http://localhost:34000}"
H5P_CONTAINER="${H5P_CONTAINER_NAME:-eduagentic-h5p-app}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log_info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

require_jq() {
  if ! command -v jq &>/dev/null; then
    log_error "jq is required. Install: sudo apt install jq"
    exit 1
  fi
}

require_creds() {
  if [[ ! -f "$CREDS_FILE" ]]; then
    log_error "Credentials file not found: $CREDS_FILE"
    exit 1
  fi
}

require_auth() {
  if [[ ! -f "$AUTH_CACHE" ]]; then
    log_error "Not authenticated. Run: $0 auth"
    exit 1
  fi
  # Check expiry
  local exp
  exp=$(jq -r '.expires_at // 0' "$AUTH_CACHE" 2>/dev/null)
  local now
  now=$(date +%s)
  if (( now >= exp )); then
    log_warn "Token expired. Re-authenticating..."
    local country
    country=$(jq -r '.country // "uk"' "$AUTH_CACHE")
    local user_type
    user_type=$(jq -r '.user_type // "teacher"' "$AUTH_CACHE")
    do_auth "$country" "$user_type"
  fi
}

get_token() {
  jq -r '.access_token' "$AUTH_CACHE"
}

get_tenant() {
  jq -r '.tenant_code' "$AUTH_CACHE"
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

do_auth() {
  require_jq
  require_creds

  local country="${1:-uk}"
  local user_type="${2:-teacher}"

  log_info "Authenticating as $user_type from $country..."

  # Read credentials
  local api_key api_secret
  api_key=$(jq -r '.playground_tenant.edusynapse.api_key' "$CREDS_FILE")
  api_secret=$(jq -r '.playground_tenant.edusynapse.api_secret' "$CREDS_FILE")

  # Read user info based on type and country
  local user_id email first_name last_name country_code framework_code
  if [[ "$user_type" == "teacher" ]]; then
    user_id=$(jq -r ".teachers.${country}.id" "$CREDS_FILE")
    email=$(jq -r ".teachers.${country}.email" "$CREDS_FILE")
    local full_name
    full_name=$(jq -r ".teachers.${country}.name" "$CREDS_FILE")
    first_name=$(echo "$full_name" | awk '{print $1}')
    last_name=$(echo "$full_name" | awk '{print $2}')
    country_code=$(jq -r ".teachers.${country}.country_code" "$CREDS_FILE")
  elif [[ "$user_type" == "student" ]]; then
    user_id=$(jq -r ".students.${country}.id" "$CREDS_FILE")
    email=$(jq -r ".students.${country}.email" "$CREDS_FILE")
    local full_name
    full_name=$(jq -r ".students.${country}.name" "$CREDS_FILE")
    first_name=$(echo "$full_name" | awk '{print $1}')
    last_name=$(echo "$full_name" | awk '{print $2}')
    country_code=$(jq -r ".students.${country}.country_code" "$CREDS_FILE")
  elif [[ "$user_type" == "parent" ]]; then
    user_id=$(jq -r ".parents.${country}.id" "$CREDS_FILE")
    email=$(jq -r ".parents.${country}.email" "$CREDS_FILE")
    local full_name
    full_name=$(jq -r ".parents.${country}.name" "$CREDS_FILE")
    first_name=$(echo "$full_name" | awk '{print $1}')
    last_name=$(echo "$full_name" | awk '{print $2}')
    country_code=$(jq -r ".parents.${country}.country_code" "$CREDS_FILE")
  else
    log_error "Unknown user_type: $user_type (use teacher, student, or parent)"
    exit 1
  fi

  framework_code=$(jq -r ".frameworks.${country}" "$CREDS_FILE")

  if [[ "$user_id" == "null" || -z "$user_id" ]]; then
    log_error "User not found for country=$country type=$user_type"
    exit 1
  fi

  # Exchange token
  local response
  response=$(curl -s -w "\n%{http_code}" -X POST "${API_BASE}/api/v1/auth/exchange" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${api_key}" \
    -H "X-API-Secret: ${api_secret}" \
    -d "{
      \"user\": {
        \"external_id\": \"${user_id}\",
        \"email\": \"${email}\",
        \"first_name\": \"${first_name}\",
        \"last_name\": \"${last_name}\",
        \"user_type\": \"${user_type}\"
      }
    }")

  local http_code body
  http_code=$(echo "$response" | tail -1)
  body=$(echo "$response" | sed '$d')

  if [[ "$http_code" != "200" ]]; then
    log_error "Auth failed (HTTP $http_code): $body"
    exit 1
  fi

  # Save auth cache with metadata
  local access_token expires_in
  access_token=$(echo "$body" | jq -r '.access_token')
  expires_in=$(echo "$body" | jq -r '.expires_in')
  local expires_at
  expires_at=$(( $(date +%s) + expires_in ))

  jq -n \
    --arg token "$access_token" \
    --arg tenant "playground" \
    --arg country "$country" \
    --arg user_type "$user_type" \
    --arg user_id "$user_id" \
    --arg email "$email" \
    --arg name "${first_name} ${last_name}" \
    --arg country_code "$country_code" \
    --arg framework "$framework_code" \
    --argjson expires_at "$expires_at" \
    '{
      access_token: $token,
      tenant_code: $tenant,
      country: $country,
      user_type: $user_type,
      user_id: $user_id,
      email: $email,
      name: $name,
      country_code: $country_code,
      framework_code: $framework,
      expires_at: $expires_at
    }' > "$AUTH_CACHE"

  log_ok "Authenticated as ${first_name} ${last_name} (${user_type}, ${country_code})"
  log_info "Token expires at: $(date -d @${expires_at} 2>/dev/null || date -r ${expires_at} 2>/dev/null || echo $expires_at)"
}

do_chat() {
  require_jq
  require_auth

  local message="$1"
  local session_id="${2:-}"
  local language="${3:-en}"

  local token tenant country_code framework_code grade_level user_type
  token=$(get_token)
  tenant=$(get_tenant)
  country_code=$(jq -r '.country_code' "$AUTH_CACHE")
  framework_code=$(jq -r '.framework_code' "$AUTH_CACHE")
  user_type=$(jq -r '.user_type' "$AUTH_CACHE")

  # Build request body
  local body
  if [[ -n "$session_id" ]]; then
    body=$(jq -n \
      --arg msg "$message" \
      --arg sid "$session_id" \
      '{ message: $msg, session_id: $sid }')
  else
    body=$(jq -n \
      --arg msg "$message" \
      --arg lang "$language" \
      --arg role "$user_type" \
      --arg cc "$country_code" \
      --arg fc "$framework_code" \
      '{
        message: $msg,
        language: $lang,
        context: {
          user_role: $role,
          country_code: $cc,
          framework_code: $fc
        }
      }')
  fi

  log_info "Sending: ${message:0:80}..."

  local response
  response=$(curl -s -w "\n%{http_code}" --max-time 300 \
    -X POST "${API_BASE}/api/v1/content-creation/chat" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${token}" \
    -H "X-Tenant-Code: ${tenant}" \
    -d "$body")

  local http_code resp_body
  http_code=$(echo "$response" | tail -1)
  resp_body=$(echo "$response" | sed '$d')

  if [[ "$http_code" != "200" ]]; then
    log_error "Request failed (HTTP $http_code)"
    echo "$resp_body" | jq . 2>/dev/null || echo "$resp_body"
    return 1
  fi

  # Parse and display response
  echo ""
  echo -e "${GREEN}━━━ Response ━━━${NC}"

  local sid phase agent msg content_type h5p_id
  sid=$(echo "$resp_body" | jq -r '.session_id // "?"')
  phase=$(echo "$resp_body" | jq -r '.workflow_phase // "?"')
  agent=$(echo "$resp_body" | jq -r '.current_agent // "?"')
  msg=$(echo "$resp_body" | jq -r '.message // ""')
  content_type=$(echo "$resp_body" | jq -r '.generated_content.content_type // "-"')
  h5p_id=$(echo "$resp_body" | jq -r '.generated_content.h5p_id // "-"')

  echo -e "  ${CYAN}Session:${NC}      $sid"
  echo -e "  ${CYAN}Phase:${NC}        $phase"
  echo -e "  ${CYAN}Agent:${NC}        $agent"
  echo -e "  ${CYAN}Content Type:${NC} $content_type"
  echo -e "  ${CYAN}H5P ID:${NC}      $h5p_id"
  echo ""
  echo -e "${CYAN}Message:${NC}"
  echo "$msg"
  echo ""

  # Show AI content summary if present
  local has_content
  has_content=$(echo "$resp_body" | jq -r '.generated_content.ai_content // null')
  if [[ "$has_content" != "null" ]]; then
    echo -e "${YELLOW}━━━ AI Content ━━━${NC}"
    echo "$resp_body" | jq '.generated_content.ai_content | keys' 2>/dev/null

    # Show item counts for common array fields
    for field in questions exercises statements cards dialogs words panels events; do
      local count
      count=$(echo "$resp_body" | jq -r ".generated_content.ai_content.${field} | length // 0" 2>/dev/null)
      if [[ "$count" != "0" && "$count" != "null" ]]; then
        echo -e "  ${field}: ${GREEN}${count} items${NC}"
      fi
    done
    echo ""
  fi

  # Show preview URL if available
  local preview_url
  preview_url=$(echo "$resp_body" | jq -r '.generated_content.preview_url // null')
  if [[ "$preview_url" != "null" ]]; then
    echo -e "${GREEN}Preview:${NC} $preview_url"
    echo ""
  fi

  # Hint for next step
  if [[ "$phase" == "reviewing" ]]; then
    echo -e "${YELLOW}Next:${NC} $0 chat \"approve\" $sid"
    echo -e "      $0 chat \"modify: <instructions>\" $sid"
  elif [[ "$phase" == "gathering_requirements" ]]; then
    echo -e "${YELLOW}Next:${NC} $0 chat \"<your response>\" $sid"
  elif [[ "$phase" == "completed" && "$h5p_id" != "-" && "$h5p_id" != "null" ]]; then
    echo -e "${YELLOW}Verify:${NC} $0 h5p $h5p_id"
  fi

  # Save last session for convenience
  echo "$sid" > "$SCRIPT_DIR/.content-test-last-session"

  # Also save full response
  echo "$resp_body" | jq . > "$SCRIPT_DIR/.content-test-last-response.json" 2>/dev/null
}

do_h5p() {
  local content_id="$1"
  log_info "Reading H5P content.json for ID: $content_id"

  docker exec "$H5P_CONTAINER" cat "/app/h5p/content/${content_id}/content.json" 2>/dev/null | jq . 2>/dev/null
  if [[ $? -ne 0 ]]; then
    log_error "Could not read content.json for ID $content_id"
    return 1
  fi
}

do_h5p_ls() {
  local content_id="$1"
  log_info "Listing H5P content dir for ID: $content_id"
  docker exec "$H5P_CONTAINER" find "/app/h5p/content/${content_id}" -type f -exec ls -lh {} \; 2>/dev/null
}

do_status() {
  require_jq

  if [[ ! -f "$AUTH_CACHE" ]]; then
    log_warn "Not authenticated. Run: $0 auth"
    return
  fi

  echo -e "${CYAN}━━━ Auth Status ━━━${NC}"
  echo -e "  Name:       $(jq -r '.name' "$AUTH_CACHE")"
  echo -e "  Type:       $(jq -r '.user_type' "$AUTH_CACHE")"
  echo -e "  Country:    $(jq -r '.country_code' "$AUTH_CACHE")"
  echo -e "  Framework:  $(jq -r '.framework_code' "$AUTH_CACHE")"
  echo -e "  Tenant:     $(jq -r '.tenant_code' "$AUTH_CACHE")"

  local exp now
  exp=$(jq -r '.expires_at' "$AUTH_CACHE")
  now=$(date +%s)
  if (( now >= exp )); then
    echo -e "  Token:      ${RED}EXPIRED${NC}"
  else
    local remaining=$(( exp - now ))
    echo -e "  Token:      ${GREEN}Valid${NC} ($((remaining/60))m remaining)"
  fi

  if [[ -f "$SCRIPT_DIR/.content-test-last-session" ]]; then
    echo -e "  Last Session: $(cat "$SCRIPT_DIR/.content-test-last-session")"
  fi
  echo ""
  echo -e "${CYAN}API:${NC} $API_BASE"
  echo -e "${CYAN}H5P:${NC} $H5P_CONTAINER"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

cmd="${1:-help}"
shift || true

case "$cmd" in
  auth)
    do_auth "${1:-uk}" "${2:-teacher}"
    ;;
  chat)
    if [[ -z "${1:-}" ]]; then
      log_error "Usage: $0 chat \"message\" [session_id] [language]"
      exit 1
    fi
    do_chat "$1" "${2:-}" "${3:-en}"
    ;;
  h5p)
    if [[ -z "${1:-}" ]]; then
      log_error "Usage: $0 h5p <content_id>"
      exit 1
    fi
    do_h5p "$1"
    ;;
  h5p-ls)
    if [[ -z "${1:-}" ]]; then
      log_error "Usage: $0 h5p-ls <content_id>"
      exit 1
    fi
    do_h5p_ls "$1"
    ;;
  status)
    do_status
    ;;
  help|*)
    echo "EduSynapse Content Creation Test Helper"
    echo ""
    echo "Usage:"
    echo "  $0 auth [country] [user_type]     Authenticate (default: uk teacher)"
    echo "  $0 chat \"message\" [session_id]    Send chat message"
    echo "  $0 h5p <content_id>               Read H5P content.json"
    echo "  $0 h5p-ls <content_id>            List H5P content files"
    echo "  $0 status                         Show current auth state"
    echo ""
    echo "Countries: uk, usa, rwanda, malawi"
    echo "User types: teacher, student, parent"
    echo ""
    echo "Example flow:"
    echo "  $0 auth uk teacher"
    echo "  $0 chat \"Create 3 MC questions about fractions for Year 5\""
    echo "  $0 chat \"approve\" <session_id>"
    echo "  $0 h5p <h5p_id>"
    ;;
esac
