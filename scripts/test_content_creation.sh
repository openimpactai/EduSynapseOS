#!/bin/bash
#
# H5P Content Creation System - Integration Test Script
# Tests the complete content creation workflow against the running API
#

set -e

BASE_URL="http://localhost:34000/api/v1"
CREDS_FILE="${EDUSYNAPSE_CREDS_FILE:-./scripts/.playground-result.json}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "======================================================================"
echo "     H5P Content Creation System - Integration Test Suite"
echo "======================================================================"
echo -e "${NC}"

# Load credentials
API_KEY=$(jq -r '.playground_tenant.edusynapse.api_key' "$CREDS_FILE")
API_SECRET=$(jq -r '.playground_tenant.edusynapse.api_secret' "$CREDS_FILE")
UK_TEACHER_ID=$(jq -r '.teachers.uk.id' "$CREDS_FILE")
UK_TEACHER_EMAIL=$(jq -r '.teachers.uk.email' "$CREDS_FILE")
USA_TEACHER_ID=$(jq -r '.teachers.usa.id' "$CREDS_FILE")
UK_STUDENT_ID=$(jq -r '.students.uk.id' "$CREDS_FILE")

echo -e "${YELLOW}Credentials Loaded:${NC}"
echo "  API Key: ${API_KEY:0:20}..."
echo "  UK Teacher: $UK_TEACHER_EMAIL"
echo ""

# Get auth token
echo -e "${YELLOW}[1/8] Getting Auth Token...${NC}"
AUTH_RESPONSE=$(curl -s -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -H "X-API-Secret: $API_SECRET" \
  -d "{\"email\": \"$UK_TEACHER_EMAIL\", \"password\": \"playground123\"}")

ACCESS_TOKEN=$(echo "$AUTH_RESPONSE" | jq -r '.access_token // empty')
if [ -z "$ACCESS_TOKEN" ]; then
  echo -e "${RED}[FAIL]${NC} Could not get auth token"
  echo "$AUTH_RESPONSE"
  exit 1
fi
echo -e "${GREEN}[OK]${NC} Auth token obtained"

# Test 1: Health check
echo -e "\n${YELLOW}[2/8] API Health Check...${NC}"
HEALTH=$(curl -s "$BASE_URL/../health")
if echo "$HEALTH" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
  echo -e "${GREEN}[OK]${NC} API is healthy"
else
  echo -e "${RED}[FAIL]${NC} API health check failed"
  echo "$HEALTH"
fi

# Test 2: List content types
echo -e "\n${YELLOW}[3/8] Listing Content Types...${NC}"
CONTENT_TYPES=$(curl -s "$BASE_URL/content-creation/content-types" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "X-API-Key: $API_KEY" \
  -H "X-API-Secret: $API_SECRET")

TYPE_COUNT=$(echo "$CONTENT_TYPES" | jq -r '.total // 0')
if [ "$TYPE_COUNT" -gt 0 ]; then
  echo -e "${GREEN}[OK]${NC} Found $TYPE_COUNT content types"
  echo "  Categories: $(echo "$CONTENT_TYPES" | jq -r '.categories | join(", ")')"
else
  echo -e "${RED}[FAIL]${NC} No content types found"
  echo "$CONTENT_TYPES"
fi

# Test 3: Start content creation chat (UK Teacher - English)
echo -e "\n${YELLOW}[4/8] Starting Content Creation Chat (UK Teacher)...${NC}"
CHAT_RESPONSE=$(curl -s -X POST "$BASE_URL/content-creation/chat" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -H "X-API-Secret: $API_SECRET" \
  -d '{
    "message": "I want to create a multiple choice quiz about photosynthesis for Year 5 students",
    "language": "en",
    "context": {
      "user_role": "teacher",
      "country_code": "GB",
      "framework_code": "UK-NC-2014",
      "grade_level": 5,
      "subject_code": "SCI"
    }
  }')

SESSION_ID=$(echo "$CHAT_RESPONSE" | jq -r '.session_id // empty')
WORKFLOW_PHASE=$(echo "$CHAT_RESPONSE" | jq -r '.workflow_phase // empty')

if [ -n "$SESSION_ID" ]; then
  echo -e "${GREEN}[OK]${NC} Chat session created"
  echo "  Session ID: ${SESSION_ID:0:36}"
  echo "  Phase: $WORKFLOW_PHASE"
  echo "  Message: $(echo "$CHAT_RESPONSE" | jq -r '.message[:80]')..."
else
  echo -e "${RED}[FAIL]${NC} Failed to create chat session"
  echo "$CHAT_RESPONSE" | jq .
fi

# Test 4: Get session info
if [ -n "$SESSION_ID" ]; then
  echo -e "\n${YELLOW}[5/8] Getting Session Info...${NC}"
  SESSION_INFO=$(curl -s "$BASE_URL/content-creation/sessions/$SESSION_ID" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "X-API-Key: $API_KEY" \
    -H "X-API-Secret: $API_SECRET")

  SESSION_STATUS=$(echo "$SESSION_INFO" | jq -r '.status // empty')
  if [ -n "$SESSION_STATUS" ]; then
    echo -e "${GREEN}[OK]${NC} Session info retrieved"
    echo "  Status: $SESSION_STATUS"
    echo "  Phase: $(echo "$SESSION_INFO" | jq -r '.workflow_phase')"
    echo "  Messages: $(echo "$SESSION_INFO" | jq -r '.message_count')"
  else
    echo -e "${RED}[FAIL]${NC} Failed to get session info"
    echo "$SESSION_INFO" | jq .
  fi
fi

# Test 5: Continue conversation
if [ -n "$SESSION_ID" ]; then
  echo -e "\n${YELLOW}[6/8] Continuing Conversation...${NC}"
  CONTINUE_RESPONSE=$(curl -s -X POST "$BASE_URL/content-creation/chat" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -H "X-API-Secret: $API_SECRET" \
    -d "{
      \"session_id\": \"$SESSION_ID\",
      \"message\": \"Yes, please create 5 questions about the process of photosynthesis\"
    }")

  NEW_PHASE=$(echo "$CONTINUE_RESPONSE" | jq -r '.workflow_phase // empty')
  if [ -n "$NEW_PHASE" ]; then
    echo -e "${GREEN}[OK]${NC} Conversation continued"
    echo "  Phase: $NEW_PHASE"
    echo "  Message: $(echo "$CONTINUE_RESPONSE" | jq -r '.message[:80]')..."

    # Check if content was generated
    HAS_CONTENT=$(echo "$CONTINUE_RESPONSE" | jq -r '.generated_content != null')
    if [ "$HAS_CONTENT" = "true" ]; then
      echo -e "  ${GREEN}Content Generated!${NC}"
      echo "    Type: $(echo "$CONTINUE_RESPONSE" | jq -r '.generated_content.content_type')"
      echo "    Title: $(echo "$CONTINUE_RESPONSE" | jq -r '.generated_content.title')"
    fi
  else
    echo -e "${RED}[FAIL]${NC} Failed to continue conversation"
    echo "$CONTINUE_RESPONSE" | jq .
  fi
fi

# Test 6: Get message history
if [ -n "$SESSION_ID" ]; then
  echo -e "\n${YELLOW}[7/8] Getting Message History...${NC}"
  MESSAGES=$(curl -s "$BASE_URL/content-creation/sessions/$SESSION_ID/messages?limit=10" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "X-API-Key: $API_KEY" \
    -H "X-API-Secret: $API_SECRET")

  MSG_COUNT=$(echo "$MESSAGES" | jq -r '.messages | length')
  if [ "$MSG_COUNT" -gt 0 ]; then
    echo -e "${GREEN}[OK]${NC} Retrieved $MSG_COUNT messages"
  else
    echo -e "${YELLOW}[WARN]${NC} No messages found"
  fi
fi

# Test 7: End session
if [ -n "$SESSION_ID" ]; then
  echo -e "\n${YELLOW}[8/8] Ending Session...${NC}"
  END_RESPONSE=$(curl -s -X POST "$BASE_URL/content-creation/sessions/$SESSION_ID/end" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "X-API-Key: $API_KEY" \
    -H "X-API-Secret: $API_SECRET")

  SUCCESS=$(echo "$END_RESPONSE" | jq -r '.success // false')
  if [ "$SUCCESS" = "true" ]; then
    echo -e "${GREEN}[OK]${NC} Session ended successfully"
    echo "  Generated: $(echo "$END_RESPONSE" | jq -r '.generated_count') items"
    echo "  Exported: $(echo "$END_RESPONSE" | jq -r '.exported_count') items"
  else
    echo -e "${YELLOW}[WARN]${NC} Session end response:"
    echo "$END_RESPONSE" | jq .
  fi
fi

echo -e "\n${BLUE}"
echo "======================================================================"
echo "                    Test Summary"
echo "======================================================================"
echo -e "${NC}"
echo -e "Content Types Available: ${GREEN}$TYPE_COUNT${NC}"
echo -e "Session Created: ${GREEN}$SESSION_ID${NC}"
echo ""
echo -e "${GREEN}All tests completed!${NC}"
