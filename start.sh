#!/usr/bin/env bash
# =============================================================================
#  VoiceAPI — Local Launcher
#  Starts the FastAPI backend (port 8000) and Next.js web UI (port 3000)
#  in a single terminal session.
#
#  Usage:
#    chmod +x start.sh
#    ./start.sh
#
#  Optional env overrides:
#    API_PORT=8001 WEB_PORT=3001 ./start.sh
# =============================================================================

set -euo pipefail

# ── Configurable ports ────────────────────────────────────────────────────────
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

banner() {
  echo ""
  echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════╗${NC}"
  echo -e "${CYAN}${BOLD}║        VoiceAPI — Local Development          ║${NC}"
  echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════╝${NC}"
  echo ""
}

# ── Resolve repo root (directory this script lives in) ───────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="${REPO_ROOT}/web"

# ── Cleanup on exit ───────────────────────────────────────────────────────────
API_PID=""
WEB_PID=""

cleanup() {
  echo ""
  echo -e "${YELLOW}Shutting down services…${NC}"
  [[ -n "$API_PID" ]] && kill "$API_PID" 2>/dev/null && echo "  stopped API server (pid $API_PID)"
  [[ -n "$WEB_PID" ]] && kill "$WEB_PID" 2>/dev/null && echo "  stopped web UI   (pid $WEB_PID)"
  echo -e "${GREEN}Done.${NC}"
  exit 0
}
trap cleanup SIGINT SIGTERM

# ── Pre-flight checks ─────────────────────────────────────────────────────────
check_python() {
  if ! command -v python3 &>/dev/null; then
    echo -e "${RED}ERROR: python3 not found. Install Python 3.9+ and try again.${NC}" >&2
    exit 1
  fi
  PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  echo -e "  ${GREEN}✓${NC} Python ${PY_VER}"
}

check_node() {
  if ! command -v node &>/dev/null; then
    echo -e "${RED}ERROR: Node.js not found. Install Node 18+ and try again.${NC}" >&2
    exit 1
  fi
  NODE_VER=$(node --version)
  echo -e "  ${GREEN}✓${NC} Node ${NODE_VER}"
}

check_pip_deps() {
  echo -e "  Checking Python dependencies…"
  if ! python3 -c "import fastapi, uvicorn, torch, soundfile" 2>/dev/null; then
    echo -e "  ${YELLOW}Installing Python dependencies (this may take a while)…${NC}"
    pip install -r "${REPO_ROOT}/requirements.txt"
  else
    echo -e "  ${GREEN}✓${NC} Python packages present"
  fi
}

check_npm_deps() {
  if [[ ! -d "${WEB_DIR}/node_modules" ]]; then
    echo -e "  ${YELLOW}Installing Node.js dependencies…${NC}"
    npm install --prefix "${WEB_DIR}"
  else
    echo -e "  ${GREEN}✓${NC} node_modules present"
  fi
}

ensure_env_local() {
  local env_file="${WEB_DIR}/.env.local"
  if [[ ! -f "$env_file" ]]; then
    echo "NEXT_PUBLIC_API_BASE=http://localhost:${API_PORT}" > "$env_file"
    echo -e "  ${GREEN}✓${NC} Created ${env_file}"
  else
    # Update the port if it was overridden
    if [[ "$API_PORT" != "8000" ]]; then
      sed -i.bak "s|NEXT_PUBLIC_API_BASE=.*|NEXT_PUBLIC_API_BASE=http://localhost:${API_PORT}|" "$env_file"
      rm -f "${env_file}.bak"
    fi
    echo -e "  ${GREEN}✓${NC} ${env_file} exists"
  fi
}

check_models() {
  local models_dir="${REPO_ROOT}/models"
  if [[ ! -d "$models_dir" ]]; then
    echo -e "  ${YELLOW}⚠  models/ directory not found.${NC}"
    echo -e "  Run:  python3 -m src.cli download --all"
    echo ""
  else
    local count
    count=$(find "$models_dir" -maxdepth 2 \( -name "*.pt" -o -name "*.pth" \) | wc -l | tr -d ' ')
    echo -e "  ${GREEN}✓${NC} models/ present — ${count} model file(s) found"
  fi
}

# ── Start API server ──────────────────────────────────────────────────────────
start_api() {
  echo ""
  echo -e "${BOLD}Starting API server on http://localhost:${API_PORT} …${NC}"
  cd "${REPO_ROOT}"
  python3 -m uvicorn src.api:app \
    --host 0.0.0.0 \
    --port "${API_PORT}" \
    --log-level info \
    2>&1 | sed "s/^/  ${CYAN}[API]${NC} /" &
  API_PID=$!
  echo -e "  ${GREEN}✓${NC} API server started (pid ${API_PID})"
}

# ── Start web UI ──────────────────────────────────────────────────────────────
start_web() {
  echo ""
  echo -e "${BOLD}Starting web UI on http://localhost:${WEB_PORT} …${NC}"
  cd "${WEB_DIR}"
  PORT="${WEB_PORT}" npm run dev \
    2>&1 | sed "s/^/  ${GREEN}[WEB]${NC} /" &
  WEB_PID=$!
  echo -e "  ${GREEN}✓${NC} Web UI started (pid ${WEB_PID})"
}

# ── Wait for API to be ready ──────────────────────────────────────────────────
wait_for_api() {
  echo ""
  echo -n "  Waiting for API to be ready"
  local attempts=0
  local max_attempts=30
  while ! curl -sf "http://localhost:${API_PORT}/health" >/dev/null 2>&1; do
    sleep 1
    echo -n "."
    attempts=$((attempts + 1))
    if [[ $attempts -ge $max_attempts ]]; then
      echo ""
      echo -e "  ${YELLOW}⚠  API did not respond within ${max_attempts}s — it may still be loading models.${NC}"
      return
    fi
  done
  echo ""
  echo -e "  ${GREEN}✓${NC} API is ready"
}

# ── Main ──────────────────────────────────────────────────────────────────────
banner

echo -e "${BOLD}Pre-flight checks…${NC}"
check_python
check_node
check_pip_deps
check_npm_deps
ensure_env_local
check_models

start_api

# Give uvicorn a moment to bind the port before checking
sleep 2
wait_for_api

start_web

echo ""
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  Web UI  →  http://localhost:${WEB_PORT}${NC}"
echo -e "${BOLD}  API     →  http://localhost:${API_PORT}${NC}"
echo -e "${BOLD}  API docs →  http://localhost:${API_PORT}/docs${NC}"
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  Press ${BOLD}Ctrl+C${NC} to stop both services."
echo ""

# Keep script alive; forward signals via cleanup trap
wait
