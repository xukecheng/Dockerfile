#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# OpenClaw Sandbox Browser - Entrypoint
# ---------------------------------------------------------------------------

export DISPLAY=:1
export HOME=/data/chrome-profile
export XDG_CONFIG_HOME="${HOME}/.config"
export XDG_CACHE_HOME="${HOME}/.cache"

# Environment variables with defaults
CDP_PORT="${CDP_PORT:-9222}"
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
ENABLE_NOVNC="${ENABLE_NOVNC:-1}"
HEADLESS="${HEADLESS:-0}"
CHROME_EXTRA_ARGS="${CHROME_EXTRA_ARGS:-}"
SCREEN_RESOLUTION="${SCREEN_RESOLUTION:-1920x1080x24}"

# Internal CDP port (Caddy proxies CDP_PORT -> CHROME_CDP_PORT)
if [[ "${CDP_PORT}" -ge 65535 ]]; then
  CHROME_CDP_PORT="$((CDP_PORT - 1))"
else
  CHROME_CDP_PORT="$((CDP_PORT + 1))"
fi

mkdir -p "${HOME}" "${HOME}/.chrome" "${XDG_CONFIG_HOME}" "${XDG_CACHE_HOME}"

# ---------------------------------------------------------------------------
# PID tracking for cleanup
# ---------------------------------------------------------------------------
XVFB_PID=""
CADDY_PID=""
VNC_PID=""
WEBSOCKIFY_PID=""
CHROME_PID=""

cleanup() {
  echo "[entrypoint] Shutting down..."
  local pids=("$CHROME_PID" "$CADDY_PID" "$WEBSOCKIFY_PID" "$VNC_PID" "$XVFB_PID")
  for pid in "${pids[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  # Give processes a moment, then force kill
  sleep 1
  for pid in "${pids[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  echo "[entrypoint] Cleanup complete."
  exit 0
}

trap cleanup SIGTERM SIGINT SIGHUP

# ---------------------------------------------------------------------------
# Start Xvfb
# ---------------------------------------------------------------------------
echo "[entrypoint] Starting Xvfb (${SCREEN_RESOLUTION})..."
Xvfb :1 -screen 0 "${SCREEN_RESOLUTION}" -ac -nolisten tcp &
XVFB_PID=$!
sleep 0.5

# ---------------------------------------------------------------------------
# Build Chromium arguments
# ---------------------------------------------------------------------------
build_chrome_args() {
  local args=()

  if [[ "${HEADLESS}" == "1" ]]; then
    args+=("--headless=new" "--disable-gpu")
  fi

  args+=(
    "--remote-debugging-address=127.0.0.1"
    "--remote-debugging-port=${CHROME_CDP_PORT}"
    "--user-data-dir=${HOME}/.chrome"
    "--no-first-run"
    "--no-default-browser-check"
    "--disable-dev-shm-usage"
    "--disable-background-networking"
    "--disable-features=TranslateUI"
    "--disable-breakpad"
    "--disable-crash-reporter"
    "--metrics-recording-only"
    "--no-sandbox"
    # Anti-detection
    "--disable-blink-features=AutomationControlled"
    "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
  )

  # GPU acceleration (if /dev/dri/renderD* exists)
  if ls /dev/dri/renderD* 1>/dev/null 2>&1; then
    echo "[entrypoint] GPU device detected, enabling hardware acceleration."
    args+=(
      "--enable-gpu-rasterization"
      "--enable-zero-copy"
      "--ignore-gpu-blocklist"
      "--enable-features=VaapiVideoDecoder,VaapiVideoEncoder"
      "--disable-software-rasterizer"
    )
    # Remove --disable-gpu if it was added for headless
    local filtered=()
    for arg in "${args[@]}"; do
      [[ "$arg" != "--disable-gpu" ]] && filtered+=("$arg")
    done
    args=("${filtered[@]}")
  fi

  # User extra args
  if [[ -n "${CHROME_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    args+=(${CHROME_EXTRA_ARGS})
  fi

  echo "${args[@]}"
}

# ---------------------------------------------------------------------------
# Start Chromium with crash recovery
# ---------------------------------------------------------------------------
start_chromium() {
  # Clean up stale lock files
  rm -f "${HOME}/.chrome/SingletonLock" \
        "${HOME}/.chrome/SingletonCookie" \
        "${HOME}/.chrome/SingletonSocket" 2>/dev/null || true

  local chrome_args
  chrome_args=$(build_chrome_args)

  echo "[entrypoint] Starting Chromium (CDP on 127.0.0.1:${CHROME_CDP_PORT})..."
  # shellcheck disable=SC2086
  chromium ${chrome_args} about:blank &
  CHROME_PID=$!

  # Wait for CDP to be ready
  local ready=0
  for _ in $(seq 1 50); do
    if curl -sS --max-time 1 "http://127.0.0.1:${CHROME_CDP_PORT}/json/version" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 0.2
  done

  if [[ "$ready" -eq 1 ]]; then
    echo "[entrypoint] Chromium CDP ready."
  else
    echo "[entrypoint] WARNING: Chromium CDP did not become ready in time."
  fi
}

start_chromium

# ---------------------------------------------------------------------------
# Start Caddy reverse proxy (bypass Chrome Host header check)
# ---------------------------------------------------------------------------
echo "[entrypoint] Starting Caddy reverse proxy (:${CDP_PORT} -> 127.0.0.1:${CHROME_CDP_PORT})..."
cat > /tmp/Caddyfile << EOF
{
  auto_https off
  admin off
}
:${CDP_PORT} {
  reverse_proxy 127.0.0.1:${CHROME_CDP_PORT} {
    header_up Host 127.0.0.1
  }
}
EOF

caddy run --config /tmp/Caddyfile &
CADDY_PID=$!

# ---------------------------------------------------------------------------
# Start noVNC (optional)
# ---------------------------------------------------------------------------
if [[ "${ENABLE_NOVNC}" == "1" && "${HEADLESS}" != "1" ]]; then
  echo "[entrypoint] Starting VNC on :${VNC_PORT}, noVNC on :${NOVNC_PORT}..."
  x11vnc -display :1 -rfbport "${VNC_PORT}" -shared -forever -nopw -localhost &
  VNC_PID=$!
  websockify --web /usr/share/novnc/ "${NOVNC_PORT}" "localhost:${VNC_PORT}" &
  WEBSOCKIFY_PID=$!
fi

echo "[entrypoint] All services started. CDP available at :${CDP_PORT}"

# ---------------------------------------------------------------------------
# Monitor loop: restart Chromium on crash, exit if Caddy/Xvfb dies
# ---------------------------------------------------------------------------
while true; do
  # Check critical services
  if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    echo "[entrypoint] Xvfb exited unexpectedly. Shutting down."
    cleanup
  fi
  if ! kill -0 "$CADDY_PID" 2>/dev/null; then
    echo "[entrypoint] Caddy exited unexpectedly. Shutting down."
    cleanup
  fi

  # Auto-restart Chromium on crash
  if ! kill -0 "$CHROME_PID" 2>/dev/null; then
    echo "[entrypoint] Chromium crashed (PID ${CHROME_PID}). Restarting in 2s..."
    sleep 2
    start_chromium
  fi

  sleep 3
done
