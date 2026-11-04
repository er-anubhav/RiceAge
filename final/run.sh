#!/usr/bin/env bash
# ==============================================================================
# RiceAge Automated Grain Analyzer Launcher
# Starts the Modern React Laboratory Dashboard in Fullscreen Touchscreen Kiosk Mode
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "    RiceAge Modern Laboratory Dashboard (M5.0)         "
echo "========================================================"

# Display environment for Wayland / labwc
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

# Ensure proper screen orientation on Addi-Data HDMI Touchscreen
if command -v wlr-randr >/dev/null 2>&1; then
    wlr-randr --output HDMI-A-1 --transform normal 2>/dev/null || true
fi

# Look for Python
if [ -f "$SCRIPT_DIR/../.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/../.venv/bin/python"
elif [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "/usr/bin/python3" ]; then
    PYTHON="/usr/bin/python3"
else
    PYTHON="python3"
fi

echo "Using Python: $($PYTHON --version) ($PYTHON)"

# Clean up any lingering background backend or kiosk instances safely
pgrep -f 'python.*app.py' | grep -v $$ | xargs kill 2>/dev/null || true
pgrep -f 'chromium' | grep -v $$ | xargs kill 2>/dev/null || true
rm -rf /tmp/chromium-kiosk/Singleton* /tmp/chromium-kiosk/Default/Preferences 2>/dev/null || true


# Start Flask Web Backend in background
echo "Starting RiceAge backend (Flask + Real ML Inference)..."
"$PYTHON" "$SCRIPT_DIR/app.py" --headless > "$SCRIPT_DIR/app.log" 2>&1 &
BACKEND_PID=$!

# Ensure backend process is cleaned up when kiosk exits
cleanup() {
    echo "Shutting down RiceAge..."
    kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for backend to be ready
echo "Waiting for backend on http://127.0.0.1:5000/api/health..."
MAX_TRIES=30
TRIES=0
until "$PYTHON" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/api/health', timeout=1)" 2>/dev/null; do
    sleep 0.5
    TRIES=$((TRIES + 1))
    if [ "$TRIES" -ge "$MAX_TRIES" ]; then
        echo "Backend failed to start. Check app.log:"
        tail -n 25 "$SCRIPT_DIR/app.log"
        exit 1
    fi
done
echo "Backend is online and serving Modern React Dashboard!"

# Locate Chromium binary
CHROMIUM_BIN="/usr/bin/chromium"
if [ ! -x "$CHROMIUM_BIN" ]; then
    CHROMIUM_BIN="$(which chromium || which chromium-browser || echo 'chromium')"
fi

echo "Launching Modern React Dashboard in fullscreen kiosk mode..."
mkdir -p /tmp/chromium-kiosk

exec "$CHROMIUM_BIN" \
    --kiosk \
    --ozone-platform=wayland \
    --password-store=basic \
    --no-default-browser-check \
    --enable-features=OverlayScrollbar \
    --noerrdialogs \
    --disable-infobars \
    --no-first-run \
    --disable-session-crashed-bubble \
    --disable-features=Translate,OptimizationHints,MediaRouter \
    --fast \
    --fast-start \
    --disable-pinch \
    --overscroll-history-navigation=0 \
    --check-for-update-interval=31536000 \
    --user-data-dir=/tmp/chromium-kiosk \
    http://127.0.0.1:5000

