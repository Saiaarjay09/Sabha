#!/bin/bash
# Install Sabha as a permanent service on this Mac.
#
#   1. a launchd agent that starts the server at login and restarts it if it dies
#   2. a Tailscale Funnel path that publishes it on the existing hostname
#
# Safe to re-run: launchctl bootout is tolerated failing, and the funnel
# command is idempotent.
set -u

LABEL=com.hiringcouncil.server
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
SRC="$(cd "$(dirname "$0")" && pwd)/$LABEL.plist"
PORT=8700
PATH_PREFIX=/council

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$(command -v python3)"
[ -z "$PYTHON" ] && { echo "python3 not found on PATH"; exit 1; }

mkdir -p "$HOME/Library/Logs/Sabha"

# The committed plist is a template — substitute this machine's paths rather
# than shipping one person's home directory in a public repository.
sed -e "s|__PYTHON__|$PYTHON|g" \
    -e "s|__REPO_DIR__|$REPO_DIR|g" \
    -e "s|__HOME__|$HOME|g" \
    "$SRC" > "$PLIST"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null
launchctl bootstrap "gui/$(id -u)" "$PLIST" || { echo "launchctl bootstrap failed"; exit 1; }
launchctl kickstart -k "gui/$(id -u)/$LABEL"

echo "waiting for the server to come up…"
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    echo "server is up on 127.0.0.1:$PORT"
    break
  fi
  sleep 1
done

# Ollama must also survive a reboot, or the council has no models to run on.
if ! launchctl list 2>/dev/null | grep -q ollama; then
  echo "NOTE: Ollama doesn't look like a login item. Open the Ollama app once and"
  echo "      enable 'Launch at login', or the site will return an error after a reboot."
fi

tailscale funnel --bg --set-path "$PATH_PREFIX" "http://127.0.0.1:$PORT" \
  && echo "published at https://$(tailscale status --json | /usr/bin/python3 -c 'import json,sys;print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')$PATH_PREFIX/"
