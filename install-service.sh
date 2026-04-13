#!/bin/bash

# Install AtlasAI as a systemd service on Jetson (JetPack / Ubuntu)
# Run as: sudo bash install-service.sh

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="atlasai"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
RUN_USER="${SUDO_USER:-$(whoami)}"

# Must be run as root (via sudo)
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo bash install-service.sh"
    exit 1
fi

echo "Installing AtlasAI service..."
echo "  Repo:  $REPO_DIR"
echo "  User:  $RUN_USER"

# Ensure the user is in the docker and audio groups
usermod -aG docker "$RUN_USER" 2>/dev/null || true
usermod -aG audio  "$RUN_USER" 2>/dev/null || true

# Make start.sh executable
chmod +x "$REPO_DIR/start.sh"

# Write the service file, substituting the real user and directory
sed \
    -e "s|ATLASAI_USER|$RUN_USER|g" \
    -e "s|ATLASAI_DIR|$REPO_DIR|g" \
    "$REPO_DIR/atlasai.service" > "$SERVICE_FILE"

echo "  Service file written to $SERVICE_FILE"

# Reload systemd and enable the service
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

echo ""
echo "AtlasAI service installed and enabled."
echo ""
echo "Commands:"
echo "  Start now:   sudo systemctl start atlasai"
echo "  Stop:        sudo systemctl stop atlasai"
echo "  Status:      sudo systemctl status atlasai"
echo "  Logs:        journalctl -u atlasai -f"
echo "  Disable:     sudo systemctl disable atlasai"
echo ""
echo "The service will start automatically on next boot."
echo "To start it now without rebooting: sudo systemctl start atlasai"
