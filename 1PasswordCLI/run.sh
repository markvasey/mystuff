#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1

LIVE_VAULT="/home/markvasey/Dropbox/Apps/1Password/1Password.opvault"
LOCAL_VAULT="1Password.opvault"

if [ $# -gt 0 ]; then
    VAULT_PATH="$1"
else
    echo "Select vault to open:"
    echo "  1) Live vault ($LIVE_VAULT) [default]"
    echo "  2) Local copy ($LOCAL_VAULT)"
    read -r -p "Enter choice [1/2]: " choice

    case "$choice" in
        2|local|Local|LOCAL)
            VAULT_PATH="$LOCAL_VAULT"
            ;;
        *)
            VAULT_PATH="$LIVE_VAULT"
            ;;
    esac
fi

echo "Using vault: $VAULT_PATH"
~/.sdkman/candidates/java/current/bin/java -jar target/1PasswordCLI.jar "$VAULT_PATH"
