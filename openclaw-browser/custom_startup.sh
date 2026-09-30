#!/usr/bin/env bash
set -ex

# ---------------------------------------------------------------------------
# Add custom VNC user (kasmweb only creates kasm_user/kasm_viewer by default)
# ---------------------------------------------------------------------------
if [ -n "$VNC_USER" ] && [ "$VNC_USER" != "kasm_user" ] && [ -n "$VNC_PW" ]; then
    echo -e "${VNC_PW}\n${VNC_PW}\n" | kasmvncpasswd -u "$VNC_USER" -wo 2>/dev/null || true
    echo "[openclaw] VNC user '${VNC_USER}' created"
fi

# ---------------------------------------------------------------------------
# Chrome profile migration (Chrome 136+ refuses CDP on the default profile dir)
# One-time copy keeps logins: both dirs use --password-store=basic, same key
# ---------------------------------------------------------------------------
CHROME_USER_DATA_DIR="${CHROME_USER_DATA_DIR:-$HOME/chrome-profile}"
OLD_PROFILE_DIR="$HOME/.config/google-chrome"
if [ ! -d "$CHROME_USER_DATA_DIR" ] && [ -d "$OLD_PROFILE_DIR" ]; then
    cp -a "$OLD_PROFILE_DIR" "$CHROME_USER_DATA_DIR.tmp"
    rm -f "$CHROME_USER_DATA_DIR.tmp"/Singleton*
    mv "$CHROME_USER_DATA_DIR.tmp" "$CHROME_USER_DATA_DIR"
    echo "[openclaw] Migrated Chrome profile $OLD_PROFILE_DIR -> $CHROME_USER_DATA_DIR"
fi

# kasm's /usr/bin/google-chrome only does this for ~/.config/google-chrome.
# The lock records the old container's hostname, so after a recreate Chrome
# blocks on an "another computer is using this profile" dialog without it.
rm -f "$CHROME_USER_DATA_DIR"/Singleton*
if [ -f "$CHROME_USER_DATA_DIR/Default/Preferences" ]; then
    sed -i -e 's/"exited_cleanly":false/"exited_cleanly":true/' \
           -e 's/"exit_type":"Crashed"/"exit_type":"None"/' \
        "$CHROME_USER_DATA_DIR/Default/Preferences"
fi

# ---------------------------------------------------------------------------
# Caddy reverse proxy for CDP (bypass Chrome Host header check)
# ---------------------------------------------------------------------------
CDP_PORT="${CDP_PORT:-9222}"
CHROME_CDP_PORT="9223"

MCP_PROXY_PORT="8765"

# MCP_TOKEN set -> /mcp /sse /messages require "Authorization: Bearer <token>"
# or "X-API-Key: <token>" (claude.ai custom connectors reserve Authorization
# for OAuth, so their static header has to be X-API-Key).
# /ping stays open for health checks. CDP itself has no auth mechanism.
set +x
MCP_AUTH=""
if [ -n "$MCP_TOKEN" ]; then
    MCP_AUTH="@unauthorized {
      not path /ping
      not header Authorization \"Bearer ${MCP_TOKEN}\"
      not header X-API-Key \"${MCP_TOKEN}\"
    }
    respond @unauthorized 401"
fi

cat > /tmp/Caddyfile << EOF
{
  auto_https off
  admin off
}
:${CDP_PORT} {
  @mcp path /mcp /sse /messages /ping
  handle @mcp {
    ${MCP_AUTH}
    reverse_proxy 127.0.0.1:${MCP_PROXY_PORT}
  }
  handle {
    reverse_proxy 127.0.0.1:${CHROME_CDP_PORT}
  }
}
EOF
chmod 600 /tmp/Caddyfile
echo "[openclaw] MCP auth: $([ -n "$MCP_TOKEN" ] && echo bearer || echo NONE)"
set -x

caddy run --config /tmp/Caddyfile &
echo "[openclaw] Caddy started (:${CDP_PORT} -> CDP :${CHROME_CDP_PORT} + MCP :${MCP_PROXY_PORT})"

# ---------------------------------------------------------------------------
# MCP Server (chrome-devtools-mcp via mcp-proxy, streamable HTTP)
# chrome-devtools-mcp uses lazy connection: connects to Chrome on first tool
# call, auto-reconnects if Chrome restarts (--browserUrl mode)
# ---------------------------------------------------------------------------
# MCP_EXTRA_ARGS: extra chrome-devtools-mcp flags, e.g. --screenshotFormat=jpeg
# (read -a splits on whitespace without glob-expanding URL patterns like *://10.*)
read -ra MCP_EXTRA <<< "${MCP_EXTRA_ARGS:-}"
mcp-proxy --port "${MCP_PROXY_PORT}" -- \
  chrome-devtools-mcp \
    --browserUrl "http://127.0.0.1:${CHROME_CDP_PORT}" \
    --no-usage-statistics \
    --no-performance-crux \
    "${MCP_EXTRA[@]}" &
echo "[openclaw] MCP server started (chrome-devtools-mcp via mcp-proxy :${MCP_PROXY_PORT})"

# ---------------------------------------------------------------------------
# Original kasmweb Chrome startup logic
# ---------------------------------------------------------------------------
START_COMMAND="google-chrome"
PGREP="chrome"
MAXIMIZE="true"
DEFAULT_ARGS=""

if [[ $MAXIMIZE == 'true' ]] ; then
    DEFAULT_ARGS+=" --start-maximized"
fi
ARGS=${APP_ARGS:-$DEFAULT_ARGS}

options=$(getopt -o gau: -l go,assign,url: -n "$0" -- "$@") || exit
eval set -- "$options"

while [[ $1 != -- ]]; do
    case $1 in
        -g|--go) GO='true'; shift 1;;
        -a|--assign) ASSIGN='true'; shift 1;;
        -u|--url) OPT_URL=$2; shift 2;;
        *) echo "bad option: $1" >&2; exit 1;;
    esac
done
shift

for arg; do
    echo "arg! $arg"
done

FORCE=$2

kasm_exec() {
    if [ -n "$OPT_URL" ] ; then
        URL=$OPT_URL
    elif [ -n "$1" ] ; then
        URL=$1
    fi

    if [ -n "$URL" ] ; then
        /usr/bin/filter_ready
        /usr/bin/desktop_ready
        $START_COMMAND $ARGS $OPT_URL
    else
        echo "No URL specified for exec command. Doing nothing."
    fi
}

kasm_startup() {
    if [ -n "$KASM_URL" ] ; then
        URL=$KASM_URL
    elif [ -z "$URL" ] ; then
        URL=$LAUNCH_URL
    fi

    if [ -z "$DISABLE_CUSTOM_STARTUP" ] ||  [ -n "$FORCE" ] ; then

        echo "Entering process startup loop"
        set +x
        while true
        do
            if ! pgrep -x $PGREP > /dev/null
            then
                /usr/bin/filter_ready
                /usr/bin/desktop_ready
                set +e
                $START_COMMAND $ARGS $URL
                set -e
            fi
            sleep 1
        done
        set -x

    fi

}

if [ -n "$GO" ] || [ -n "$ASSIGN" ] ; then
    kasm_exec
else
    kasm_startup
fi
