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
# Caddy reverse proxy for CDP (bypass Chrome Host header check)
# ---------------------------------------------------------------------------
CDP_PORT="${CDP_PORT:-9222}"
CHROME_CDP_PORT="9223"

MCP_PROXY_PORT="8765"

cat > /tmp/Caddyfile << EOF
{
  auto_https off
  admin off
}
:${CDP_PORT} {
  handle /mcp {
    reverse_proxy 127.0.0.1:${MCP_PROXY_PORT}
  }
  handle /sse {
    reverse_proxy 127.0.0.1:${MCP_PROXY_PORT}
  }
  handle /ping {
    reverse_proxy 127.0.0.1:${MCP_PROXY_PORT}
  }
  handle {
    reverse_proxy 127.0.0.1:${CHROME_CDP_PORT}
  }
}
EOF

caddy run --config /tmp/Caddyfile &
echo "[openclaw] Caddy started (:${CDP_PORT} -> CDP :${CHROME_CDP_PORT} + MCP :${MCP_PROXY_PORT})"

# ---------------------------------------------------------------------------
# MCP Server (chrome-devtools-mcp via mcp-proxy, streamable HTTP)
# chrome-devtools-mcp uses lazy connection: connects to Chrome on first tool
# call, auto-reconnects if Chrome restarts (--browserUrl mode)
# ---------------------------------------------------------------------------
mcp-proxy --port "${MCP_PROXY_PORT}" -- \
  chrome-devtools-mcp \
    --browserUrl "http://127.0.0.1:${CHROME_CDP_PORT}" \
    --no-usage-statistics \
    --no-performance-crux &
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
