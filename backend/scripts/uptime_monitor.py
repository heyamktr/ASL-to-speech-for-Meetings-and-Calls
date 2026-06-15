"""Basic uptime monitor with alerting (Week 5).

Polls the server's health/readiness endpoint on an interval. After N consecutive
failures it fires an alert (POST to a webhook and/or a log line), and fires a
recovery alert when the server comes back. Designed to run as a tiny sidecar
process, a cron job, or inside a container.

Usage:
    python scripts/uptime_monitor.py --url http://localhost:8000/health
    python scripts/uptime_monitor.py --url https://api.example.com/health \\
        --interval 30 --failures 3 --webhook https://hooks.slack.com/services/XXX

Env vars (override flags): MONITOR_URL, MONITOR_INTERVAL, MONITOR_FAILURES,
ALERT_WEBHOOK_URL.

Slack/Discord-compatible: the webhook receives JSON {"text": "..."}.
"""

import argparse
import json
import logging
import os
import time
import urllib.request

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("uptime")


def check(url: str, timeout: float) -> tuple[bool, str]:
    """Return (healthy, detail)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            code = resp.getcode()
            if 200 <= code < 300:
                return True, f"HTTP {code}"
            return False, f"HTTP {code}"
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def send_alert(webhook: str | None, message: str) -> None:
    logger.warning("ALERT: %s", message)
    if not webhook:
        return
    try:
        payload = json.dumps({"text": message}).encode()
        req = urllib.request.Request(
            webhook, data=payload, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=10).read()
        logger.info("alert delivered to webhook")
    except Exception as exc:
        logger.error("failed to deliver alert: %s", exc)


def monitor(
    url: str,
    interval: float,
    failure_threshold: int,
    webhook: str | None,
    timeout: float,
    once: bool = False,
) -> None:
    consecutive_failures = 0
    alerted_down = False

    logger.info(
        "monitoring %s every %.0fs (alert after %d consecutive failures)",
        url, interval, failure_threshold,
    )
    while True:
        healthy, detail = check(url, timeout)
        if healthy:
            if alerted_down:
                send_alert(webhook, f":white_check_mark: RECOVERED — {url} is back up ({detail})")
                alerted_down = False
            consecutive_failures = 0
            logger.info("up (%s)", detail)
        else:
            consecutive_failures += 1
            logger.warning("down (%s) [%d/%d]", detail, consecutive_failures, failure_threshold)
            if consecutive_failures >= failure_threshold and not alerted_down:
                send_alert(
                    webhook,
                    f":rotating_light: DOWN — {url} failed {consecutive_failures} "
                    f"consecutive checks ({detail})",
                )
                alerted_down = True

        if once:
            break
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("MONITOR_URL", "http://localhost:8000/health"))
    parser.add_argument("--interval", type=float,
                        default=float(os.getenv("MONITOR_INTERVAL", "30")))
    parser.add_argument("--failures", type=int, default=int(os.getenv("MONITOR_FAILURES", "3")))
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--webhook", default=os.getenv("ALERT_WEBHOOK_URL"))
    parser.add_argument("--once", action="store_true", help="run a single check and exit")
    args = parser.parse_args()

    monitor(args.url, args.interval, args.failures, args.webhook, args.timeout, once=args.once)
