"""Synthetic workload generator to induce KV-cache and queue spikes on vLLM."""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import random
import time

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("workload_generator")


def send_completion(
    base_url: str,
    model: str,
    prompt_tokens_target: int,
    max_tokens: int,
) -> bool:
    """Dispatches an inference request designed to allocate KV-cache blocks."""
    # Synthetic long prompt
    prompt = "Explain quantum gravity and holographic dualities in depth. " * (
        prompt_tokens_target // 10
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    url = f"{base_url}/v1/completions"
    try:
        t0 = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=60, verify=False)
        dur = time.perf_counter() - t0
        if resp.status_code == 200:
            logger.info("Request completed in %.2fs (HTTP 200)", dur)
            return True
        else:
            logger.warning("Request status %d: %s", resp.status_code, resp.text[:120])
    except Exception as e:
        logger.debug("Request dispatch error: %s", e)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM Kimi K3 KV-cache stress generator")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM endpoint URL")
    parser.add_argument("--model", default="kimi-k3", help="Model name")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent client requests")
    parser.add_argument("--rounds", type=int, default=5, help="Number of burst rounds")
    args = parser.parse_args()

    logger.info("Dispatching %d burst rounds with concurrency=%d...", args.rounds, args.concurrency)

    for r in range(args.rounds):
        logger.info("=== Starting Burst Round %d/%d ===", r + 1, args.rounds)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(
                    send_completion,
                    base_url=args.url,
                    model=args.model,
                    prompt_tokens_target=random.randint(1000, 4000),
                    max_tokens=random.randint(256, 1024),
                )
                for _ in range(args.concurrency)
            ]
            concurrent.futures.wait(futures)
        time.sleep(5)


if __name__ == "__main__":
    main()
