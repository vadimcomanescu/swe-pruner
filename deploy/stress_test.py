#!/usr/bin/env python3
"""
Stress test for the Winnow pruning service.

Floods the /prune endpoint with concurrent requests using real code files
from this repo, then reports latency percentiles, throughput, and error rates.

Usage:
    python deploy/stress_test.py                          # defaults: local, 20 concurrent
    python deploy/stress_test.py --url http://localhost:17845 -n 50
    python deploy/stress_test.py --url https://your-modal-url.modal.run -n 30 --token sk_winnow_...
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

QUERIES = [
    "error handling and validation",
    "neural network forward pass",
    "API endpoint routing",
    "database query patterns",
    "token scoring and aggregation",
    "chunking and overlap logic",
    "test fixtures and assertions",
    "CLI argument parsing",
    "docker container execution",
    "rate limiting and auth",
]


@dataclass
class RequestResult:
    file: str
    status: int
    latency_s: float
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""
    code_size: int = 0


@dataclass
class StressReport:
    results: list[RequestResult] = field(default_factory=list)
    wall_time_s: float = 0.0

    @property
    def ok(self) -> list[RequestResult]:
        return [r for r in self.results if r.status == 200]

    @property
    def failed(self) -> list[RequestResult]:
        return [r for r in self.results if r.status != 200]

    def print_report(self) -> None:
        total = len(self.results)
        ok = self.ok
        failed = self.failed

        print(f"\n{'=' * 70}")
        print(f"  STRESS TEST REPORT")
        print(f"{'=' * 70}")
        print(f"  Total requests:  {total}")
        print(f"  Successful:      {len(ok)} ({100 * len(ok) / total:.0f}%)")
        print(f"  Failed:          {len(failed)} ({100 * len(failed) / total:.0f}%)")
        print(f"  Wall clock:      {self.wall_time_s:.1f}s")
        if ok:
            print(f"  Throughput:      {len(ok) / self.wall_time_s:.2f} req/s")

        if ok:
            lats = sorted(r.latency_s for r in ok)
            print(f"\n  Latency (successful requests):")
            print(f"    min:   {lats[0]:.3f}s")
            print(f"    p50:   {lats[len(lats) // 2]:.3f}s")
            print(f"    p90:   {lats[int(len(lats) * 0.9)]:.3f}s")
            print(f"    p95:   {lats[int(len(lats) * 0.95)]:.3f}s")
            print(f"    p99:   {lats[min(int(len(lats) * 0.99), len(lats) - 1)]:.3f}s")
            print(f"    max:   {lats[-1]:.3f}s")
            print(f"    mean:  {statistics.mean(lats):.3f}s")

            tokens_in = sum(r.tokens_in for r in ok)
            tokens_out = sum(r.tokens_out for r in ok)
            reduction = (1 - tokens_out / tokens_in) * 100 if tokens_in else 0
            print(f"\n  Token stats:")
            print(f"    Total in:      {tokens_in:,}")
            print(f"    Total out:     {tokens_out:,}")
            print(f"    Reduction:     {reduction:.1f}%")
            print(f"    Avg in/req:    {tokens_in // len(ok):,}")

        if failed:
            print(f"\n  Failures by status code:")
            by_status: dict[int, int] = {}
            for r in failed:
                by_status[r.status] = by_status.get(r.status, 0) + 1
            for code, cnt in sorted(by_status.items()):
                print(f"    {code}: {cnt}")
            print(f"\n  Failed files (first 10):")
            for r in failed[:10]:
                print(f"    {r.file}: {r.status} - {r.error[:80]}")

        # Per-request detail
        print(f"\n  {'File':<45} {'Status':>6} {'Latency':>8} {'In':>7} {'Out':>7}")
        print(f"  {'-' * 45} {'-' * 6} {'-' * 8} {'-' * 7} {'-' * 7}")
        for r in sorted(self.results, key=lambda x: -x.latency_s):
            name = Path(r.file).name[:44]
            status_str = f"{r.status}"
            print(f"  {name:<45} {status_str:>6} {r.latency_s:>7.2f}s {r.tokens_in:>7} {r.tokens_out:>7}")

        print(f"{'=' * 70}\n")


def collect_code_files(repo_root: Path, max_files: int = 60) -> list[Path]:
    """Collect Python files from the repo, sorted by size descending (bigger = harder)."""
    files = []
    for p in repo_root.rglob("*.py"):
        if "__pycache__" in str(p) or ".venv" in str(p):
            continue
        if p.stat().st_size < 100:
            continue
        files.append(p)
    files.sort(key=lambda p: -p.stat().st_size)
    return files[:max_files]


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    file_path: Path,
    query: str,
    headers: dict[str, str],
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    code = file_path.read_text(errors="replace")
    payload = {
        "query": query,
        "code": code,
        "threshold": 0.5,
        "always_keep_first_frags": False,
        "chunk_overlap_tokens": 50,
    }

    async with semaphore:
        t0 = time.monotonic()
        try:
            resp = await client.post(
                f"{url}/prune",
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            latency = time.monotonic() - t0

            if resp.status_code == 200:
                data = resp.json()
                return RequestResult(
                    file=str(file_path),
                    status=200,
                    latency_s=latency,
                    tokens_in=data.get("origin_token_cnt", 0),
                    tokens_out=data.get("left_token_cnt", 0),
                    code_size=len(code),
                )
            else:
                error = resp.text[:200]
                return RequestResult(
                    file=str(file_path),
                    status=resp.status_code,
                    latency_s=latency,
                    error=error,
                    code_size=len(code),
                )
        except httpx.TimeoutException:
            return RequestResult(
                file=str(file_path),
                status=0,
                latency_s=time.monotonic() - t0,
                error="Client timeout (60s)",
                code_size=len(code),
            )
        except Exception as e:
            return RequestResult(
                file=str(file_path),
                status=0,
                latency_s=time.monotonic() - t0,
                error=str(e)[:200],
                code_size=len(code),
            )


async def run_stress_test(
    url: str,
    concurrency: int,
    token: str | None,
    max_files: int,
) -> StressReport:
    repo_root = Path(__file__).parent.parent
    files = collect_code_files(repo_root, max_files)
    print(f"Collected {len(files)} code files")
    print(f"Target: {url}")
    print(f"Concurrency: {concurrency}")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    semaphore = asyncio.Semaphore(concurrency)

    # Assign a query to each file (round-robin)
    tasks = []
    async with httpx.AsyncClient() as client:
        # Health check first
        try:
            health = await client.get(f"{url}/health", timeout=5.0)
            print(f"Health: {health.json()}")
        except Exception as e:
            print(f"Health check failed: {e}")

        t0 = time.monotonic()
        for i, fpath in enumerate(files):
            query = QUERIES[i % len(QUERIES)]
            tasks.append(send_request(client, url, fpath, query, headers, semaphore))

        print(f"Firing {len(tasks)} requests...")
        results = await asyncio.gather(*tasks)
        wall_time = time.monotonic() - t0

    report = StressReport(results=list(results), wall_time_s=wall_time)
    return report


def main():
    parser = argparse.ArgumentParser(description="Stress test Winnow pruning service")
    parser.add_argument("--url", default="http://127.0.0.1:17845", help="Service URL")
    parser.add_argument("-n", "--concurrency", type=int, default=20, help="Max concurrent requests")
    parser.add_argument("--token", default=None, help="Bearer token for auth")
    parser.add_argument("--max-files", type=int, default=50, help="Max files to send")
    args = parser.parse_args()

    report = asyncio.run(run_stress_test(args.url, args.concurrency, args.token, args.max_files))
    report.print_report()


if __name__ == "__main__":
    main()
