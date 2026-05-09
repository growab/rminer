#!/usr/bin/env python3
"""rpow2 GPU miner.

Mines rpow2 SHA-256 proof-of-work on any Vulkan-capable GPU (AMD, NVIDIA,
Intel, integrated). Uses Taichi to JIT-compile a SPIR-V compute shader from
the Python source — no separate driver toolchain required.

Quick start:

    pip install -r requirements.txt
    export RPOW_COOKIE='rpow_session=...'   # paste from your browser DevTools
    python rpow2_gpu_miner.py

Run forever, or stop after N tokens with `--rounds N`. See `--help` for all
flags.
"""

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from queue import Queue

import numpy as np
import taichi as ti

# --------------------------------------------------------------------------
# rpow2 API constants. Override via env if you're testing against a fork.
# --------------------------------------------------------------------------
API_BASE = os.environ.get("RPOW_API_BASE", "https://api.rpow2.com")
ORIGIN = os.environ.get("RPOW_ORIGIN", "https://rpow2.com")
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"


# --------------------------------------------------------------------------
# HTTP helper. Stdlib only, no `requests` dependency.
# --------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, status, body):
        super().__init__(f"http {status}: {body}")
        self.status = status
        self.body = body


def http(method: str, path: str, cookie: str, body=None, timeout: float = 60.0):
    headers = {
        "cookie": cookie,
        "origin": ORIGIN,
        "user-agent": USER_AGENT,
        "accept": "application/json",
    }
    data = None
    if body is not None:
        headers["content-type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{path}", data=data, method=method, headers=headers
    )

    # Infinite retry with capped backoff for network/SSL errors AND 502/503/504
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode("utf-8")
                return r.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"raw": raw}

            # Retry on 502/503/504 (gateway/service errors)
            if e.code in (502, 503, 504):
                attempt += 1
                wait = min(2 ** min(attempt - 1, 4), 20)  # cap at 20s
                print(f"\r\033[KHTTP {e.code} from {path}. Retry #{attempt} in {wait}s...", file=sys.stderr, flush=True)
                time.sleep(wait)
                continue

            raise ApiError(e.code, parsed) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            attempt += 1
            wait = min(2 ** min(attempt - 1, 4), 20)  # cap at 20s
            print(f"\r\033[KNetwork error: {e}. Retry #{attempt} in {wait}s...", file=sys.stderr, flush=True)
            time.sleep(wait)


# --------------------------------------------------------------------------
# Taichi / Vulkan kernel.
#
# rpow2 PoW spec (matches the on-site browser worker):
#   preimage = nonce_prefix_bytes || little-endian uint64 nonce
#   accept iff trailing_zero_bits(SHA-256(preimage)) >= difficulty_bits
#
# `nonce_prefix` is 16 bytes (32 hex chars) issued by POST /challenge.
# Total preimage is 24 bytes; the SHA-256 padded message fits in one block.
# --------------------------------------------------------------------------
ti.init(arch=ti.vulkan, log_level=ti.WARN)

_K_TABLE = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)
_H0_TABLE = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)

K = ti.field(dtype=ti.u32, shape=64)
H0 = ti.field(dtype=ti.u32, shape=8)
for _i, _v in enumerate(_K_TABLE):
    K[_i] = _v
for _i, _v in enumerate(_H0_TABLE):
    H0[_i] = _v


@ti.func
def _rotr(x: ti.u32, n: ti.i32) -> ti.u32:
    return (x >> n) | (x << (32 - n))


@ti.kernel
def mine_kernel(
    n_threads: ti.i32,
    iters: ti.i32,
    base_nonce_lo: ti.u32,
    base_nonce_hi: ti.u32,
    p0: ti.u32, p1: ti.u32, p2: ti.u32, p3: ti.u32,
    bit_mask: ti.u32,
    check_h6: ti.i32,  # 1 if target_bits > 32, else 0
    h6_mask: ti.u32,   # mask for h6 when checking > 32 bits
    result: ti.types.ndarray(dtype=ti.u32, ndim=1),  # [found, nonce_lo, nonce_hi]
):
    for gid in range(n_threads):
        # 64-bit arithmetic: local_base = base_nonce + gid * iters
        offset = ti.u64(gid) * ti.u64(iters)
        local_base_lo = base_nonce_lo + ti.u32(offset & ti.u64(0xFFFFFFFF))
        local_base_hi = base_nonce_hi + ti.u32(offset >> 32)
        # Handle carry from lo to hi
        if local_base_lo < base_nonce_lo:
            local_base_hi += ti.u32(1)

        for k in range(iters):
            # 64-bit nonce = local_base + k
            nonce_lo = local_base_lo + ti.u32(k)
            nonce_hi = local_base_hi
            if nonce_lo < local_base_lo:  # carry
                nonce_hi += ti.u32(1)

            # SHA-256 reads each 4-byte word big-endian; nonce bytes are LE,
            # so W[4] is the byte-swap of nonce_lo, W[5] is byte-swap of nonce_hi
            w4 = ((nonce_lo & ti.u32(0xff)) << 24) \
                | (((nonce_lo >> 8) & ti.u32(0xff)) << 16) \
                | (((nonce_lo >> 16) & ti.u32(0xff)) << 8) \
                | ((nonce_lo >> 24) & ti.u32(0xff))
            w5 = ((nonce_hi & ti.u32(0xff)) << 24) \
                | (((nonce_hi >> 8) & ti.u32(0xff)) << 16) \
                | (((nonce_hi >> 16) & ti.u32(0xff)) << 8) \
                | ((nonce_hi >> 24) & ti.u32(0xff))

            W = ti.Vector.zero(ti.u32, 64)
            W[0] = p0
            W[1] = p1
            W[2] = p2
            W[3] = p3
            W[4] = w4
            W[5] = w5
            W[6] = ti.u32(0x80000000)  # SHA-256 padding marker
            W[15] = ti.u32(192)        # bit length: 24 bytes = 192 bits

            for t in range(16, 64):
                s0 = _rotr(W[t - 15], 7) ^ _rotr(W[t - 15], 18) ^ (W[t - 15] >> 3)
                s1 = _rotr(W[t - 2], 17) ^ _rotr(W[t - 2], 19) ^ (W[t - 2] >> 10)
                W[t] = W[t - 16] + s0 + W[t - 7] + s1

            a = H0[0]; b = H0[1]; c = H0[2]; d = H0[3]
            e = H0[4]; f = H0[5]; g = H0[6]; h = H0[7]

            for t in range(64):
                S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
                ch = (e & f) ^ ((~e) & g)
                temp1 = h + S1 + ch + K[t] + W[t]
                S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
                mj = (a & b) ^ (a & c) ^ (b & c)
                temp2 = S0 + mj
                h = g; g = f; f = e
                e = d + temp1
                d = c; c = b; b = a
                a = temp1 + temp2

            h7 = H0[7] + h
            h6 = H0[6] + g
            # For target_bits in [1..32], "trailing zero bits >= target" iff
            # (h7 & ((1<<bits)-1)) == 0, because h7 is the last 4 digest bytes
            # and the rpow2 trailing-zero count starts at byte[31].
            # For 33-34 bits: h7 must be 0 AND check h6 with mask
            match = False
            if check_h6 == 1:
                # 33-34 bits: h7==0 and (h6 & h6_mask)==0
                if h7 == ti.u32(0) and (h6 & h6_mask) == ti.u32(0):
                    match = True
            else:
                # 1-32 bits: check h7 only
                if (h7 & bit_mask) == ti.u32(0):
                    match = True

            if match:
                prev = ti.atomic_or(result[0], ti.u32(1))
                if prev == ti.u32(0):
                    result[1] = nonce_lo
                    result[2] = nonce_hi


def _hex_prefix_to_uint32_be(prefix_hex: str):
    pb = bytes.fromhex(prefix_hex)
    if len(pb) != 16:
        raise ValueError(f"expected 16-byte prefix, got {len(pb)}")
    return tuple(int.from_bytes(pb[i:i + 4], "big") for i in range(0, 16, 4))


def _trailing_zero_bits(d: bytes) -> int:
    z = 0
    for byte in reversed(d):
        if byte == 0:
            z += 8
            continue
        c = 0
        while not (byte & (1 << c)):
            c += 1
        return z + c
    return z


def verify(prefix_hex: str, nonce: int, target_bits: int) -> bool:
    """Sanity-check that a (prefix, nonce) really meets the target."""
    msg = bytes.fromhex(prefix_hex) + nonce.to_bytes(8, "little")
    return _trailing_zero_bits(hashlib.sha256(msg).digest()) >= target_bits


def solve(prefix_hex, target_bits, n_threads, iters, attempt_cap, challenge_id=None, quiet=False, status_callback=None):
    """Return (winning_nonce, total_attempts). Raises RuntimeError on cap."""
    if not (1 <= target_bits <= 34):
        raise ValueError(f"target_bits must be 1..34, got {target_bits}")
    p0, p1, p2, p3 = _hex_prefix_to_uint32_be(prefix_hex)
    # For 33-34 bits: need h7==0 (32 bits) + 1-2 bits from h6
    if target_bits <= 32:
        bit_mask = np.uint32((1 << target_bits) - 1)
        check_h6 = 0
        h6_mask = np.uint32(0)
    else:
        bit_mask = np.uint32(0xFFFFFFFF)  # all 32 bits for h7
        check_h6 = 1
        # 33 bits: h6 & 1 == 0, 34 bits: h6 & 3 == 0
        h6_mask = np.uint32((1 << (target_bits - 32)) - 1)
    base = np.uint64(0)
    total = 0
    t_start = time.time()
    last_update = t_start

    while total < attempt_cap:
        result_buf = np.zeros(3, dtype=np.uint32)
        base_lo = np.uint32(base & 0xFFFFFFFF)
        base_hi = np.uint32((base >> 32) & 0xFFFFFFFF)
        mine_kernel(
            n_threads, iters, base_lo, base_hi,
            np.uint32(p0), np.uint32(p1), np.uint32(p2), np.uint32(p3),
            bit_mask, check_h6, h6_mask, result_buf,
        )
        ti.sync()
        total += n_threads * iters

        # Live status update every 0.5s
        now = time.time()
        if not quiet and (now - last_update) >= 0.5:
            elapsed = now - t_start
            hashrate = total / elapsed if elapsed > 0 else 0
            if status_callback:
                status_callback(challenge_id, target_bits, total, hashrate)
            last_update = now

        if int(result_buf[0]) == 1:
            nonce_lo = int(result_buf[1])
            nonce_hi = int(result_buf[2])
            nonce = nonce_lo | (nonce_hi << 32)
            if not verify(prefix_hex, nonce, target_bits):
                raise RuntimeError(
                    f"kernel returned nonce={nonce} that does not verify on CPU; "
                    "this should never happen — please file a bug"
                )
            return nonce, total
        base += np.uint64(n_threads) * np.uint64(iters)

    raise RuntimeError(f"attempt_cap reached after {total} hashes")


# --------------------------------------------------------------------------
# Status bar rendering
# --------------------------------------------------------------------------
def render_status_bar(stats, started_at):
    """Render persistent status bar at bottom of terminal."""
    elapsed = time.time() - started_at
    rate = stats["minted"] / elapsed if elapsed > 0 else 0

    status = (
        f"SESSION: minted={stats['minted']}  pending={stats['pending']}  "
        f"failures={stats['failures']}  elapsed={elapsed:.0f}s  "
        f"rate={rate:.2f}/s (~{rate*3600:.0f}/hr)"
    )

    # Clear line and print status with \r (carriage return)
    print(f"\r\033[K{status}", end="", flush=True, file=sys.stderr)


# --------------------------------------------------------------------------
# Live mining loop.
# --------------------------------------------------------------------------
def submit_worker(cookie, result_queue, stats, stop_event, quiet, started_at):
    """Background thread: submit solved challenges to /mint."""
    while not stop_event.is_set():
        try:
            item = result_queue.get(timeout=0.5)
        except:
            continue
        if item is None:  # poison pill
            break

        cid, nonce, bits, solve_ms, attempts = item
        stats["pending"] -= 1
        try:
            _, m = http(
                "POST", "/mint", cookie,
                {"challenge_id": cid, "solution_nonce": str(nonce)},
            )
            stats["minted"] += 1
            token_id = (m or {}).get("token", {}).get("id", "?")
            if not quiet:
                pending = stats["pending"]
                print(
                    f"\r\033[Kminted #{stats['minted']:<5d}  bits={bits}  solve={solve_ms:>5.0f}ms  "
                    f"attempts={attempts:>10,}  token={token_id}  pending={pending}",
                    flush=True,
                )
        except ApiError as e:
            stats["failures"] += 1
            print(f"\r\033[K[!] /mint failed (challenge {cid}): {e}",
                  file=sys.stderr, flush=True)


def main():
    p = argparse.ArgumentParser(
        description="GPU-accelerated rpow2 miner (Vulkan/Taichi).",
    )
    p.add_argument(
        "--cookie",
        default=os.environ.get("RPOW_COOKIE"),
        help="rpow2 session cookie. Format: 'rpow_session=<value>'. "
             "Defaults to $RPOW_COOKIE.",
    )
    p.add_argument("--rounds", type=int, default=0,
                   help="stop after N successful mints (0 = run forever)")
    p.add_argument("--threads", type=int, default=1 << 20,
                   help="GPU threads per kernel launch (default: 1048576)")
    p.add_argument("--iters", type=int, default=64,
                   help="nonces per thread per launch (default: 64)")
    p.add_argument("--attempt-cap", type=int, default=16,
                   help="multiplier for max attempts (default: 16 = 16x expected for difficulty, 0 = unlimited)")
    p.add_argument("--quiet", action="store_true",
                   help="only print summary, no per-mint lines")
    p.add_argument("--no-pipeline", action="store_true",
                   help="disable parallel submit (for debugging)")
    args = p.parse_args()

    if not args.cookie:
        sys.exit(
            "no cookie supplied. set $RPOW_COOKIE or pass --cookie. "
            "tip: in your browser, DevTools → Network → click any /api request "
            "→ copy the 'cookie' header (starts with 'rpow_session=')."
        )
    if not args.cookie.startswith("rpow_session="):
        sys.exit("cookie does not start with 'rpow_session='. paste the full "
                 "header value, not just the JWT-looking part.")

    # Confirm auth before warming up the GPU.
    try:
        status, me = http("GET", "/me", args.cookie)
    except ApiError as e:
        sys.exit(f"auth check failed: {e}")
    if status != 200 or not me or "email" not in me:
        sys.exit(f"unexpected /me response: {me}")

    # Support both v1 and v2 API response formats
    # v2: balance_base_units, minted_base_units (string, base units)
    # v1: balance, minted (integer, token count)
    if "balance_base_units" in me:
        # v2 format: convert base units to tokens (divide by 1e9)
        balance = int(me["balance_base_units"]) / 1e9
        minted = int(me["minted_base_units"]) / 1e9
    else:
        # v1 format: direct token counts
        balance = me.get("balance", 0)
        minted = me.get("minted", 0)

    print(f"signed in: {me['email']}  balance={balance:.2f}  minted={minted:.2f}",
          file=sys.stderr, flush=True)

    print("compiling SPIR-V kernel (first launch only)...", file=sys.stderr, flush=True)
    # Warm-up — never finds (mask=0xFFFFFFFF) but JIT-compiles the kernel.
    warm = np.zeros(3, dtype=np.uint32)
    mine_kernel(
        args.threads, args.iters, np.uint32(0), np.uint32(0),
        np.uint32(0), np.uint32(0), np.uint32(0), np.uint32(0),
        np.uint32(0xFFFFFFFF), 0, np.uint32(0), warm,
    )
    ti.sync()
    print("kernel ready.\n", file=sys.stderr, flush=True)

    stats = {"minted": 0, "failures": 0, "pending": 0}
    started_at = time.time()
    stop_event = threading.Event()
    result_queue = Queue(maxsize=4)  # buffer up to 4 solved challenges

    def stop_summary(*_):
        elapsed = time.time() - started_at
        print(file=sys.stderr)
        print("---- summary ----", file=sys.stderr)
        print(f"minted:    {stats['minted']}", file=sys.stderr)
        print(f"failures:  {stats['failures']}", file=sys.stderr)
        print(f"elapsed:   {elapsed:.1f}s", file=sys.stderr)
        if elapsed > 0:
            print(f"avg rate:  {stats['minted']/elapsed:.2f} tokens/sec  "
                  f"(~{stats['minted']/elapsed*3600:.0f}/hour)", file=sys.stderr)
        stop_event.set()
        result_queue.put(None)  # poison pill
        sys.exit(0)

    signal.signal(signal.SIGINT,  stop_summary)
    signal.signal(signal.SIGTERM, stop_summary)

    # Start submit worker thread (unless disabled)
    if not args.no_pipeline:
        submit_thread = threading.Thread(
            target=submit_worker,
            args=(args.cookie, result_queue, stats, stop_event, args.quiet, started_at),
            daemon=True,
        )
        submit_thread.start()

    def update_mining_status(cid, bits, attempts, hashrate):
        """Callback for live mining status."""
        cid_short = cid[:8] if cid else "?"
        print(
            f"\r\033[K[mining] cid={cid_short}...  bits={bits}  "
            f"attempts={attempts:>12,}  rate={hashrate/1e6:>6.1f}MH/s",
            end="", file=sys.stderr, flush=True
        )

    while True:
        if args.rounds and stats["minted"] >= args.rounds:
            break

        try:
            _, ch = http("POST", "/challenge", args.cookie)
        except ApiError as e:
            stats["failures"] += 1
            print(f"\r\033[K[!] /challenge failed: {e}", file=sys.stderr, flush=True)
            time.sleep(1.0)
            continue

        cid    = ch["challenge_id"]
        prefix = ch["nonce_prefix"]
        bits   = ch["difficulty_bits"]

        # Calculate attempt cap from multiplier
        # multiplier=0 means unlimited, otherwise: cap = 2^(bits + log2(multiplier))
        if args.attempt_cap == 0:
            cap = 1 << 63  # effectively unlimited (max uint64 range)
        else:
            # For multiplier M: expected attempts = 2^bits, so M*2^bits = 2^(bits + log2(M))
            # For M=16: 2^(bits+4), for M=32: 2^(bits+5), etc.
            import math
            multiplier_bits = int(math.log2(args.attempt_cap)) if args.attempt_cap > 0 else 0
            cap = max(1 << 38, 1 << (bits + multiplier_bits))

        t0 = time.time()
        try:
            nonce, attempts = solve(
                prefix, bits, args.threads, args.iters, cap,
                challenge_id=cid, quiet=args.quiet,
                status_callback=update_mining_status if not args.no_pipeline else None,
            )
        except RuntimeError as e:
            stats["failures"] += 1
            print(f"\r\033[K[!] solve failed for challenge {cid}: {e}",
                  file=sys.stderr, flush=True)
            continue
        solve_ms = (time.time() - t0) * 1000.0

        if args.no_pipeline:
            # Old synchronous path
            try:
                _, m = http(
                    "POST", "/mint", args.cookie,
                    {"challenge_id": cid, "solution_nonce": str(nonce)},
                )
            except ApiError as e:
                stats["failures"] += 1
                print(f"\r\033[K[!] /mint failed (challenge {cid}): {e}",
                      file=sys.stderr, flush=True)
                continue

            stats["minted"] += 1
            token_id = (m or {}).get("token", {}).get("id", "?")
            if not args.quiet:
                print(
                    f"\r\033[Kminted #{stats['minted']:<5d}  bits={bits}  solve={solve_ms:>5.0f}ms  "
                    f"attempts={attempts:>10,}  token={token_id}",
                    flush=True,
                )
        else:
            # Pipeline: queue result for background submit
            stats["pending"] += 1
            result_queue.put((cid, nonce, bits, solve_ms, attempts))

    stop_summary()


if __name__ == "__main__":
    main()
