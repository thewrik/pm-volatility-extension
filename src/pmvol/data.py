"""Restartable acquisition of public Kalshi hourly candlesticks.

The pipeline deliberately separates raw API payloads from the analysis panel.
Every network response is cached, and a manifest records configuration and file
hashes so the independent replication can be audited later.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_ROOT = "https://external-api.kalshi.com/trade-api/v2"
UTC = timezone.utc


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _number(value: Any) -> float:
    if value is None or value == "":
        return np.nan
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class MarketJob:
    market: dict[str, Any]
    series: str
    category: str
    start: datetime
    end: datetime

    @property
    def ticker(self) -> str:
        return str(self.market["ticker"])


class KalshiClient:
    """Small public-data client with per-thread sessions and bounded retries."""

    def __init__(self, api_root: str = API_ROOT, timeout: float = 30.0) -> None:
        self.api_root = api_root.rstrip("/")
        self.timeout = timeout
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            retry = Retry(
                total=7,
                backoff_factor=0.6,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET",),
                respect_retry_after_header=True,
            )
            session = requests.Session()
            session.headers.update(
                {"User-Agent": "pm-volatility-extension/0.1 (academic replication)"}
            )
            session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=32))
            self._local.session = session
        return session

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self._session().get(
            f"{self.api_root}/{path.lstrip('/')}", params=params, timeout=self.timeout
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"unexpected non-object response for {path}")
        return payload

    def historical_markets(self, series_ticker: str) -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        cursor = ""
        while True:
            params: dict[str, Any] = {"series_ticker": series_ticker, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            payload = self.get_json("historical/markets", params)
            markets.extend(payload.get("markets", []))
            cursor = str(payload.get("cursor", ""))
            if not cursor:
                return markets

    def historical_candlesticks(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        period_minutes: int,
    ) -> dict[str, Any]:
        return self.get_json(
            f"historical/markets/{quote(ticker, safe='')}/candlesticks",
            {
                "start_ts": int(start.timestamp()),
                "end_ts": int(end.timestamp()),
                "period_interval": period_minutes,
            },
        )


def _known_deadline(market: dict[str, Any]) -> datetime | None:
    """Choose the scheduled deadline without using realized settlement timing."""

    for key in ("expected_expiration_time", "expiration_time", "latest_expiration_time"):
        value = _parse_time(market.get(key))
        if value is not None:
            return value
    return _parse_time(market.get("close_time"))


def select_market_jobs(config: dict[str, Any], client: KalshiClient) -> list[MarketJob]:
    global_start = _parse_time(config["start"])
    global_end = _parse_time(config["end"])
    if global_start is None or global_end is None or global_start >= global_end:
        raise ValueError("config start/end are invalid")
    min_lifetime = float(config.get("minimum_market_lifetime_hours", 48))
    min_volume = float(config.get("minimum_market_volume", 0))
    cap = config.get("max_markets_per_series")
    sampling = str(config.get("market_sampling", "recent"))
    if sampling not in {"recent", "time_stratified"}:
        raise ValueError("market_sampling must be 'recent' or 'time_stratified'")
    jobs: list[MarketJob] = []

    for spec in config["series"]:
        series = str(spec["ticker"])
        category = str(spec["category"])
        candidates: list[MarketJob] = []
        for market in client.historical_markets(series):
            open_time = _parse_time(market.get("open_time"))
            close_time = _parse_time(market.get("close_time"))
            deadline = _known_deadline(market)
            if open_time is None or close_time is None or deadline is None:
                continue
            query_start = max(open_time, global_start)
            query_end = min(close_time, global_end)
            lifetime_hours = (deadline - open_time).total_seconds() / 3600.0
            if query_end <= query_start or lifetime_hours < min_lifetime:
                continue
            if _number(market.get("volume_fp", market.get("volume"))) < min_volume:
                continue
            candidates.append(
                MarketJob(
                    market=market,
                    series=series,
                    category=category,
                    start=query_start,
                    end=query_end,
                )
            )

        # A smoke-test cap is deterministic and cannot use future score outcomes.
        candidates.sort(
            key=lambda job: (
                _parse_time(job.market.get("settlement_ts")) or datetime.min.replace(tzinfo=UTC),
                job.ticker,
            ),
        )
        if cap is not None and len(candidates) > int(cap):
            cap = int(cap)
            if sampling == "recent":
                candidates = candidates[-cap:]
            else:
                # Systematic sampling spans the full time range and depends only
                # on metadata ordering, never forecast outcomes or scores.
                indices = np.linspace(0, len(candidates) - 1, cap).round().astype(int)
                candidates = [candidates[index] for index in indices]
        jobs.extend(candidates)
    return jobs


def _chunks(start: datetime, end: datetime, days: int = 80) -> Iterable[tuple[datetime, datetime]]:
    cursor = start
    step = timedelta(days=days)
    while cursor < end:
        chunk_end = min(cursor + step, end)
        yield cursor, chunk_end
        # Inclusive API endpoints can duplicate the boundary candle; de-duplicate later.
        cursor = chunk_end


def _write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(path)


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid cached payload: {path}")
    return payload


def fetch_market(
    job: MarketJob,
    client: KalshiClient,
    cache_dir: Path,
    *,
    period_minutes: int,
    refresh: bool = False,
) -> tuple[MarketJob, list[dict[str, Any]], list[Path]]:
    candles: list[dict[str, Any]] = []
    cache_paths: list[Path] = []
    for start, end in _chunks(job.start, job.end):
        stamp = f"{int(start.timestamp())}-{int(end.timestamp())}-{period_minutes}"
        path = cache_dir / job.series / job.ticker / f"{stamp}.json.gz"
        if path.exists() and not refresh:
            payload = _read_gzip_json(path)
        else:
            payload = client.historical_candlesticks(
                job.ticker, start, end, period_minutes=period_minutes
            )
            _write_gzip_json(path, payload)
            # Gentle pacing per worker makes accidental API pressure less likely.
            time.sleep(0.04)
        candles.extend(payload.get("candlesticks", []))
        cache_paths.append(path)
    deduplicated = {int(row["end_period_ts"]): row for row in candles}
    return job, [deduplicated[key] for key in sorted(deduplicated)], cache_paths


def candles_to_frame(job: MarketJob, candles: list[dict[str, Any]]) -> pd.DataFrame:
    deadline = _known_deadline(job.market)
    if deadline is None:
        raise ValueError(f"missing deadline for {job.ticker}")
    rows: list[dict[str, Any]] = []
    for candle in candles:
        bid = _number(candle.get("yes_bid", {}).get("close"))
        ask = _number(candle.get("yes_ask", {}).get("close"))
        price_fields = candle.get("price", {})
        timestamp = datetime.fromtimestamp(int(candle["end_period_ts"]), tz=UTC)
        quote_valid = np.isfinite(bid) and np.isfinite(ask) and 0 <= bid <= ask <= 1
        rows.append(
            {
                "ticker": job.ticker,
                "series": job.series,
                "category": job.category,
                "timestamp": timestamp,
                "deadline": deadline,
                "price": (bid + ask) / 2.0 if quote_valid else np.nan,
                "bid": bid,
                "ask": ask,
                "spread": ask - bid if quote_valid else np.nan,
                "last_price": _number(price_fields.get("close")),
                "previous_last_price": _number(price_fields.get("previous")),
                "volume": _number(candle.get("volume")),
                "open_interest": _number(candle.get("open_interest")),
                "time_to_resolution": (deadline - timestamp).total_seconds() / 3600.0,
            }
        )
    return pd.DataFrame(rows)


def finalize_panel(frame: pd.DataFrame, period_minutes: int = 60) -> pd.DataFrame:
    if frame.empty:
        return frame
    panel = frame.sort_values(["ticker", "timestamp"]).drop_duplicates(
        ["ticker", "timestamp"], keep="last"
    )
    group = panel.groupby("ticker", sort=False, observed=True)
    panel["next_timestamp"] = group["timestamp"].shift(-1)
    panel["price_next"] = group["price"].shift(-1)
    expected_seconds = period_minutes * 60
    panel["horizon_seconds"] = (
        panel["next_timestamp"] - panel["timestamp"]
    ).dt.total_seconds()
    panel["innovation"] = panel["price_next"] - panel["price"]
    panel["updated"] = (panel["innovation"].abs() > 1e-12).astype("Int8")
    panel["lag_updated"] = group["updated"].shift(1).fillna(0).astype("Int8")
    panel["forecast_month"] = panel["timestamp"].dt.strftime("%Y-%m")
    panel["valid_one_hour"] = panel["horizon_seconds"] == expected_seconds
    panel["valid_forecast"] = (
        panel["valid_one_hour"]
        & panel["price"].between(0, 1, inclusive="neither")
        & panel["price_next"].between(0, 1, inclusive="both")
        & panel["spread"].between(0, 1, inclusive="both")
        # The forecast horizon must end no later than the scheduled deadline;
        # otherwise the finite deadline clock would mechanically release all
        # uncertainty even though the API may continue showing stale quotes.
        & (panel["time_to_resolution"] >= period_minutes / 60.0)
    )
    return panel.reset_index(drop=True)


def build_panel(
    config_path: Path,
    output_path: Path,
    *,
    workers: int = 8,
    refresh: bool = False,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    period_minutes = int(config.get("period_minutes", 60))
    data_root = output_path.parent.parent
    cache_dir = data_root / "raw" / "candlesticks"
    client = KalshiClient()
    jobs = select_market_jobs(config, client)
    if not jobs:
        raise RuntimeError("configuration selected no markets")

    frames: list[pd.DataFrame] = []
    cache_paths: list[Path] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                fetch_market,
                job,
                client,
                cache_dir,
                period_minutes=period_minutes,
                refresh=refresh,
            ): job
            for job in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                returned_job, candles, paths = future.result()
                frames.append(candles_to_frame(returned_job, candles))
                cache_paths.extend(paths)
            except Exception as exc:  # retain an auditable failure list and continue
                failures.append({"ticker": job.ticker, "error": repr(exc)})
            if completed % 50 == 0 or completed == len(futures):
                print(
                    f"downloaded {completed}/{len(futures)} markets; "
                    f"failures={len(failures)}",
                    flush=True,
                )

    if not frames:
        raise RuntimeError(f"all market downloads failed: {failures[:3]}")
    panel = finalize_panel(pd.concat(frames, ignore_index=True), period_minutes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False, compression="gzip")

    manifest = {
        "created_at": _iso(datetime.now(UTC)),
        "api_root": client.api_root,
        "config": config,
        "config_sha256": _sha256(config_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "markets_selected": len(jobs),
        "markets_succeeded": len(frames),
        "failures": failures,
        "rows": int(len(panel)),
        "valid_forecasts": int(panel["valid_forecast"].sum()),
        "active_updates": int((panel["valid_forecast"] & (panel["updated"] == 1)).sum()),
        "cache_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in sorted(set(cache_paths))
        ],
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/core_panel.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/core_panel.csv.gz")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_panel(args.config, args.output, workers=args.workers, refresh=args.refresh)
    print(json.dumps({key: value for key, value in manifest.items() if key != "cache_files"}, indent=2))


if __name__ == "__main__":
    main()
