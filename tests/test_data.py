from datetime import datetime, timezone

import numpy as np
import pytest

from pmvol.data import MarketJob, candles_to_frame, finalize_panel


def _candle(ts, bid, ask, volume="0.00"):
    return {
        "end_period_ts": ts,
        "yes_bid": {"close": bid},
        "yes_ask": {"close": ask},
        "price": {"close": None, "previous": "0.40"},
        "volume": volume,
        "open_interest": "12.00",
    }


def test_candles_to_panel_enforces_exact_horizon_and_no_lookahead_deadline():
    market = {
        "ticker": "TEST-1",
        "open_time": "2025-01-01T00:00:00Z",
        "close_time": "2025-01-01T05:00:00Z",
        "expected_expiration_time": "2025-01-01T06:00:00Z",
    }
    job = MarketJob(
        market=market,
        series="TEST",
        category="Test",
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 1, 1, 5, tzinfo=timezone.utc),
    )
    base = int(job.start.timestamp())
    candles = [
        _candle(base + 3600, "0.30", "0.50", "3.00"),
        _candle(base + 7200, "0.32", "0.52", "0.00"),
        # Deliberate two-hour gap: this row must not forecast the next candle.
        _candle(base + 14400, "0.40", "0.60", "1.00"),
    ]
    raw = candles_to_frame(job, candles)
    assert raw.loc[0, "price"] == 0.4
    assert raw.loc[0, "spread"] == 0.2
    assert raw.loc[0, "time_to_resolution"] == 5
    panel = finalize_panel(raw)
    assert bool(panel.loc[0, "valid_forecast"])
    assert panel.loc[0, "innovation"] == pytest.approx(0.02)
    assert not bool(panel.loc[1, "valid_one_hour"])
    assert np.isnan(panel.loc[2, "price_next"])
