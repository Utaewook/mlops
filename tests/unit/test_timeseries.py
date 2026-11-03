import numpy as np
import pandas as pd
import pytest

from mlops.tasks.timeseries import (
    FEATURES,
    HORIZON_HOURS,
    LOAD_COLUMNS,
    OT_LAGS,
    SLICES,
    build_features,
    split_by_quarter,
)
from mlops.tasks.timeseries import TASK as TIMESERIES


def make_raw(start="2016-07-01", hours=24 * 10):
    dates = pd.date_range(start, periods=hours, freq="h")
    steps = np.arange(hours, dtype=float)
    loads = {column: steps * (i + 1) for i, column in enumerate(LOAD_COLUMNS)}
    return pd.DataFrame({"date": dates, **loads, "OT": steps + 0.5})


def ot_at(raw, timestamps):
    return raw.set_index("date")["OT"].loc[timestamps].to_numpy()


def test_target_is_oil_temperature_horizon_hours_ahead():
    raw = make_raw()

    features = build_features(raw)

    horizon = pd.Timedelta(HORIZON_HOURS, unit="h")
    assert (features["target_time"] == features["timestamp"] + horizon).all()
    assert np.array_equal(features["target"].to_numpy(), ot_at(raw, features["target_time"]))


def test_lag_features_hold_past_oil_temperature():
    raw = make_raw()

    features = build_features(raw)

    for lag in OT_LAGS:
        past = features["timestamp"] - pd.Timedelta(lag, unit="h")
        assert np.array_equal(features[f"OT_lag{lag}"].to_numpy(), ot_at(raw, past)), lag


def test_features_ignore_values_after_the_prediction_time():
    raw = make_raw()
    cutoff = raw["date"].iloc[100]
    future_changed = raw.assign(
        OT=raw["OT"].where(raw["date"] <= cutoff, -999.0),
        HUFL=raw["HUFL"].where(raw["date"] <= cutoff, -999.0),
    )

    original = build_features(raw)
    changed = build_features(future_changed)

    before = original["timestamp"] <= cutoff
    columns = list(FEATURES)
    pd.testing.assert_frame_equal(
        original.loc[before, columns], changed.loc[original.index[before], columns]
    )


def test_rows_without_full_history_or_target_are_dropped():
    raw = make_raw()

    features = build_features(raw)

    assert features["timestamp"].min() == raw["date"].min() + pd.Timedelta(max(OT_LAGS), unit="h")
    assert features["target_time"].max() == raw["date"].max()
    assert not features[list(FEATURES)].isna().any().any()


def test_irregular_timestamps_are_rejected():
    raw = make_raw().drop(index=50)

    with pytest.raises(ValueError, match="1시간"):
        build_features(raw)


def test_quarter_slices_follow_the_target_time():
    features = build_features(make_raw(start="2016-09-25"))

    slices = split_by_quarter(features)

    assert tuple(slices) == SLICES
    assert (slices["2016Q3"]["target_time"] < pd.Timestamp("2016-10-01")).all()
    assert (slices["2016Q4"]["target_time"] >= pd.Timestamp("2016-10-01")).all()
    assert len(slices["2016Q3"]) + len(slices["2016Q4"]) == len(features)


def test_quarter_slices_reject_data_outside_the_known_range():
    features = build_features(make_raw(start="2019-01-01"))

    with pytest.raises(ValueError, match="2019Q1"):
        split_by_quarter(features)


def test_evaluate_compares_against_the_persistence_forecast():
    df = pd.DataFrame({"target": [10.0, 12.0], "OT_lag0": [9.0, 9.0]})

    metrics = TIMESERIES.evaluate(df, np.array([10.0, 11.0]))

    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["naive_mae"] == pytest.approx(2.0)
    assert metrics["skill"] == pytest.approx(0.75)


def test_model_learns_the_change_and_exports_the_forecast(tmp_path):
    raw = make_raw(hours=24 * 30)
    daily = raw.assign(OT=10.0 + 5.0 * np.sin(2 * np.pi * raw["date"].dt.hour / 24))
    features = build_features(daily)
    params = {**TIMESERIES.default_params, "n_estimators": 50, "learning_rate": 0.3}

    model = TIMESERIES.train(features, features, params)
    TIMESERIES.export(model, tmp_path)

    forecast = TIMESERIES.predict(model, features)
    metrics = TIMESERIES.evaluate(features, forecast)
    assert metrics["mae"] < metrics["naive_mae"] / 2
    assert np.allclose(TIMESERIES.predict_exported(tmp_path, features), forecast, atol=1e-3)
