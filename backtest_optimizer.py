#!/usr/bin/env python3
"""
STH-MNVH V6 パラメータ最適化バックテスト
Hyperliquid 1分足データを使用

使い方:
  python3 backtest_optimizer.py

出力:
  - コンソールにグリッドサーチ結果
  - data/backtest_results.json に詳細結果
"""

import json
import math
import time
import statistics
import itertools
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ======================================================================
# 1. Hyperliquid Data Fetcher
# ======================================================================

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"

# ======================================================================
# Synthetic Data Generator (for when API is unavailable)
# ======================================================================

# Realistic 1-min crypto market parameters (calibrated from empirical data)
COIN_PROFILES = {
    "BTC":  {"price": 85000.0, "vol_ann": 0.55, "vol_usd_1m": 5_000_000, "mr_rho": 0.08},
    "ETH":  {"price": 3200.0,  "vol_ann": 0.65, "vol_usd_1m": 3_000_000, "mr_rho": 0.10},
    "SOL":  {"price": 140.0,   "vol_ann": 0.85, "vol_usd_1m": 1_500_000, "mr_rho": 0.12},
    "DOGE": {"price": 0.18,    "vol_ann": 0.95, "vol_usd_1m": 800_000,   "mr_rho": 0.15},
    "XRP":  {"price": 2.40,    "vol_ann": 0.75, "vol_usd_1m": 1_000_000, "mr_rho": 0.11},
    "AVAX": {"price": 28.0,    "vol_ann": 0.80, "vol_usd_1m": 500_000,   "mr_rho": 0.13},
    "LINK": {"price": 18.0,    "vol_ann": 0.78, "vol_usd_1m": 400_000,   "mr_rho": 0.12},
    "ARB":  {"price": 0.55,    "vol_ann": 0.90, "vol_usd_1m": 350_000,   "mr_rho": 0.14},
}


def generate_synthetic_candles(
    coin: str,
    n_bars: int = 4320,
    seed: Optional[int] = None,
    base_timestamp_ms: Optional[int] = None,
) -> pd.DataFrame:
    """
    暗号資産1分足の合成データを生成

    - 対数正規リターン + 平均回帰 + fat tails (Student-t df=5)
    - GARCH(1,1) ボラティリティクラスタリング
    - 日中パターン（セッション周期性）
    """
    profile = COIN_PROFILES.get(coin, {
        "price": 10.0, "vol_ann": 0.80, "vol_usd_1m": 200_000, "mr_rho": 0.12
    })

    rng = np.random.default_rng(seed)
    sigma_1m = profile["vol_ann"] / math.sqrt(525600)
    rho = profile["mr_rho"]

    returns = np.zeros(n_bars)
    prices = np.zeros(n_bars)
    volumes = np.zeros(n_bars)

    prices[0] = profile["price"]
    base_vol = profile["vol_usd_1m"]

    h = sigma_1m ** 2
    garch_alpha = 0.08
    garch_beta = 0.88
    garch_omega = sigma_1m ** 2 * (1 - garch_alpha - garch_beta)

    cumulative_drift = 0.0

    for i in range(1, n_bars):
        z = rng.standard_t(df=5) * math.sqrt(3.0 / 5.0)
        h = garch_omega + garch_alpha * (returns[i - 1] ** 2) + garch_beta * h
        sigma_t = math.sqrt(max(h, 1e-16))

        mr_component = -rho * cumulative_drift
        r = mr_component + sigma_t * z
        returns[i] = r
        cumulative_drift += r

        prices[i] = prices[i - 1] * math.exp(r)

        minute_of_day = i % 1440
        diurnal = 1.0 + 0.5 * math.sin(2 * math.pi * (minute_of_day - 870) / 1440)
        diurnal += 0.3 * math.exp(-((minute_of_day) ** 2) / (2 * 120 ** 2))
        vol_shock = max(0.3, 1.0 + 2.0 * (abs(r) / sigma_1m - 1.0))
        volumes[i] = base_vol * diurnal * vol_shock * rng.lognormal(0, 0.3)

    volumes[0] = base_vol

    if base_timestamp_ms is None:
        base_timestamp_ms = int(time.time() * 1000) - n_bars * 60000
    timestamps = [base_timestamp_ms + i * 60000 for i in range(n_bars)]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.roll(prices, 1),
        "high": prices * (1 + np.abs(returns) * 0.3),
        "low": prices * (1 - np.abs(returns) * 0.3),
        "close": prices,
        "volume": volumes,
    })
    df.loc[0, "open"] = prices[0]
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def add_cross_correlation(
    candle_dict: Dict[str, pd.DataFrame],
    market_factor_strength: float = 0.4,
    seed: int = 123,
) -> Dict[str, pd.DataFrame]:
    """銘柄間に市場ファクター相関を追加"""
    rng = np.random.default_rng(seed)
    coins = list(candle_dict.keys())
    n_bars = min(len(df) for df in candle_dict.values())

    market_returns = rng.standard_t(df=5, size=n_bars) * 0.001 * math.sqrt(3.0 / 5.0)

    for coin in coins:
        df = candle_dict[coin].copy()
        log_prices = np.log(df["close"].values[:n_bars])
        idio_returns = np.diff(log_prices, prepend=log_prices[0])

        blended_returns = (
            (1 - market_factor_strength) * idio_returns
            + market_factor_strength * market_returns[:len(idio_returns)]
        )

        new_prices = np.exp(np.cumsum(blended_returns) + log_prices[0] - blended_returns[0])
        df.loc[:n_bars - 1, "close"] = new_prices
        df.loc[:n_bars - 1, "open"] = np.roll(new_prices, 1)
        df.loc[0, "open"] = new_prices[0]
        df.loc[:n_bars - 1, "high"] = new_prices * (1 + np.abs(blended_returns) * 0.3)
        df.loc[:n_bars - 1, "low"] = new_prices * (1 - np.abs(blended_returns) * 0.3)

        candle_dict[coin] = df

    return candle_dict


def try_fetch_or_generate(coins: List[str], lookback_hours: int = 72) -> Dict[str, pd.DataFrame]:
    """APIからの取得を試み、失敗したら合成データを生成"""
    candle_data: Dict[str, pd.DataFrame] = {}
    api_available = True

    try:
        test_payload = {"type": "allMids"}
        resp = requests.post(HYPERLIQUID_INFO_URL, json=test_payload, timeout=5)
        resp.raise_for_status()
        print("  ✅ Hyperliquid API 接続成功")
    except Exception as e:
        print(f"  ⚠️ Hyperliquid API 接続不可: {type(e).__name__}")
        print("  📊 合成データモードに切り替え（暗号資産1分足の統計特性をシミュレーション）")
        api_available = False

    if api_available:
        for coin in coins:
            try:
                df = fetch_candles(coin, "1m", lookback_hours)
                if not df.empty and len(df) >= 100:
                    candle_data[coin] = df
                    print(f"   {coin}: ✅ {len(df)} bars (API)")
            except Exception:
                pass
            time.sleep(0.2)

    if len(candle_data) < 3:
        if api_available:
            print("  ⚠️ API データ不足、合成データにフォールバック")
        n_bars = lookback_hours * 60
        common_base_ts = int(time.time() * 1000) - n_bars * 60000
        for i, coin in enumerate(coins):
            df = generate_synthetic_candles(coin, n_bars=n_bars, seed=42 + i, base_timestamp_ms=common_base_ts)
            candle_data[coin] = df
            profile = COIN_PROFILES.get(coin, {})
            print(f"   {coin}: ✅ {len(df)} bars (synthetic, σ_ann={profile.get('vol_ann', 0.8):.0%}, ρ_mr={profile.get('mr_rho', 0.12):.2f})")

        candle_data = add_cross_correlation(candle_data, market_factor_strength=0.35, seed=999)
        print("   🔗 市場ファクター相関追加済 (β=0.35)")

    return candle_data


def fetch_candles(coin: str, interval: str = "1m", lookback_hours: int = 72) -> pd.DataFrame:
    """Hyperliquid APIから1分足ローソク足データを取得"""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - lookback_hours * 3600 * 1000

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }

    resp = requests.post(HYPERLIQUID_INFO_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame()

    rows = []
    for c in data:
        rows.append({
            "timestamp": int(c["t"]),
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"]),
        })

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ======================================================================
# 2. Basis Functions (from the bot)
# ======================================================================


def hermite_basis(x: float, order: int) -> List[float]:
    o = max(0, int(order))
    vals = [1.0]
    if o == 0:
        return vals
    vals.append(x)
    for n in range(1, o):
        vals.append(
            (x * vals[n] - math.sqrt(float(n)) * vals[n - 1])
            / math.sqrt(float(n + 1))
        )
    return vals


def laguerre_basis(x: float, order: int) -> List[float]:
    o = max(0, int(order))
    x = max(0.0, float(x))
    vals = [1.0]
    if o == 0:
        return vals
    vals.append(1.0 - x)
    for n in range(2, o + 1):
        n_f = float(n)
        vals.append(
            (((2.0 * n_f - 1.0) - x) * vals[n - 1] - (n_f - 1.0) * vals[n - 2])
            / n_f
        )
    return vals


def legendre_basis(x: float, order: int) -> List[float]:
    o = max(0, int(order))
    x = max(-1.0, min(1.0, float(x)))
    vals = [1.0]
    if o == 0:
        return vals
    vals.append(x)
    for n in range(2, o + 1):
        n_f = float(n)
        vals.append(
            (((2.0 * n_f - 1.0) * x) * vals[n - 1] - (n_f - 1.0) * vals[n - 2])
            / n_f
        )
    return vals


# ======================================================================
# 3. Backtest Engine
# ======================================================================


@dataclass
class BacktestConfig:
    """バックテストパラメータ"""
    # Core MNVH
    sth_lambda: float = 0.93
    lambda_p: float = 0.4
    lambda_t: float = 0.997

    # Adam
    eta: float = 0.008
    beta1: float = 0.9
    beta2: float = 0.999

    # Tensor dimensions
    hermite_order: int = 3
    laguerre_order: int = 3
    legendre_order: int = 3

    # Penalties
    turnover_penalty: float = 0.002
    cost_penalty: float = 0.001
    weight_uniform_mix: float = 0.05

    # Risk
    max_single_coin_fraction: float = 0.20
    cov_lookback: int = 60
    cov_shrinkage: float = 0.3
    max_portfolio_vol: float = 0.03

    # Position sizing
    target_gross_usd: float = 1000.0

    # Anti-stall
    min_virtual_gross_ratio: float = 0.3
    stall_recovery_gain: float = 0.35

    # Costs
    maker_fee_bps: float = 1.0  # 0.01%
    taker_fee_bps: float = 3.5  # 0.035%
    min_trade_usd: float = 15.0


@dataclass
class BacktestResult:
    """バックテスト結果"""
    config: Dict[str, Any] = field(default_factory=dict)
    total_pnl_usd: float = 0.0
    total_fees_usd: float = 0.0
    net_pnl_usd: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_turnover_usd: float = 0.0
    fee_over_gross_pct: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    n_steps: int = 0
    annualized_return_pct: float = 0.0
    annualized_vol_pct: float = 0.0
    avg_gross_usd: float = 0.0
    pnl_per_step: List[float] = field(default_factory=list)


def zscore_map(data_map: Dict[str, float], coins: List[str]) -> Dict[str, float]:
    vals = [data_map.get(c, 0.0) for c in coins]
    if not vals:
        return {}
    mean_v = sum(vals) / len(vals)
    std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
    denom = max(std_v, 1e-8)
    return {c: (data_map.get(c, 0.0) - mean_v) / denom for c in coins}


def run_backtest(
    candle_data: Dict[str, pd.DataFrame],
    cfg: BacktestConfig,
    verbose: bool = False,
) -> BacktestResult:
    """
    1分足データでMNVH V6戦略のバックテストを実行

    candle_data: {coin: DataFrame with columns [close, volume, ...]}
    """
    coins = list(candle_data.keys())
    if len(coins) < 2:
        return BacktestResult(config=asdict(cfg))

    # Align timestamps across all coins
    all_timestamps = None
    for coin in coins:
        ts_set = set(candle_data[coin]["timestamp"].values)
        if all_timestamps is None:
            all_timestamps = ts_set
        else:
            all_timestamps = all_timestamps & ts_set

    if not all_timestamps or len(all_timestamps) < cfg.cov_lookback + 10:
        print(f"  Not enough aligned data: {len(all_timestamps) if all_timestamps else 0} bars")
        return BacktestResult(config=asdict(cfg))

    sorted_ts = sorted(all_timestamps)
    n_steps = len(sorted_ts)

    # Build price/volume matrices
    price_matrix = np.zeros((n_steps, len(coins)))
    volume_matrix = np.zeros((n_steps, len(coins)))

    for j, coin in enumerate(coins):
        df = candle_data[coin].set_index("timestamp")
        for i, ts in enumerate(sorted_ts):
            price_matrix[i, j] = df.loc[ts, "close"]
            volume_matrix[i, j] = df.loc[ts, "volume"]

    # Use close as proxy for both bid and ask (spread model below)
    # Simulate bid/ask from close with a small spread
    spread_bps = 2.0  # 0.02% half-spread
    bid_matrix = price_matrix * (1.0 - spread_bps / 10000.0)
    ask_matrix = price_matrix * (1.0 + spread_bps / 10000.0)

    # Generate modes
    modes = [
        f"{tb}_{cb}_{ta}_{ca}_{lv}_{ob}"
        for tb in range(cfg.hermite_order + 1)
        for cb in range(cfg.hermite_order + 1)
        for ta in range(cfg.hermite_order + 1)
        for ca in range(cfg.hermite_order + 1)
        for lv in range(cfg.laguerre_order + 1)
        for ob in range(cfg.legendre_order + 1)
        if not (tb == 0 and cb == 0 and ta == 0 and ca == 0 and lv == 0 and ob == 0)
    ]
    n_modes = len(modes)
    mode_indices = {m: i for i, m in enumerate(modes)}

    # Parse mode tuples for fast lookup
    mode_tuples = []
    for mode in modes:
        parts = mode.split("_")
        mode_tuples.append(tuple(int(p) for p in parts))

    # Pre-compute mode index arrays for vectorized Phi construction
    mode_tb_idx = np.array([t[0] for t in mode_tuples], dtype=int)
    mode_cb_idx = np.array([t[1] for t in mode_tuples], dtype=int)
    mode_ta_idx = np.array([t[2] for t in mode_tuples], dtype=int)
    mode_ca_idx = np.array([t[3] for t in mode_tuples], dtype=int)
    mode_lv_idx = np.array([t[4] for t in mode_tuples], dtype=int)
    mode_ob_idx = np.array([t[5] for t in mode_tuples], dtype=int)

    # Initialize state
    epsilon = 1e-8
    w = np.ones(n_modes) / n_modes
    m_adam = np.zeros(n_modes)
    v_adam = np.zeros(n_modes)
    t_step = 0

    n_c = len(coins)
    ewma_p_bid_arr = np.zeros(n_c)
    ewma_t_bid_arr = np.zeros(n_c)
    ewma_p_ask_arr = np.zeros(n_c)
    ewma_t_ask_arr = np.zeros(n_c)

    virtual_q = np.zeros(n_c)
    prev_Phi: Optional[np.ndarray] = None  # (n_modes, n_coins)

    alpha_p = 1.0 - cfg.lambda_p
    alpha_t = 1.0 - cfg.lambda_t

    # Result tracking
    pnl_per_step: List[float] = []
    total_fees = 0.0
    total_turnover = 0.0
    n_trades = 0
    gross_history: List[float] = []
    holdings = np.zeros(len(coins))  # actual holdings in coin units

    price_history_list: List[np.ndarray] = []

    # Warmup period
    warmup = max(cfg.cov_lookback + 2, 10)

    for step in range(1, n_steps):
        prev_prices = price_matrix[step - 1]
        curr_prices = price_matrix[step]
        prev_bids = bid_matrix[step - 1]
        curr_bids = bid_matrix[step]
        prev_asks = ask_matrix[step - 1]
        curr_asks = ask_matrix[step]
        curr_volumes = volume_matrix[step]

        # --- MTM PnL from holdings ---
        price_changes = curr_prices - prev_prices
        mtm_pnl = float(np.sum(holdings * price_changes))

        # --- Compute returns ---
        safe_prev_bid = np.where(prev_bids > 0, prev_bids, 1.0)
        safe_prev_ask = np.where(prev_asks > 0, prev_asks, 1.0)
        r_bid = np.log(curr_bids / safe_prev_bid)
        r_ask = np.log(curr_asks / safe_prev_ask)
        r_mid = 0.5 * (r_bid + r_ask)

        R_bid = float(np.mean(r_bid))
        R_ask = float(np.mean(r_ask))

        # --- EWMA state tracking (vectorized) ---
        s_bid_vec = r_bid - R_bid
        s_ask_vec = r_ask - R_ask
        ewma_p_bid_arr = (1 - alpha_p) * ewma_p_bid_arr + alpha_p * s_bid_vec
        ewma_p_ask_arr = (1 - alpha_p) * ewma_p_ask_arr + alpha_p * s_ask_vec
        ewma_t_bid_arr = (1 - alpha_t) * ewma_t_bid_arr + alpha_t * r_bid
        ewma_t_ask_arr = (1 - alpha_t) * ewma_t_ask_arr + alpha_t * r_ask

        # z-score (vectorized)
        def zscore_vec(arr):
            m = np.mean(arr)
            s = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
            return (arr - m) / max(s, 1e-8)

        u_bid_v = zscore_vec(ewma_p_bid_arr)
        z_bid_v = zscore_vec(ewma_t_bid_arr)
        u_ask_v = zscore_vec(ewma_p_ask_arr)
        z_ask_v = zscore_vec(ewma_t_ask_arr)

        # --- OBI (simulated from volume asymmetry) ---
        vol_mean = float(np.mean(curr_volumes)) if np.mean(curr_volumes) > 0 else 1.0
        obi_arr = np.clip((curr_volumes - vol_mean) / max(vol_mean, 1e-8), -1.0, 1.0)

        # --- Build Phi tensor ---
        n_c = len(coins)
        h_tb = np.zeros((n_c, cfg.hermite_order + 1))
        h_cb = np.zeros((n_c, cfg.hermite_order + 1))
        h_ta = np.zeros((n_c, cfg.hermite_order + 1))
        h_ca = np.zeros((n_c, cfg.hermite_order + 1))
        l_vol = np.zeros((n_c, cfg.laguerre_order + 1))
        p_obi = np.zeros((n_c, cfg.legendre_order + 1))

        for j in range(n_c):
            h_tb[j] = hermite_basis(float(z_bid_v[j]), cfg.hermite_order)
            h_cb[j] = hermite_basis(float(u_bid_v[j]), cfg.hermite_order)
            h_ta[j] = hermite_basis(float(z_ask_v[j]), cfg.hermite_order)
            h_ca[j] = hermite_basis(float(u_ask_v[j]), cfg.hermite_order)
            l_vol[j] = laguerre_basis(float(curr_volumes[j]), cfg.laguerre_order)
            p_obi[j] = legendre_basis(float(obi_arr[j]), cfg.legendre_order)

        # Phi: (n_modes, n_coins) — vectorized via index arrays
        Phi = (
            h_tb[:, mode_tb_idx]
            * h_cb[:, mode_cb_idx]
            * h_ta[:, mode_ta_idx]
            * h_ca[:, mode_ca_idx]
            * l_vol[:, mode_lv_idx]
            * p_obi[:, mode_ob_idx]
        ).T  # (n_modes, n_coins)

        # --- Adam Ascent (vectorized) ---
        if prev_Phi is not None and step > warmup:
            t_step += 1
            n_coins = len(coins)
            # alpha_rewards: (n_modes,) = prev_Phi @ r_mid
            alpha_rewards = prev_Phi @ r_mid
            # mode_turnovers: mean |Phi - prev_Phi| per mode
            mode_turnovers = np.mean(np.abs(Phi - prev_Phi), axis=1)
            # mode_signals: mean |prev_Phi| per mode
            mode_signals = np.mean(np.abs(prev_Phi), axis=1)
            g_t_vec = alpha_rewards - cfg.turnover_penalty * mode_turnovers - cfg.cost_penalty * mode_signals

            m_adam = cfg.beta1 * m_adam + (1 - cfg.beta1) * g_t_vec
            v_adam = cfg.beta2 * v_adam + (1 - cfg.beta2) * (g_t_vec ** 2)

            m_hat = m_adam / (1 - cfg.beta1 ** t_step)
            v_hat = v_adam / (1 - cfg.beta2 ** t_step)
            w = w + cfg.eta * (m_hat / (np.sqrt(v_hat) + epsilon))

            w = np.maximum(0.0, w)
            s = np.sum(w)
            if s > 0:
                w = w / s
            else:
                w = np.ones(n_modes) / n_modes

            if cfg.weight_uniform_mix > 0:
                uniform_w = 1.0 / n_modes
                w = (1.0 - cfg.weight_uniform_mix) * w + cfg.weight_uniform_mix * uniform_w

        prev_Phi = Phi.copy()

        if step <= warmup:
            price_history_list.append(curr_prices.copy())
            pnl_per_step.append(mtm_pnl)
            continue

        # --- Score & Dollar Neutral ---
        scores = np.dot(w, Phi)  # (n_coins,)
        scores_centered = scores - np.mean(scores)
        abs_sum = np.sum(np.abs(scores_centered))

        if abs_sum > 0:
            target_usd_arr = (scores_centered / abs_sum) * cfg.target_gross_usd
        else:
            target_usd_arr = np.zeros(len(coins))

        # --- Single-name cap ---
        if cfg.max_single_coin_fraction < 1.0 and cfg.target_gross_usd > 0:
            cap = cfg.max_single_coin_fraction * cfg.target_gross_usd
            target_usd_arr = np.clip(target_usd_arr, -cap, cap)
            target_usd_arr = target_usd_arr - np.mean(target_usd_arr)
            target_usd_arr = np.clip(target_usd_arr, -cap, cap)
            gross_after = float(np.sum(np.abs(target_usd_arr)))
            if gross_after > cfg.target_gross_usd and gross_after > 0:
                target_usd_arr = target_usd_arr * (cfg.target_gross_usd / gross_after)

        # --- Portfolio Vol Cap ---
        price_history_list.append(curr_prices.copy())
        if len(price_history_list) > cfg.cov_lookback + 1:
            price_history_list = price_history_list[-(cfg.cov_lookback + 1):]

        if len(price_history_list) >= cfg.cov_lookback and cfg.max_portfolio_vol > 0:
            ph = np.array(price_history_list)
            log_rets = np.diff(np.log(np.maximum(ph, 1e-12)), axis=0)
            if log_rets.shape[0] >= 2:
                cov_mat = np.cov(log_rets, rowvar=False)
                if np.isscalar(cov_mat):
                    cov_mat = np.array([[float(cov_mat)]])
                diag = np.diag(np.diag(cov_mat))
                cov_mat = (1.0 - cfg.cov_shrinkage) * cov_mat + cfg.cov_shrinkage * diag
                cov_mat = cov_mat + np.eye(cov_mat.shape[0]) * 1e-10

                exposure_w = target_usd_arr / max(cfg.target_gross_usd, 1e-8)
                port_var = float(np.dot(exposure_w, np.dot(cov_mat, exposure_w)))
                port_vol = math.sqrt(max(port_var, 0.0))
                if port_vol > cfg.max_portfolio_vol:
                    scale = cfg.max_portfolio_vol / max(port_vol, 1e-12)
                    target_usd_arr = target_usd_arr * scale

        # --- Virtual Q update ---
        virtual_q = cfg.sth_lambda * virtual_q + (1.0 - cfg.sth_lambda) * target_usd_arr

        virtual_gross = float(np.sum(np.abs(virtual_q)))
        gross_history.append(virtual_gross)

        # Anti-stall
        target_gross_after = float(np.sum(np.abs(target_usd_arr)))
        if cfg.min_virtual_gross_ratio > 0 and target_gross_after > 0:
            floor_gross = cfg.min_virtual_gross_ratio * target_gross_after
            if virtual_gross < floor_gross and cfg.stall_recovery_gain > 0:
                virtual_q = (
                    (1.0 - cfg.stall_recovery_gain) * virtual_q
                    + cfg.stall_recovery_gain * target_usd_arr
                )

        # --- Execution simulation ---
        step_fees = 0.0
        step_turnover = 0.0

        for j in range(len(coins)):
            target_sz = virtual_q[j] / curr_prices[j] if curr_prices[j] > 0 else 0.0
            diff_sz = target_sz - holdings[j]
            abs_diff_usd = abs(diff_sz) * curr_prices[j]

            if abs_diff_usd < cfg.min_trade_usd:
                continue

            # Execute trade (maker pricing)
            holdings[j] = target_sz
            fee = abs_diff_usd * cfg.maker_fee_bps / 10000.0
            step_fees += fee
            step_turnover += abs_diff_usd
            n_trades += 1

        total_fees += step_fees
        total_turnover += step_turnover

        net_step_pnl = mtm_pnl - step_fees
        pnl_per_step.append(net_step_pnl)

    # --- Compute metrics ---
    pnl_arr = np.array(pnl_per_step)
    cumulative_pnl = np.cumsum(pnl_arr)
    total_pnl = float(np.sum(pnl_arr))
    net_pnl = total_pnl  # fees already deducted

    # Sharpe (annualized, 1-min bars)
    if len(pnl_arr) > 1 and np.std(pnl_arr) > 0:
        sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr) * math.sqrt(525600))  # 365.25*24*60
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cumulative_pnl)
    drawdown = cumulative_pnl - peak
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    max_dd_pct = (max_dd / max(cfg.target_gross_usd, 1.0)) * 100.0

    # Win rate
    positive_steps = int(np.sum(pnl_arr > 0))
    win_rate = positive_steps / max(len(pnl_arr), 1)

    # Annualized
    minutes = len(pnl_arr)
    years = minutes / 525600.0
    ann_return = (total_pnl / max(cfg.target_gross_usd, 1.0)) / max(years, 1e-8) * 100.0
    ann_vol = float(np.std(pnl_arr) * math.sqrt(525600) / max(cfg.target_gross_usd, 1.0)) * 100.0

    avg_gross = float(np.mean(gross_history)) if gross_history else 0.0
    fee_over_gross = (total_fees / max(total_turnover, 1e-8)) * 100.0

    return BacktestResult(
        config=asdict(cfg),
        total_pnl_usd=total_pnl + total_fees,  # gross pnl
        total_fees_usd=total_fees,
        net_pnl_usd=net_pnl,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd_pct,
        total_turnover_usd=total_turnover,
        fee_over_gross_pct=fee_over_gross,
        win_rate=win_rate,
        n_trades=n_trades,
        n_steps=minutes,
        annualized_return_pct=ann_return,
        annualized_vol_pct=ann_vol,
        avg_gross_usd=avg_gross,
        pnl_per_step=[],  # omit for JSON size
    )


# ======================================================================
# 4. Parameter Grid Search
# ======================================================================


def build_param_grid() -> List[BacktestConfig]:
    """パラメータグリッドを構築（計算量を制御）"""
    grid = {
        "sth_lambda": [0.88, 0.93, 0.97],
        "lambda_p": [0.3, 0.5],
        "lambda_t": [0.995, 0.998],
        "eta": [0.005, 0.010],
        "hermite_order": [2, 3],
        "turnover_penalty": [0.0, 0.003],
        "cost_penalty": [0.0, 0.002],
        "weight_uniform_mix": [0.0, 0.05],
        "max_single_coin_fraction": [0.20, 0.35],
        "cov_shrinkage": [0.3, 0.5],
    }

    # Fixed params
    fixed = {
        "beta1": 0.9,
        "beta2": 0.999,
        "laguerre_order": 3,
        "legendre_order": 3,
        "cov_lookback": 60,
        "max_portfolio_vol": 0.03,
        "target_gross_usd": 1000.0,
        "min_virtual_gross_ratio": 0.3,
        "stall_recovery_gain": 0.35,
        "maker_fee_bps": 1.0,
        "taker_fee_bps": 3.5,
        "min_trade_usd": 15.0,
    }

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    configs = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        params.update(fixed)
        configs.append(BacktestConfig(**params))

    return configs


def get_target_coins() -> List[str]:
    """バックテスト対象銘柄（プロファイル定義済み）"""
    return list(COIN_PROFILES.keys())


# ======================================================================
# 5. Main
# ======================================================================


def main():
    print("=" * 70)
    print("🚀 STH-MNVH V6 パラメータ最適化バックテスト")
    print("=" * 70)

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- Select coins & fetch data ---
    target_coins = get_target_coins()
    lookback_hours = 72  # 3 days
    print(f"\n📡 対象銘柄: {target_coins}")
    print(f"📥 {lookback_hours}時間分の1分足データを取得中...")
    candle_data = try_fetch_or_generate(target_coins, lookback_hours)

    if len(candle_data) < 3:
        print("❌ 十分なデータが取得できませんでした")
        return

    print(f"\n✅ {len(candle_data)} 銘柄のデータ取得完了")

    # --- Save raw data ---
    raw_data_path = data_dir / "candle_data_cache.json"
    cache = {}
    for coin, df in candle_data.items():
        df_save = df.drop(columns=["datetime"], errors="ignore")
        cache[coin] = df_save.to_dict(orient="records")
    with open(raw_data_path, "w") as f:
        json.dump(cache, f)
    print(f"💾 ローソク足キャッシュ保存: {raw_data_path}")

    # --- Build parameter grid ---
    print("\n🔧 パラメータグリッド構築中...")
    all_configs = build_param_grid()
    print(f"   合計 {len(all_configs)} 組み合わせ")

    # Limit grid size for tractability
    max_configs = 64
    if len(all_configs) > max_configs:
        print(f"   ⚡ 計算量制限: ランダム {max_configs} 組み合わせを選択")
        rng = np.random.default_rng(42)
        indices = rng.choice(len(all_configs), size=max_configs, replace=False)
        all_configs = [all_configs[i] for i in indices]

    # --- Run backtests ---
    print(f"\n🏃 バックテスト実行中 ({len(all_configs)} 構成)...")
    results: List[BacktestResult] = []

    for i, cfg in enumerate(all_configs):
        label = (
            f"λ={cfg.sth_lambda:.2f} λp={cfg.lambda_p:.1f} λt={cfg.lambda_t:.3f} "
            f"η={cfg.eta:.3f} H={cfg.hermite_order} "
            f"tp={cfg.turnover_penalty:.3f} cp={cfg.cost_penalty:.3f}"
        )
        print(f"   [{i + 1}/{len(all_configs)}] {label}", end=" ", flush=True)

        t0 = time.perf_counter()
        result = run_backtest(candle_data, cfg)
        elapsed = time.perf_counter() - t0

        results.append(result)
        print(
            f"→ Net PnL: ${result.net_pnl_usd:+.2f} | "
            f"Sharpe: {result.sharpe_ratio:.2f} | "
            f"DD: {result.max_drawdown_pct:.1f}% | "
            f"{elapsed:.1f}s"
        )

    # --- Sort by Sharpe ratio ---
    results.sort(key=lambda r: r.sharpe_ratio, reverse=True)

    # --- Print top results ---
    print("\n" + "=" * 70)
    print("📊 上位10構成（Sharpe Ratio順）")
    print("=" * 70)

    for rank, r in enumerate(results[:10], 1):
        c = r.config
        print(f"\n--- Rank #{rank} ---")
        print(f"  Net PnL:      ${r.net_pnl_usd:+.2f}")
        print(f"  Sharpe:       {r.sharpe_ratio:.3f}")
        print(f"  Ann. Return:  {r.annualized_return_pct:+.1f}%")
        print(f"  Ann. Vol:     {r.annualized_vol_pct:.1f}%")
        print(f"  Max DD:       {r.max_drawdown_pct:.2f}%")
        print(f"  Win Rate:     {r.win_rate:.1%}")
        print(f"  Trades:       {r.n_trades}")
        print(f"  Turnover:     ${r.total_turnover_usd:,.0f}")
        print(f"  Fees:         ${r.total_fees_usd:.2f} ({r.fee_over_gross_pct:.2f}% of turnover)")
        print(f"  Avg Gross:    ${r.avg_gross_usd:.0f}")
        print(f"  --- Key Params ---")
        print(f"  sth_lambda={c['sth_lambda']:.2f}  lambda_p={c['lambda_p']:.1f}  lambda_t={c['lambda_t']:.3f}")
        print(f"  eta={c['eta']:.3f}  hermite_order={c['hermite_order']}")
        print(f"  turnover_penalty={c['turnover_penalty']:.3f}  cost_penalty={c['cost_penalty']:.3f}")
        print(f"  weight_uniform_mix={c['weight_uniform_mix']:.2f}  max_single_coin_fraction={c['max_single_coin_fraction']:.2f}")
        print(f"  cov_shrinkage={c['cov_shrinkage']:.1f}")

    # --- Save all results ---
    results_path = data_dir / "backtest_results.json"
    output = {
        "run_timestamp": datetime.now().isoformat(),
        "coins": list(candle_data.keys()),
        "lookback_hours": lookback_hours,
        "n_configs_tested": len(all_configs),
        "results": [asdict(r) for r in results],
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n💾 結果保存: {results_path}")

    # --- Generate recommended config ---
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print("🏆 推奨パラメータ（Sharpe最良構成）")
        print("=" * 70)
        recommended = {
            "sth_lambda": best.config["sth_lambda"],
            "lambda_p": best.config["lambda_p"],
            "lambda_t": best.config["lambda_t"],
            "eta": best.config["eta"],
            "beta1": best.config["beta1"],
            "beta2": best.config["beta2"],
            "hermite_order": best.config["hermite_order"],
            "laguerre_order": best.config["laguerre_order"],
            "legendre_order": best.config["legendre_order"],
            "turnover_penalty": best.config["turnover_penalty"],
            "cost_penalty": best.config["cost_penalty"],
            "weight_uniform_mix": best.config["weight_uniform_mix"],
            "max_single_coin_fraction": best.config["max_single_coin_fraction"],
            "cov_lookback": best.config["cov_lookback"],
            "cov_shrinkage": best.config["cov_shrinkage"],
            "max_portfolio_vol": best.config["max_portfolio_vol"],
            "min_virtual_gross_ratio": best.config["min_virtual_gross_ratio"],
            "stall_recovery_gain": best.config["stall_recovery_gain"],
        }
        rec_path = data_dir / "recommended_config.json"
        with open(rec_path, "w") as f:
            json.dump(recommended, f, indent=2)
        print(json.dumps(recommended, indent=2))
        print(f"\n💾 推奨設定保存: {rec_path}")

    print("\n✅ バックテスト完了")


if __name__ == "__main__":
    main()
