# STH-MNVH V6 金融Bot 完全解析

## 概要

このBotは **Market-Neutral Volatility Harvesting (MNVH)** 戦略を **Hyperliquid DEX** 上の無期限先物（Perpetual）で自動実行するシステムである。理論的基盤は **Stochastic Portfolio Theory (SPT)** の Shannon's Demon（リバランス効果）であり、それを6次元スペクトル多項式テンソルと Adam オプティマイザーで拡張した「V6」アーキテクチャである。

---

## アーキテクチャ全体像

```
[60秒ループ]
  ① 設定＆状態読み込み (config JSON + state JSON)
  ② データ取得 (残高・mid価格・板情報・出来高)
  ③ 6次元特徴量テンソル Φ の構築 (Hermite×4, Laguerre×1, Legendre×1)
  ④ Adam Ascent によるモード重み w の適応学習
  ⑤ スコア算出 → ドルニュートラル制約 → リスク制約適用
  ⑥ virtual_q（仮想在庫）への一次指数収束
  ⑦ 全注文キャンセル → 差分を指値注文で執行
  ⑧ 状態永続化 → 60秒スリープ
```

---

## 理論と実装の対応

### 1. MNVH の核心（論文 Equation 1）

**論文の式:**

```
q_{i,t+1} = λ·q_{i,t} + β·(w_i / p_{i,t})·(R_t − r_{i,t})
```

- `q_{i,t}`: 銘柄 i の保有数量
- `λ`: 在庫減衰率
- `β`: トレード強度
- `R_t = Σ_j w_j r_{j,t}`: ポートフォリオリターン
- `r_{i,t}`: 銘柄 i のリターン
- 直感: 市場平均より下がった銘柄を買い、上がった銘柄を売る（分散収穫）

**Bot実装での対応:**

- `virtual_q[c]` → `q_{i,t}`（ただしUSD建て）
- `sth_lambda` (デフォルト 0.95) → `λ`
- `target_usd_dict[c]` → `β·(w_i/p_i)·(R_t − r_{i,t})` のターゲットポジション
- 更新式: `virtual_q[c] = sth_lambda * virtual_q[c] + (1 - sth_lambda) * target_usd_dict[c]`

論文の「数量ベース更新」をUSD建てのターゲットポジションへの一次指数収束として実装している。

---

### 2. 6次元スペクトル多項式テンソル

Botの最大の革新点。単純な `(R_t − r_{i,t})` シグナルを6軸の直交多項式基底で展開する。

| 軸 | 入力変数 | 基底関数 | 物理的意味 |
|---|---|---|---|
| `time_bid` | bid リターンの時系列 EWMA z-score | **Hermite** | マクロ的な bid トレンド |
| `cross_bid` | bid リターンの断面 EWMA z-score | **Hermite** | 銘柄間の相対的 bid 乖離 |
| `time_ask` | ask リターンの時系列 EWMA z-score | **Hermite** | マクロ的な ask トレンド |
| `cross_ask` | ask リターンの断面 EWMA z-score | **Hermite** | 銘柄間の相対的 ask 乖離 |
| `volume` | 直近出来高 (非負) | **Laguerre** | 流動性・活況度 |
| `OBI` | Order Book Imbalance [-1,1] | **Legendre** | 板の需給バランス |

**テンソル要素:**

```python
Φ[mode][coin] = H_tb(z_bid) × H_cb(u_bid) × H_ta(z_ask) × H_ca(u_ask) × L_lv(volume) × P_ob(obi)
```

**モード数** = `(hermite_order+1)^4 × (laguerre_order+1) × (legendre_order+1) − 1`

デフォルト order=5 の場合: `6^4 × 6 × 6 − 1 = 46,655 モード`

**基底関数の選択根拠:**

- **Hermite多項式**: 標準正規分布の重み関数で直交。z-score化されたリターンに対して最適な基底。実装は三項漸化式 `H_{n+1}(x) = (x·H_n(x) − √n·H_{n-1}(x)) / √(n+1)` で正規化されたprobabilist版。
- **Laguerre多項式**: `[0, ∞)` 上で `e^{-x}` 重みで直交。出来高（非負値）に自然にフィット。三項漸化式 `L_n(x) = ((2n-1-x)·L_{n-1}(x) − (n-1)·L_{n-2}(x)) / n`。
- **Legendre多項式**: `[-1, 1]` 上で一様重みで直交。OBI（-1〜+1にクリップ）にフィット。三項漸化式 `P_n(x) = ((2n-1)·x·P_{n-1}(x) − (n-1)·P_{n-2}(x)) / n`。

---

### 3. Bid/Ask 分離トラッキング

論文のオリジナル MNVH は mid price のみを使うが、このBotは bid と ask を分離追跡する。

**断面 EWMA (`ewma_p`)**: `alpha_p = 1 - lambda_p` (デフォルト 0.5)

```python
s_bid = r_bid − R_bid   # クロスセクション平均からの乖離
ewma_p_bid[c] = (1 − alpha_p) * ewma_p_bid[c] + alpha_p * s_bid
```

速い反応（α=0.5）で、銘柄間の相対的な位置づけを追跡。

**時系列 EWMA (`ewma_t`)**: `alpha_t = 1 - lambda_t` (デフォルト 0.001)

```python
ewma_t_bid[c] = (1 − alpha_t) * ewma_t_bid[c] + alpha_t * r_bid
```

非常に遅い反応（α=0.001）で、長期トレンドを追跡。

bid/ask の非対称な動きを捉えることで、「買い板が強いが売り板が弱い」などの板の微細構造を特徴量に反映する。

---

### 4. Adam Ascent メタコントローラー

46,655 モードの重み `w_dict` をオンライン学習で適応する。

**報酬関数:**

```
g_t = Σ_c Φ_prev[mode][c] × r_mid[c]     (前ステップの特徴量 × 実現リターン)
    − turnover_penalty × mode_turnover    (モード回転ペナルティ)
    − cost_penalty × mode_signal          (コストペナルティ)
```

**Adam 更新:**

```
m = β1·m + (1−β1)·g_t          (一次モーメント)
v = β2·v + (1−β2)·g_t²         (二次モーメント)
m̂ = m / (1 − β1^t)             (バイアス補正)
v̂ = v / (1 − β2^t)             (バイアス補正)
w += η · m̂ / (√v̂ + ε)         (重み更新)
```

**正規化:**

```
w = max(0, w)                   (負の重みをゼロに切り捨て)
w = w / Σw                      (合計1に正規化)
w = (1−mix)·w + mix·(1/N)      (崩壊防止: 均一分布を混合)
```

各モードが「前のステップでどれだけ利益に貢献したか」を追跡し、有用なモードの重みを増やす。多腕バンディット問題のソフトマックス解に類似。

---

### 5. ドルニュートラル制約（Control Law）

```python
scores = Σ_mode w[mode] × Φ[mode][c]           # 各銘柄のスコア
scores_centered = scores − mean(scores)         # 厳密にゼロサム化
target_usd = (scores_centered / Σ|scores|) × target_gross_usd
```

ロング合計 = ショート合計（マーケットニュートラル）を常に強制。

---

### 6. リスク管理

| 制約 | パラメータ | 実装 |
|---|---|---|
| 銘柄集中度制限 | `max_single_coin_fraction` | 各銘柄のグロスを `target_gross_usd × fraction` でクリップ後再中心化 |
| ポートフォリオ Vol 上限 | `max_portfolio_vol` | 共分散行列（縮約推定）で Vol 算出、超過時スケールダウン |
| 共分散縮約 | `cov_shrinkage` (0.2) | `Σ = (1−s)·Σ_sample + s·diag(Σ)` (Ledoit-Wolf 近似) |
| Anti-stall | `min_virtual_gross_ratio`, `stall_recovery_gain` | virtual_gross が閾値未満の場合ターゲットへの収束を加速 |

---

### 7. 執行ロジック

```
差分計算: diff_sz = target_sz − current_sz
最小取引額フィルター: abs(diff_usd) < min_trade_usd → skip

Makerモード（デフォルト）:
  買い → Best Bid に指値（板に並ぶ、手数料節約）
  売り → Best Ask に指値

Takerモード:
  買い → Best Ask + slippage% で即約定
  売り → Best Bid − slippage% で即約定
```

- 毎サイクル冒頭で全既存注文をキャンセル → 新しい注文を発行
- Hyperliquid の EIP-712 署名（msgpack + keccak256 + typed data signing）で認証

---

### 8. 状態永続化 (`hyperliquid-live-state.json`)

| キー | 内容 |
|---|---|
| `virtual_q` | 仮想在庫（USD建て） |
| `historical_Phi` | 前ステップの特徴量テンソル（Adam報酬計算用） |
| `ewma_p_bid/ask` | 断面 EWMA 状態 |
| `ewma_t_bid/ask` | 時系列 EWMA 状態 |
| `m_adam`, `v_adam` | Adam 一次/二次モーメント |
| `w_t` | モード重み (~47K エントリ) |
| `t_step` | Adam ステップ数 |
| `price/bid/ask_history` | 価格履歴（共分散行列計算用） |
| `cov_matrix`, `cov_coins` | 共分散行列キャッシュ |

---

## 論文との差異・拡張点

| 論文（MNVH / Ledger-SPT） | Bot 実装（V6） |
|---|---|
| シグナル: `R_t − r_{i,t}` のみ | 6次元テンソル展開 + Adam 適応重み |
| mid price ベース | bid/ask 分離トラッキング |
| 自己金融制約（κスケーリング） | `virtual_q` + `sth_lambda` による漸近収束（簡略化） |
| PnL 分解台帳（MTM + Trade + Fee） | JSONL トレードログ + Prometheus メトリクス |
| β-ニュートラルヘッジ | ドルニュートラル（scores centered）+ Vol cap |
| バックテスト想定 | ライブ実行（60秒サイクル） |

---

## 潜在的リスク・注意点

1. **モード数爆発** (order=5 → ~47K) — Adam 収束に多数のサイクルが必要
2. **自己金融制約の省略** — 論文の κ スケーリングが未実装、レバレッジ超過リスク
3. **板情報の逐次取得** — `sleep(0.02)` × 銘柄数で遅延蓄積
4. **60秒固定サイクル** — 処理時間超過時にサイクル遅延
5. **状態ファイル肥大** — 47K モードの Phi + weights が JSON で毎サイクル書き出し
