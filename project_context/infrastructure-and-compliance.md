# Infrastructure & Compliance

## Last Updated: 2026-03-21

---

## Broker Stack

### Architecture: Split Data / Execution

| Role | Broker | Why | Key Specs |
|---|---|---|---|
| **Data feed** | Fyers | Only retail TBT (tick-by-tick), 50-level DOM, <10ms claimed | TBT WS: `wss://rtsocket-api.fyers.in/versova` |
| **Primary execution** | Dhan | Most SEBI-compliant, 200-level depth, 5yr historical | 25 OPS, static IP live since early 2025 |
| **Secondary execution** | Upstox | Highest OPS (50), HFT endpoint | `api-hft.upstox.com` for orders |
| **Backup** | Angel One | Explicit self-hosted exemption, 5 static IPs/key | 20 OPS (→10 post-Aug 2025) |

### Latency Budget (Realistic)

```
Exchange tick → Broker TAP:     1-50ms
Broker → Your WS:              50-800ms (1-sec snapshot batching for most)
Signal computation:             1-50ms
Order API call:                 1-30ms
Broker API ack:                 50-400ms
Broker OMS → Exchange:          100-3,000ms (ASYNC — the killer)
─────────────────────────────────────────
Total end-to-end:               500ms - 4,000ms
```

**Implication:** Minimum viable signal half-life is >30 seconds. All 7 strategies operate at 5-min+ frequency — latency is <1% of hold time. This is fine.

### Broker API Comparison (Key Metrics)

| Metric | Zerodha | Upstox | Dhan | Fyers | Angel One |
|---|---|---|---|---|---|
| Order rate | 10 OPS | **50 OPS** | 25 OPS | 10 OPS | 20→10 OPS |
| WS instruments | 9,000 | 100 | **25,000** | ? | ? |
| Depth levels | 5 | 5 | **20/200** | **50 (TBT)** | 5 |
| TBT feed | No | No | Claimed | **Yes** | No |
| Tick rate | 1/sec | Event | Event | **1000+/sec** | Event |
| Intraday hist | Since ~2015 | Multi-year | **5 years** | Multi-year | Multi-year |
| Expired options | No | No | **Yes** | No | No |
| API cost | ₹500/mo | Free | Free (data ₹499) | Free | Free |
| Static IP UI | Late/unclear | Unclear | **Live** | **Live** | **Live** |
| Self-hosted OK? | Lobbying | Unknown | **Yes** | **Yes** | **Explicit yes** |

### SEBI Compliance (Hard Deadline: 01-Apr-2026)

**Your category:** Tech-savvy retail investor, ≤10 OPS per exchange/segment.

| Requirement | How | Status |
|---|---|---|
| All API orders = algo | Automatic — broker tags with NNF ID 444444444444, algo ID 99999 | Broker handles |
| Static IP | AWS Elastic IP in ap-south-1 or ISP static IP | **DO THIS WEEK** |
| OAuth + 2FA | Already standard on all brokers | Verify per broker |
| Daily session logout | Code: `atexit.register(logout)` | **Code change needed** |
| Limit orders only | No market orders for algo orders (DOM §11 Sr.5) | **Code change needed** |
| One API key for non-registered algos | Consolidate | Configure per broker |
| Order routing server in India | AWS ap-south-1 | ✅ |
| 5yr audit trail | Keep own logs + broker maintains | Build logging from Day 1 |

**"Algos must run on broker servers" ambiguity — RESOLVED:**
- NSE FAQ (Nov 2025): "Tech-savvy clients can host locally using static IP"
- Angel One: explicit exemption for self-hosted with static IP
- Dhan: operationally allows it
- **Self-hosted Python → broker API via static IP = compliant**

### If You Exceed 10 OPS (Scaling Concern)

At ₹5 Cr with 7 strategies, you might approach 10 OPS during peak activity. If so:
- Register algos individually with exchange (via broker)
- Prepare strategy writeup + RMS writeup per algo
- Classify as White Box (execution) where possible — avoids RA registration
- Timeline: T+7 to T+10 working days for exchange approval
- Start this process at ₹1 Cr scale, don't wait for ₹5 Cr

### Data Infrastructure

| Data | Source | Cost | Depth |
|---|---|---|---|
| 1-min OHLCV (NIFTY, BankNifty) | Dhan API | Free (with account) | 5 years |
| 1-min OHLCV (top 200 stocks) | Dhan API | Free | 5 years |
| Expired options chain | Dhan API | Free | 5 years |
| Tick-by-tick live | Fyers TBT WS | Free (with account) | Live only — record yourself |
| Daily OHLCV (all NSE) | NSE website / jugaad-data | Free | 20+ years |
| India VIX | NSE website | Free | Daily since 2009 |
| Corporate actions / earnings | NSE / Trendlyne | Free/₹ | Multi-year |
| L1 1-sec snapshots | TrueData / GDFL | ₹5-15K/yr | 20 days tick, multi-year OHLCV |

**Critical:** Start recording Fyers TBT to S3/Parquet from Day 1. There is no affordable source of historical tick-level options data. You must accumulate it yourself for future strategy development.

### Monthly Infrastructure Cost (Estimated)

| Component | Cost |
|---|---|
| AWS (t3.medium + EIP + S3) | ₹3,000-5,000 |
| Dhan data API | ₹499 (or free with 25+ F&O trades/mo) |
| Fyers API | Free |
| TrueData (supplementary) | ₹500-1,500 |
| Static IP (if ISP) | ₹200-500 |
| **Total** | **₹4,000-7,500/month** |

At ₹50L capital, infrastructure is 0.1-0.2% monthly — negligible.

### Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Language | Python (Numba/Cython for hot paths) | Team already has this |
| Backtesting | Custom event-driven (existing pipeline) | Reuse equity pipeline infra |
| Execution | OpenAlgo (open-source, supports Dhan/Fyers/Angel) or custom | Evaluate Week 1 |
| Data storage | Parquet on S3 + DuckDB for queries | Fast columnar access |
| Monitoring | Grafana + Prometheus + Telegram bot | Build Week 1 |
| Multi-broker | Custom router: Fyers data → Dhan/Upstox orders | Build Week 2 |
| Concurrency | asyncio + multiprocessing for strategy isolation | Team strength |
