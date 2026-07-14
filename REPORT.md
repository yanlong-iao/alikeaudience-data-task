# Data Test — Task 1: Exploration, Enrichment & Findings

**Input:** `alikeaudience_data_test.csv` — 100,000 rows × 5 columns (`user_id`, `timestamp`, `lat_long`, `ip_address`, `user_agent`)

**Deliverables:** `pipeline.py` (reusable enrichment pipeline), `enriched_data.csv` (100k rows × 27 columns), `summary_stats.json`, `analytics_charts.png`, `geo_scatter.png`

---

## 1. Data quality audit

| Check | Result |
|---|---|
| Rows / unique users | 100,000 / 100,000 — **exactly one event per user** |
| Exact duplicate rows | 0 |
| Null values | `user_agent`: 1,909 (1.9%); all other columns complete |
| Timestamp validity | 100% parseable ISO-8601 UTC; range 2022-12-01 → 2022-12-31 |
| **Coordinate order** | ⚠️ `lat_long` is actually **`longitude latitude`** — 100% of rows have \|col0\| > 90, impossible for latitude. Pipeline corrects and documents this. |
| IP validity | 100% syntactically valid; 50,004 IPv4 / 49,996 IPv6 |
| **IP routability** | ⚠️ 4,192 IPs (4.2%) are private/reserved/non-routable — cannot originate from real public internet traffic |

Two structural observations worth flagging to the data producer:

1. **The column name `lat_long` is misleading** (it's lon-lat). Silent consumers of this feed would place every user in the wrong hemisphere.
2. **The data shows signs of synthetic generation**: non-routable IPs, a perfectly ~50/50 IPv4/IPv6 split, and a device mix (Motorola 14%, T-Mobile REVVL, Boost Celero — US-carrier models) inconsistent with the Asian geography. The strongest signal: only **13.2% of the IPv6 addresses fall inside `2000::/3`** — the block where effectively all real-world global-unicast IPv6 lives — meaning the IPs were sampled uniformly at random, not captured from traffic. In production I'd cross-validate IP-geolocation against the GPS coordinates; a high mismatch rate confirms the fields were generated independently.

## 2. Enrichment (all offline — no external APIs)

From 5 raw columns the pipeline derives 22 new features:

- **Geo** — corrected `latitude`/`longitude`; `country` + `nearest_city` via a vectorized haversine nearest-neighbor lookup over a ~190-city gazetteer of East/SE Asia; `km_to_city`; `geo_confidence` (91.8% high, <100 km from a known city). *Production swap-in: Natural Earth point-in-polygon or MaxMind GeoIP2 — same interface.*
- **Time** — `date`, `hour_utc`, `dow`, `is_weekend`, plus `local_hour` and `local_part_of_day` using longitude-approximated timezone (production: tz polygon lookup).
- **Device** — regex UA parser (cached per unique UA — 16.9k unique strings, not 100k parses): `os_family`/`os_version`, `device_brand`/`device_model` (~40 brand rules), `form_factor`, `browser`/`browser_version`, `is_webview`.
- **Network** — `ip_version`, `ip_routable`, `ip_scope` (public/private/reserved).

## 3. Key findings

**Geography.** Traffic is entirely East/Southeast Asia. Japan (30.2%) and Indonesia (30.0%) dominate, followed by South Korea (9.8%), Vietnam (7.2%), Thailand (6.6%), Philippines and Taiwan (~5.1% each), Malaysia (3.9%), Singapore (1.3%).

**Volume anomaly.** Baseline is ~2,100 events/day, but Dec 6 spikes to **12,987 (~6×)**, elevated through Dec 9, and volume ramps again Dec 28–31 (~4,200). Consistent with campaign bursts or batched backfills — worth confirming with the ingestion team before using this window for modeling.

**Daily rhythm.** Local-time activity peaks 15:00–18:00 and troughs 03:00–04:00 — a plausible human diurnal curve, which validates the timezone approximation. Weekend share is 23.4% (below the 28.6% calendar expectation → weekday-skewed usage).

**Devices.** 96.2% Android, 3.7% iOS. Samsung leads (56%), then Motorola (13.8%), Google, Apple, LG. **91.6% of traffic comes from in-app WebViews**, not standalone browsers — this is SDK/in-app traffic, so UA-based capabilities detection should target WebView, and the 1.9% missing UAs likely come from native HTTP clients that set no UA.

**Audience-building implications** (the business lens): country × device-tier × part-of-day are immediately usable segmentation dimensions; e.g. "Indonesia, budget-Android, evening" vs "Japan, Samsung flagship, afternoon" cohorts fall out of the enriched table with one `groupby`.

## 4. What I'd build next (with more time / real infra)

- **IP↔GPS cross-validation** with MaxMind GeoIP2 as a per-row trust score; flag rows where IP country ≠ GPS country (fraud / VPN / synthetic-data signal).
- **Precise geo**: point-in-polygon country assignment (Natural Earth) + tz-polygon local time; H3 hexagon indexing for spatial aggregation and home-location inference.
- **Device-price-tier mapping** (model → launch price band) — a strong proxy for purchasing power in lookalike audience models.
- **Pipeline hardening**: schema contract (e.g. pydantic/Great Expectations) with the coordinate-order and IP-routability checks as CI gates; partition output by date as Parquet; incremental runs.
- With multi-event-per-user data: sessionization, home/work location inference, impossible-travel detection, and behavioral embeddings. (Not possible here — each user appears exactly once.)

## Reproduce

```bash
python3 pipeline.py alikeaudience_data_test.csv ./output
```

Requires only `pandas`, `numpy`, `matplotlib` (stdlib for IP/UA parsing). Runs in ~1 min for 100k rows.
