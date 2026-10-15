"""Synthetic demo dataset generator.

Generates a deterministic e-commerce order book (default 120,000 rows) with
structure that makes every demo question have a *real* answer computed from
real data — nothing in the demo is hard-coded.

Deliberately embedded structure:

* an underlying growth trend plus day-of-week and festive seasonality,
* a genuine regional downturn (West, September 2026) so "why did revenue fall
  in September?" has a correct, discoverable answer,
* two single-day shocks so anomaly detection has something true to find,
* a long-tailed order-value distribution (so the skew insight is real),
* realistic imperfections — a few missing discounts, category spelling
  variants and a handful of duplicated rows — so the data-quality engine
  reports genuine findings rather than a perfect score.

Because the generator is seeded, the same dataset (and therefore the same
insights) is produced on every machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.config import settings
from app.obs.logging import get_logger

log = get_logger(__name__)

DEMO_DATASET_ID = "00000000-0000-4000-8000-000000000001"
DEMO_USER_ID = "00000000-0000-4000-8000-0000000000de"
DEMO_DATASET_NAME = "Demo — E-commerce Sales 2025-2026"

START_DATE = pd.Timestamp("2025-01-01")
END_DATE = pd.Timestamp("2026-09-30")

REGIONS: tuple[str, ...] = ("North", "South", "East", "West", "Central")
REGION_WEIGHTS = np.array([0.26, 0.24, 0.17, 0.21, 0.12])

CHANNELS: tuple[str, ...] = ("Online", "Retail", "Marketplace", "Reseller")
CHANNEL_WEIGHTS = np.array([0.46, 0.22, 0.24, 0.08])

PAYMENT_METHODS: tuple[str, ...] = ("UPI", "Card", "Net Banking", "Cash on Delivery", "Wallet")
PAYMENT_WEIGHTS = np.array([0.42, 0.26, 0.12, 0.14, 0.06])

ORDER_STATUSES: tuple[str, ...] = ("Delivered", "Shipped", "Returned", "Cancelled")
STATUS_WEIGHTS = np.array([0.86, 0.07, 0.05, 0.02])


@dataclass(frozen=True, slots=True)
class Product:
    name: str
    category: str
    base_price: float
    margin: float       # gross margin fraction
    popularity: float


PRODUCTS: tuple[Product, ...] = (
    Product("Aurora Wireless Earbuds", "Electronics", 2499.0, 0.38, 0.11),
    Product("Nimbus Noise-Cancelling Headphones", "Electronics", 7999.0, 0.34, 0.06),
    Product("Vertex 15 Laptop", "Electronics", 68999.0, 0.16, 0.025),
    Product("Pulse Smartwatch 3", "Electronics", 9499.0, 0.31, 0.055),
    Product("Lumen 4K Monitor", "Electronics", 18999.0, 0.22, 0.03),
    Product("Terra Running Shoes", "Apparel", 3299.0, 0.47, 0.09),
    Product("Breeze Cotton Shirt", "Apparel", 1499.0, 0.55, 0.10),
    Product("Summit Trek Jacket", "Apparel", 4599.0, 0.44, 0.045),
    Product("Harbour Canvas Backpack", "Accessories", 2199.0, 0.52, 0.07),
    Product("Orbit Leather Wallet", "Accessories", 1299.0, 0.58, 0.065),
    Product("Kettle Pro Electric Kettle", "Home & Kitchen", 2899.0, 0.36, 0.06),
    Product("Chef's Edge Knife Set", "Home & Kitchen", 3999.0, 0.41, 0.04),
    Product("Calm Weighted Blanket", "Home & Kitchen", 4299.0, 0.45, 0.035),
    Product("Verve Protein Powder 1kg", "Health", 2799.0, 0.49, 0.075),
    Product("Zen Yoga Mat", "Health", 1199.0, 0.56, 0.055),
    Product("Atlas Hardcover Notebook", "Stationery", 449.0, 0.62, 0.05),
    Product("Inkwell Gel Pen Pack", "Stationery", 249.0, 0.65, 0.045),
    Product("Solstice Desk Lamp", "Home & Kitchen", 1899.0, 0.43, 0.04),
)


def _daily_order_curve(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Expected order count per day, with trend, seasonality and shocks."""
    n = len(dates)
    day_index = np.arange(n, dtype="float64")

    # ~28% annual growth, compounded daily.
    trend = 150.0 * (1.0 + 0.00068) ** day_index

    # Day-of-week: weekends busier.
    dow = dates.dayofweek.to_numpy()
    dow_factor = np.array([0.92, 0.95, 0.98, 1.02, 1.12, 1.26, 1.18])[dow]

    # Month-of-year: Indian festive season (Oct-Nov) and year-end peak.
    month = dates.month.to_numpy()
    month_factor = np.array(
        # J     F     M     A     M     J     J     A     S     O     N     D
        [0.88, 0.86, 0.95, 0.97, 1.00, 0.94, 0.96, 1.03, 0.99, 1.34, 1.28, 1.15]
    )[month - 1]

    # Payday bump around the 1st and the 15th.
    day = dates.day.to_numpy()
    payday = 1.0 + 0.10 * np.exp(-((day - 1) ** 2) / 8.0) + 0.07 * np.exp(
        -((day - 15) ** 2) / 8.0
    )

    expected = trend * dow_factor * month_factor * payday

    # --- deliberate shocks, so anomaly detection has real events to find ----
    shock = np.ones(n)
    iso = dates.strftime("%Y-%m-%d").to_numpy()

    # A single-day outage: 17 Sep 2026.
    shock[iso == "2026-09-17"] *= 0.34
    # A flash-sale spike: 11 Jul 2026.
    shock[iso == "2026-07-11"] *= 2.45
    # Warehouse backlog clears over three days in Mar 2026.
    for d in ("2026-03-04", "2026-03-05"):
        shock[iso == d] *= 0.62

    noise = rng.normal(1.0, 0.085, size=n)
    return np.maximum(expected * shock * noise, 8.0)


def generate_demo_dataframe(
    *, rows: int | None = None, seed: int | None = None
) -> pd.DataFrame:
    """Build the demo order book. Deterministic for a given (rows, seed)."""
    target_rows = int(rows or settings.demo_rows)
    target_rows = max(1_000, min(target_rows, settings.max_rows))
    rng = np.random.default_rng(seed if seed is not None else settings.demo_seed)

    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    curve = _daily_order_curve(dates, rng)
    # Scale the curve so the total lands on the requested row count.
    counts = np.maximum(
        np.round(curve * (target_rows / curve.sum())).astype("int64"), 1
    )

    # Correct any rounding drift against the requested total.
    drift = target_rows - int(counts.sum())
    if drift != 0:
        order = np.argsort(-counts)
        for i in range(abs(drift)):
            idx = order[i % len(order)]
            counts[idx] += 1 if drift > 0 else -1
        counts = np.maximum(counts, 1)

    order_dates = np.repeat(dates.to_numpy(), counts)
    n = order_dates.size
    date_series = pd.Series(pd.to_datetime(order_dates))

    # Spread orders through business hours so an hourly grain is meaningful.
    minutes = rng.integers(8 * 60, 23 * 60, size=n)
    timestamps = date_series + pd.to_timedelta(minutes, unit="m")

    # ---- products -------------------------------------------------------
    popularity = np.array([p.popularity for p in PRODUCTS], dtype="float64")
    popularity = popularity / popularity.sum()
    product_idx = rng.choice(len(PRODUCTS), size=n, p=popularity)
    product_names = np.array([p.name for p in PRODUCTS])[product_idx]
    categories = np.array([p.category for p in PRODUCTS])[product_idx]
    base_prices = np.array([p.base_price for p in PRODUCTS])[product_idx]
    margins = np.array([p.margin for p in PRODUCTS])[product_idx]

    # ---- regions, with a real West-region downturn in Sep 2026 ----------
    region_idx = rng.choice(len(REGIONS), size=n, p=REGION_WEIGHTS / REGION_WEIGHTS.sum())
    regions = np.array(REGIONS)[region_idx]

    # Lose ~45% of West orders in September 2026 (a regional fulfilment
    # problem). The orders are *removed*, not moved to other regions: moving
    # them would leave total revenue unchanged and there would be no fall to
    # explain.
    month_key = date_series.dt.strftime("%Y-%m").to_numpy()
    lost = (regions == "West") & (month_key == "2026-09") & (rng.random(n) < 0.45)

    # ---- customers: a repeat-purchase pool ------------------------------
    customer_pool = max(2_000, n // 9)
    # Zipf-ish skew so a minority of customers drive a majority of orders.
    customer_ids = (
        rng.zipf(1.35, size=n) % customer_pool
    ) + 1
    customer_codes = np.char.add("CUST-", customer_ids.astype("U7"))

    # ---- quantities and pricing -----------------------------------------
    quantity = rng.choice(
        [1, 2, 3, 4, 5, 8, 12], size=n, p=[0.52, 0.22, 0.11, 0.06, 0.05, 0.025, 0.015]
    )
    # Bulk B2B orders: a thin tail that makes the value distribution skewed.
    bulk = rng.random(n) < 0.004
    quantity = quantity.astype("int64")
    quantity[bulk] = rng.integers(25, 140, size=int(bulk.sum()))

    price_jitter = rng.normal(1.0, 0.045, size=n)
    # Mild price inflation across the window.
    days_elapsed = (date_series - START_DATE).dt.days.to_numpy()
    inflation = 1.0 + 0.00013 * days_elapsed
    unit_price = np.round(base_prices * price_jitter * inflation, 2)

    discount = np.zeros(n)
    has_discount = rng.random(n) < 0.46
    discount[has_discount] = rng.choice(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        size=int(has_discount.sum()),
        p=[0.26, 0.26, 0.19, 0.15, 0.09, 0.05],
    )
    # Deeper festive discounting.
    festive = np.isin(date_series.dt.month.to_numpy(), [10, 11])
    discount[festive] = np.minimum(discount[festive] + 0.05, 0.42)

    revenue = np.round(quantity * unit_price * (1.0 - discount), 2)
    unit_cost = np.round(base_prices * (1.0 - margins) * inflation, 2)
    cost = np.round(quantity * unit_cost, 2)
    profit = np.round(revenue - cost, 2)

    shipping_days = np.clip(rng.normal(3.4, 1.5, size=n), 0.5, 14).round(1)
    satisfaction = np.clip(rng.normal(4.25, 0.72, size=n), 1.0, 5.0).round(1)

    channels = np.array(CHANNELS)[
        rng.choice(len(CHANNELS), size=n, p=CHANNEL_WEIGHTS / CHANNEL_WEIGHTS.sum())
    ]
    payments = np.array(PAYMENT_METHODS)[
        rng.choice(len(PAYMENT_METHODS), size=n, p=PAYMENT_WEIGHTS / PAYMENT_WEIGHTS.sum())
    ]
    statuses = np.array(ORDER_STATUSES)[
        rng.choice(len(ORDER_STATUSES), size=n, p=STATUS_WEIGHTS / STATUS_WEIGHTS.sum())
    ]

    order_ids = np.char.add("ORD-", np.arange(100_001, 100_001 + n).astype("U9"))

    df = pd.DataFrame(
        {
            "order_id": order_ids,
            "order_date": timestamps,
            "customer_id": customer_codes,
            "product": product_names,
            "category": categories,
            "region": regions,
            "channel": channels,
            "payment_method": payments,
            "order_status": statuses,
            "quantity": quantity,
            "unit_price": unit_price,
            "discount": np.round(discount, 4),
            "revenue": revenue,
            "cost": cost,
            "profit": profit,
            "shipping_days": shipping_days,
            "satisfaction_score": satisfaction,
        }
    )

    df = df.loc[~lost].reset_index(drop=True)
    df = _inject_realistic_imperfections(df, rng)
    df = df.sort_values("order_date", kind="mergesort").reset_index(drop=True)

    log.info(
        "demo_dataset_generated",
        extra={
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "revenue_total": float(df["revenue"].sum()),
        },
    )
    return df


def _inject_realistic_imperfections(
    df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Add the flaws a real export has, so quality findings are genuine."""
    n = len(df)

    # 1. Missing discounts (field not captured for some legacy orders).
    missing = rng.random(n) < 0.012
    df.loc[missing, "discount"] = np.nan

    # 2. Missing satisfaction scores (survey not returned).
    no_survey = rng.random(n) < 0.21
    df.loc[no_survey, "satisfaction_score"] = np.nan

    # 3. Category spelling variants from two upstream systems.
    variant_mask = rng.random(n) < 0.018
    variants = {
        "Electronics": "electronics",
        "Apparel": "APPAREL",
        "Home & Kitchen": "Home and Kitchen",
        "Accessories": " Accessories ",
        "Health": "health",
    }
    categories = df["category"].to_numpy(dtype=object).copy()
    for idx in np.flatnonzero(variant_mask):
        original = categories[idx]
        if original in variants:
            categories[idx] = variants[original]
    df["category"] = categories

    # 4. A handful of exact duplicate rows (double-submitted orders).
    duplicate_count = max(12, n // 4000)
    duplicate_idx = rng.choice(n, size=duplicate_count, replace=False)
    duplicates = df.iloc[duplicate_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # 5. A few returns booked as negative quantities (sign-convention error).
    negatives = rng.choice(len(df), size=max(6, len(df) // 12000), replace=False)
    df.loc[negatives, "quantity"] = -df.loc[negatives, "quantity"].abs()
    df.loc[negatives, "revenue"] = -df.loc[negatives, "revenue"].abs()
    df.loc[negatives, "profit"] = (
        df.loc[negatives, "revenue"] - df.loc[negatives, "cost"]
    )

    return df


def demo_csv_bytes(*, rows: int | None = None, seed: int | None = None) -> bytes:
    """The demo dataset as a CSV, for the 'download the sample file' action."""
    df = generate_demo_dataframe(rows=rows, seed=seed)
    return df.to_csv(index=False).encode("utf-8")


DEMO_QUESTIONS: tuple[str, ...] = (
    "What was total revenue?",
    "Which region generated the most profit?",
    "Why did revenue fall in September?",
    "Which product is most profitable?",
    "Show monthly revenue.",
    "Find anomalies.",
    "Forecast the next 3 months.",
    "Compare 2025 and 2026.",
    "Top 10 products by revenue",
    "What is the average order value?",
)
