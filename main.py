
import os
import re
import json
import logging
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import requests
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Ledger Question Answering API")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Set this environment variable to the complete questionData URL.
# Keep the signed URL private; do not commit it to source control.
METADATA_URL = os.environ["QUESTION_DATA_URL"]

TIMEOUT = 30
MAX_PAGES = 85
SESSION = requests.Session()


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)


def get_json(url, params=None):
    """Fetch JSON and raise a useful error for invalid responses."""
    response = SESSION.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def find_endpoints(obj):
    """Discover URLs whose keys mention orders, rates, or exports."""
    found = {}

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                name = str(key).lower()

                if isinstance(item, str) and item.startswith("http"):
                    for category in ("orders", "rates", "exports"):
                        if category in name:
                            found[category] = item

                elif isinstance(item, dict):
                    # Some APIs use {"url": "..."} or {"href": "..."}.
                    link = item.get("url") or item.get("href")
                    if isinstance(link, str) and link.startswith("http"):
                        for category in ("orders", "rates", "exports"):
                            if category in name:
                                found[category] = link
                    walk(item)

                elif isinstance(item, list):
                    walk(item)

        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(obj)
    return found


def set_page(url, page):
    """Set the page query parameter without losing other parameters."""
    parts = urlparse(url)
    params = parse_qs(parts.query, keep_blank_values=True)
    params["page"] = [str(page)]
    return urlunparse(
        parts._replace(query=urlencode(params, doseq=True))
    )


def extract_records(payload):
    """
    Extract row-like dictionaries from common API response formats.
    Inspect the real payload if the API uses another structure.
    """
    if isinstance(payload, list):
        if all(isinstance(x, dict) for x in payload):
            return payload
        result = []
        for item in payload:
            result.extend(extract_records(item))
        return result

    if isinstance(payload, dict):
        # Common names for arrays of records.
        for key in ("orders", "data", "results", "items", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return extract_records(value)

        # A single transaction record.
        if any(
            k.lower() in {
                "order_id", "orderid", "amount", "total",
                "revenue", "refund_amount"
            }
            for k in payload
        ):
            return [payload]

    return []


def load_paginated_orders(url):
    """Load pages 1 through 85, stopping if a page is empty."""
    records = []

    for page in range(1, MAX_PAGES + 1):
        payload = get_json(set_page(url, page))
        rows = extract_records(payload)

        if not rows:
            log.info("No records on page %s; stopping.", page)
            break

        records.extend(rows)
        log.info("Loaded page %s: %s rows", page, len(rows))

    return pd.DataFrame(records)


def load_all_data():
    metadata = get_json(METADATA_URL)
    endpoints = find_endpoints(metadata)

    log.info("Discovered endpoint names: %s", list(endpoints))

    if "orders" not in endpoints:
        raise RuntimeError(
            "Could not find the orders URL in questionData. "
            "Inspect the metadata JSON and update find_endpoints()."
        )

    orders = load_paginated_orders(endpoints["orders"])

    rates = (
        get_json(endpoints["rates"])
        if "rates" in endpoints else None
    )
    exports = (
        get_json(endpoints["exports"])
        if "exports" in endpoints else None
    )

    log.info("Orders columns: %s", list(orders.columns))
    if rates is not None:
        log.info("Rates sample: %s", json.dumps(rates)[:1500])
    if exports is not None:
        log.info("Exports sample: %s", json.dumps(exports)[:1500])

    return orders, rates, exports


# Load once at startup. A failed data load prevents false answers.
@app.on_event("startup")
def startup():
    global ORDERS, RATES, EXPORTS
    ORDERS, RATES, EXPORTS = load_all_data()


def normalized_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(df, candidates):
    """Match a logical field to a real column name."""
    lookup = {normalized_name(c): c for c in df.columns}

    for candidate in candidates:
        key = normalized_name(candidate)
        if key in lookup:
            return lookup[key]

    for column in df.columns:
        name = normalized_name(column)
        if any(normalized_name(c) in name for c in candidates):
            return column

    return None


def money(value):
    """Round monetary output to two decimal places."""
    result = Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    return float(result)


def parse_month(question):
    """Return inclusive start and exclusive end for a named month."""
    months = {
        "january": 1, "february": 2, "march": 3,
        "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9,
        "october": 10, "november": 11, "december": 12,
    }

    match = re.search(
        r"\b(" + "|".join(months) + r")\s+(20\d{2})\b",
        question.lower()
    )
    if not match:
        return None

    month = months[match.group(1)]
    year = int(match.group(2))
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthBegin(1)
    return start, end


def analyze(question):
    q = question.lower()
    df = ORDERS.copy()

    if df.empty:
        raise ValueError("The orders dataset is empty.")

    # Identify likely columns. Verify these mappings against actual data.
    date_col = find_column(
        df, ["order_date", "created_at", "date", "transaction_date"]
    )
    region_col = find_column(df, ["region", "sales_region"])
    product_col = find_column(
        df, ["product_name", "product", "item_name"]
    )
    customer_col = find_column(
        df, ["customer_name", "customer", "customer_id"]
    )
    currency_col = find_column(df, ["currency", "currency_code"])

    revenue_col = find_column(
        df, ["revenue", "total_amount", "order_total", "amount"]
    )
    refund_col = find_column(
        df, ["refund_amount", "refunded_amount", "refund_total"]
    )

    # Date filter, e.g. "in March 2026".
    month_range = parse_month(q)
    if month_range:
        if not date_col:
            raise ValueError("No recognizable order date column.")
        dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        start, end = month_range
        df = df.loc[
            (dates >= start.tz_localize("UTC")) &
            (dates < end.tz_localize("UTC"))
        ].copy()

    # Region filter, e.g. "from the North region".
    region_match = re.search(
        r"\b(?:in|from|for)\s+(?:the\s+)?"
        r"(north|south|east|west|central)\s+region\b", q
    )
    if region_match:
        if not region_col:
            raise ValueError("No recognizable region column.")
        df = df[
            df[region_col].astype(str).str.lower()
            == region_match.group(1)
        ]

    # Product filter: exact product names are safer than guessing.
    if product_col:
        for product in df[product_col].dropna().unique():
            if str(product).lower() in q:
                df = df[df[product_col] == product]
                break

    # Customer filter: match a full customer value mentioned in the query.
    if customer_col:
        for customer in df[customer_col].dropna().unique():
            name = str(customer).strip()
            if len(name) >= 3 and name.lower() in q:
                df = df[df[customer_col] == customer]
                break

    is_refund = "refund" in q or "refunded" in q

    if is_refund:
        # Some datasets store refunds in a separate endpoint or table.
        if not refund_col:
            raise ValueError(
                "No refund amount column found in orders. "
                "Inspect the refund/export data and map its schema."
            )
        amount_col = refund_col
    else:
        if not revenue_col:
            raise ValueError(
                "No revenue/amount column found. "
                "Inspect ORDERS.columns and configure the mapping."
            )
        amount_col = revenue_col

    amounts = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)

    # Only convert when the source currency and rate convention are known.
    # Do not assume a rate is USD-based without inspecting the rates payload.
    requested_currency = (
        "EUR" if re.search(r"\b(?:in|to)\s+EUR\b", question, re.I)
        else "INR" if re.search(r"\b(?:in|to)\s+INR\b", question, re.I)
        else "USD"
    )

    if currency_col and requested_currency:
        source_currencies = set(
            df[currency_col].dropna().astype(str).str.upper()
        )
        if source_currencies - {requested_currency}:
            raise ValueError(
                "Currency conversion is needed, but the rates schema "
                "and base-currency convention must be configured first."
            )

    return money(amounts.sum())


@app.get("/health")
def health():
    return {
        "status": "ok",
        "orders_loaded": len(ORDERS),
        "columns": list(ORDERS.columns),
    }


@app.post("/")
def answer_question(body: QuestionRequest):
    try:
        answer = analyze(body.question)
        return {"answer": answer}
    except ValueError as exc:
        log.warning("Could not answer question: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        log.exception("Question processing failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question."
        )


