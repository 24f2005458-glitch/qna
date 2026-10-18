
import os
import re
import logging
from decimal import Decimal, ROUND_HALF_UP
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import requests
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

METADATA_URL = os.environ["QUESTION_DATA_URL"]

MAX_PAGES = 85
TIMEOUT = 30

# Set this to match your rates endpoint:
#
# "units_per_usd":
#   rate means units of that currency per 1 USD.
#   Example: USD=1, EUR=0.92, INR=83.
#
# "usd_per_unit":
#   rate means USD per 1 unit of that currency.
#
# Verify the actual rates response before selecting this.
RATE_MODE = os.getenv("RATE_MODE", "units_per_usd")

# Used only if the orders dataset has no currency column.
# Change this if the API documentation establishes another
# default currency. Do not guess if currencies are mixed.
DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "USD").upper()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Ledger Question Answering API")
session = requests.Session()


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)


# ---------------------------------------------------------
# HTTP and endpoint discovery
# ---------------------------------------------------------

def get_json(url, params=None):
    response = session.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def discover_endpoints(obj):
    found = {}

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                key_lower = str(key).lower()

                if isinstance(item, str) and item.startswith("http"):
                    for name in ("orders", "rates", "exports"):
                        if name in key_lower:
                            found[name] = item

                elif isinstance(item, dict):
                    link = item.get("url") or item.get("href")

                    if isinstance(link, str) and link.startswith("http"):
                        for name in ("orders", "rates", "exports"):
                            if name in key_lower:
                                found[name] = link

                    walk(item)

                elif isinstance(item, list):
                    walk(item)

        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(obj)
    return found


def add_page_parameter(url, page):
    parts = urlparse(url)
    params = parse_qs(parts.query, keep_blank_values=True)
    params["page"] = [str(page)]

    return urlunparse(
        parts._replace(query=urlencode(params, doseq=True))
    )


def extract_records(payload):
    """Extract records from common API response structures."""

    if isinstance(payload, list):
        if all(isinstance(x, dict) for x in payload):
            return payload

        records = []
        for item in payload:
            records.extend(extract_records(item))
        return records

    if isinstance(payload, dict):
        for key in (
            "orders", "data", "results", "items", "records",
            "rates", "exports", "refunds"
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return extract_records(value)

        # A single record rather than a list.
        if any(
            key.lower() in {
                "order_id", "orderid", "amount", "total",
                "revenue", "refund_amount"
            }
            for key in payload
        ):
            return [payload]

    return []


def load_orders(url):
    records = []

    for page in range(1, MAX_PAGES + 1):
        payload = get_json(add_page_parameter(url, page))
        rows = extract_records(payload)

        if not rows:
            log.info("No records on page %d; stopping.", page)
            break

        records.extend(rows)
        log.info("Page %d: loaded %d records", page, len(rows))

    if not records:
        raise RuntimeError("No order records were found.")

    return pd.DataFrame(records)


# ---------------------------------------------------------
# Column mapping
# ---------------------------------------------------------

ALIASES = {
    "date": [
        "order_date", "created_at", "date", "transaction_date"
    ],
    "region": ["region", "sales_region", "order_region"],
    "status": [
        "status", "order_status", "payment_status",
        "payment_state", "financial_status"
    ],
    "currency": [
        "currency", "currency_code", "order_currency"
    ],
    "amount": [
        "total_amount", "order_total", "order_value",
        "total_price", "grand_total", "amount", "revenue"
    ],
    "refund": [
        "refund_amount", "refunded_amount", "refund_total"
    ],
    "product": ["product_name", "product", "item_name"],
    "customer": ["customer_name", "customer", "customer_id"],
}


def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(df, candidates):
    lookup = {norm(col): col for col in df.columns}

    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]

    # Allow common variations such as order_total_usd.
    for col in df.columns:
        normalized = norm(col)
        for candidate in candidates:
            candidate_normalized = norm(candidate)
            if (
                len(candidate_normalized) >= 5
                and candidate_normalized in normalized
            ):
                return col

    return None


def get_column(df, logical_name):
    return find_column(df, ALIASES[logical_name])


# ---------------------------------------------------------
# Currency rates
# ---------------------------------------------------------

def collect_dict_records(payload):
    """Recursively collect dictionaries from nested rate JSON."""

    records = []

    if isinstance(payload, dict):
        records.append(payload)
        for value in payload.values():
            records.extend(collect_dict_records(value))

    elif isinstance(payload, list):
        for value in payload:
            records.extend(collect_dict_records(value))

    return records


def build_rates(payload):
    """
    Return rates keyed by currency code.

    This supports common row-based formats, but the actual
    rates schema must be checked. Ambiguous rate fields should
    be mapped explicitly rather than guessed.
    """
    result = {"USD": Decimal("1")}

    for record in collect_dict_records(payload):
        code = (
            record.get("currency")
            or record.get("currency_code")
            or record.get("code")
        )

        rate = (
            record.get("rate")
            if "rate" in record
            else record.get("exchange_rate")
        )

        if code is None or rate is None:
            continue

        code = str(code).strip().upper()

        if not re.fullmatch(r"[A-Z]{3}", code):
            continue

        try:
            rate = Decimal(str(rate))
        except Exception:
            continue

        if rate > 0:
            result[code] = rate

    return result


def to_usd(amount, currency, rates):
    amount = Decimal(str(amount))
    currency = str(currency).upper().strip()

    if currency == "USD":
        return amount

    if currency not in rates:
        raise ValueError(
            f"No exchange rate found for {currency}. "
            "Check the rates endpoint and build_rates()."
        )

    rate = rates[currency]

    if RATE_MODE == "units_per_usd":
        return amount / rate

    if RATE_MODE == "usd_per_unit":
        return amount * rate

    raise ValueError(
        "RATE_MODE must be units_per_usd or usd_per_unit."
    )


# ---------------------------------------------------------
# Question interpretation
# ---------------------------------------------------------

def requested_currency(question):
    q = question.lower()

    if re.search(r"\b(eur|euros?|€)\b", q):
        return "EUR"

    if re.search(r"\b(inr|rupees?|₹)\b", q):
        return "INR"

    # Default for questions asking for US dollars, USD,
    # or order value without another currency.
    return "USD"


def requested_operation(question):
    q = question.lower()

    if re.search(
        r"\b(average|averages|on average|mean|per order|"
        r"average order value)\b", q
    ):
        return "average"

    if re.search(r"\b(count|number of|how many orders)\b", q):
        return "count"

    if re.search(r"\b(maximum|highest|largest|most expensive)\b", q):
        return "max"

    if re.search(r"\b(minimum|lowest|smallest|cheapest)\b", q):
        return "min"

    return "sum"


def requested_status(question):
    q = question.lower()

    if re.search(r"\bpaid\b", q):
        return "paid"

    if re.search(r"\b(unpaid|not paid|pending payment)\b", q):
        return "unpaid"

    if re.search(r"\b(cancelled|canceled)\b", q):
        return "cancelled"

    return None


def requested_region(question):
    match = re.search(
        r"\b(?:in|from|for)\s+(?:the\s+)?"
        r"(north|south|east|west|central)\s+region\b",
        question,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).title()

    # "West Order" / "West orders"
    match = re.search(
        r"\b(north|south|east|west|central)\s+orders?\b",
        question,
        re.IGNORECASE,
    )

    return match.group(1).title() if match else None


def requested_month(question):
    months = {
        "january": 1, "february": 2, "march": 3,
        "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9,
        "october": 10, "november": 11, "december": 12,
    }

    match = re.search(
        r"\b(" + "|".join(months) + r")\s+(20\d{2})\b",
        question.lower(),
    )

    if not match:
        return None

    start = pd.Timestamp(
        year=int(match.group(2)),
        month=months[match.group(1)],
        day=1,
    )

    return start, start + pd.offsets.MonthBegin(1)


# ---------------------------------------------------------
# Analysis engine
# ---------------------------------------------------------

def analyze(question):
    q = question.lower()
    df = ORDERS.copy()

    operation = requested_operation(question)
    target_currency = requested_currency(question)

    # Date filter.
    date_range = requested_month(question)

    if date_range:
        date_col = get_column(df, "date")

        if not date_col:
            raise ValueError(
                "The dataset has no recognized date column."
            )

        dates = pd.to_datetime(
            df[date_col], errors="coerce", utc=True
        )

        start, end = date_range
        start = start.tz_localize("UTC")
        end = end.tz_localize("UTC")

        df = df.loc[
            (dates >= start) & (dates < end)
        ].copy()

    # Region filter.
    region = requested_region(question)

    if region:
        region_col = get_column(df, "region")

        if not region_col:
            raise ValueError(
                "A region was requested, but no region column "
                "was found."
            )

        df = df[
            df[region_col].astype(str).str.strip().str.lower()
            == region.lower()
        ].copy()

    # Paid/unpaid/cancelled filter.
    status = requested_status(question)

    if status:
        status_col = get_column(df, "status")

        if not status_col:
            raise ValueError(
                f"Status '{status}' was requested, but no status "
                "column was found."
            )

        values = (
            df[status_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if status == "paid":
            # Extend this set if the API uses other documented
            # paid-status values.
            df = df[
                values.isin({"paid", "completed", "succeeded"})
            ].copy()

        elif status == "unpaid":
            df = df[
                values.isin({
                    "unpaid", "pending", "awaiting payment"
                })
            ].copy()

        elif status == "cancelled":
            df = df[
                values.isin({"cancelled", "canceled"})
            ].copy()

    # Product filter.
    product_col = get_column(df, "product")

    if product_col:
        for product in df[product_col].dropna().unique():
            if str(product).strip().lower() in q:
                df = df[df[product_col] == product].copy()
                break

    # Customer filter.
    customer_col = get_column(df, "customer")

    if customer_col:
        for customer in df[customer_col].dropna().unique():
            name = str(customer).strip()

            if len(name) >= 3 and name.lower() in q:
                df = df[df[customer_col] == customer].copy()
                break

    if operation == "count":
        return int(len(df))

    if df.empty:
        raise ValueError(
            "No orders matched the requested filters."
        )

    # Choose monetary metric.
    is_refund = bool(
        re.search(r"\b(refund|refunded|refunds)\b", q)
    )

    amount_col = (
        get_column(df, "refund")
        if is_refund
        else get_column(df, "amount")
    )

    if not amount_col:
        metric = "refund amount" if is_refund else "order amount"
        raise ValueError(
            f"Could not identify the {metric} column. "
            f"Available columns: {list(df.columns)}"
        )

    currency_col = get_column(df, "currency")

    # Convert every row before aggregating.
    converted = []

    for _, row in df.iterrows():
        raw_amount = row[amount_col]

        if pd.isna(raw_amount):
            continue

        try:
            amount = Decimal(str(raw_amount))
        except Exception:
            continue

        source_currency = (
            str(row[currency_col]).strip().upper()
            if currency_col and pd.notna(row[currency_col])
            else DEFAULT_CURRENCY
        )

        if source_currency != target_currency:
            amount = convert_currency(
                amount, source_currency, target_currency, RATES
            )

        converted.append(amount)

    if not converted:
        raise ValueError(
            "No valid monetary values were available."
        )

    if operation == "average":
        result = sum(converted) / Decimal(len(converted))
    elif operation == "max":
        result = max(converted)
    elif operation == "min":
        result = min(converted)
    else:
        result = sum(converted)

    return float(
        result.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    )


def convert_currency(amount, source, target, rates):
    """Convert through USD, using the configured rate convention."""

    source = source.upper()
    target = target.upper()

    usd_amount = to_usd(amount, source, rates)

    if target == "USD":
        return usd_amount

    if target not in rates:
        raise ValueError(
            f"No exchange rate found for {target}."
        )

    if RATE_MODE == "units_per_usd":
        return usd_amount * rates[target]

    return usd_amount / rates[target]


# ---------------------------------------------------------
# Startup and HTTP routes
# ---------------------------------------------------------

@app.on_event("startup")
def startup():
    global ORDERS, RATES, EXPORTS

    metadata = get_json(METADATA_URL)
    endpoints = discover_endpoints(metadata)

    log.info("Discovered endpoints: %s", list(endpoints))

    if "orders" not in endpoints:
        raise RuntimeError(
            "Orders URL not found in metadata. "
            "Inspect the metadata structure and update "
            "discover_endpoints()."
        )

    ORDERS = load_orders(endpoints["orders"])

    rates_payload = (
        get_json(endpoints["rates"])
        if "rates" in endpoints
        else {}
    )

    EXPORTS = (
        get_json(endpoints["exports"])
        if "exports" in endpoints
        else None
    )

    RATES = build_rates(rates_payload)

    log.info("Order columns: %s", list(ORDERS.columns))
    log.info("Currency codes loaded: %s", list(RATES.keys()))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "orders_loaded": len(ORDERS),
        "columns": list(ORDERS.columns),
        "currency_codes": list(RATES.keys()),
    }


@app.post("/")
def answer_question(body: QuestionRequest):
    try:
        return {"answer": analyze(body.question)}

    except ValueError as exc:
        log.warning("Question could not be answered: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))

    except Exception:
        log.exception("Question processing failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question."
        )

