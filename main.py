
import os
import re
import csv
import io
import json
import logging
from decimal import Decimal, InvalidOperation
from datetime import datetime, date
from zoneinfo import ZoneInfo
from collections import defaultdict, deque

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

QUESTION_DATA_URL = os.getenv("QUESTION_DATA_URL", "").strip()
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
BUSINESS_TZ = ZoneInfo("Asia/Kolkata")

# If the rates endpoint does not specify its base currency,
# assume its rates are units of each currency per USD.
DEFAULT_RATE_BASE = os.getenv("DEFAULT_RATE_BASE", "USD").upper()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Order Ledger Question API")
http = requests.Session()
http.headers.update({"Accept": "application/json"})


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)


# ---------------------------------------------------------
# HTTP and JSON helpers
# ---------------------------------------------------------

def fetch_response(url):
    response = http.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response


def fetch_json(url):
    response = fetch_response(url)
    return response.json()


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def decimal_value(value):
    if value is None:
        return None

    try:
        result = Decimal(str(value).replace(",", "").strip())
        return result if result.is_finite() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def money(value):
    return str(Decimal(value).quantize(Decimal("0.01")))


def parse_datetime(value):
    if not value:
        return None

    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))

        # API business dates use Asia/Kolkata. If a timestamp has
        # no timezone, interpret it in the business timezone.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=BUSINESS_TZ)

        return dt.astimezone(BUSINESS_TZ)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------
# Metadata and pagination
# ---------------------------------------------------------

def load_metadata():
    if not QUESTION_DATA_URL:
        raise RuntimeError("QUESTION_DATA_URL is not configured")

    metadata = fetch_json(QUESTION_DATA_URL)

    if not isinstance(metadata, dict):
        raise RuntimeError("Metadata response is not a JSON object")

    links = metadata.get("links", {})
    if not isinstance(links, dict):
        raise RuntimeError("Metadata does not contain a links object")

    if not links.get("orders"):
        raise RuntimeError("Metadata has no orders link")

    if not links.get("rates"):
        raise RuntimeError("Metadata has no rates link")

    return metadata


def fetch_all_orders(first_url):
    """
    Follow the API's `next` URL rather than constructing page URLs.
    This preserves the signed query parameters and follows the
    server's own pagination instructions.
    """
    rows = []
    seen_pages = set()
    url = first_url

    while url:
        if url in seen_pages:
            raise RuntimeError("Repeated pagination URL detected")

        seen_pages.add(url)

        payload = fetch_json(url)

        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected orders response format")

        page_rows = payload.get("orders", [])
        if not isinstance(page_rows, list):
            raise RuntimeError("The orders field is not a list")

        rows.extend(
            row for row in page_rows if isinstance(row, dict)
        )

        logger.info(
            "Loaded orders page %s of %s",
            payload.get("page", "?"),
            payload.get("pages", "?"),
        )

        # The API supplies the next page URL.
        url = payload.get("next")

    return rows


def deduplicate_orders(rows):
    """
    An order ID can occur more than once.
    Keep the record with the latest updated_at timestamp.
    """
    current = {}

    for index, row in enumerate(rows):
        order_id = row.get("id")

        if order_id is None:
            # Do not accidentally merge records lacking an ID.
            current[f"__missing_id_{index}"] = row
            continue

        key = str(order_id)

        if key not in current:
            current[key] = row
            continue

        old_dt = parse_datetime(current[key].get("updated_at"))
        new_dt = parse_datetime(row.get("updated_at"))

        if new_dt is not None and (
            old_dt is None or new_dt > old_dt
        ):
            current[key] = row

    return list(current.values())


# ---------------------------------------------------------
# Currency conversion
# ---------------------------------------------------------

class CurrencyConverter:
    """
    Builds a directed graph of currency conversion factors.

    Supported rate formats include:
      {"base": "USD", "rates": {"EUR": 0.92, "INR": 83.1}}
      {"USD": 1, "EUR": 0.92, "INR": 83.1}
      [{"currency": "EUR", "rate": 0.92}, ...]
      [{"from": "USD", "to": "EUR", "rate": 0.92}, ...]

    For a base/rates object, rates are interpreted as units of the
    quoted currency per one unit of the base currency.

    If your endpoint defines a different convention, adjust
    _read_rate_payload() to match its actual response.
    """

    def __init__(self, payload):
        self.graph = defaultdict(dict)
        self._read_rate_payload(payload)
        self.graph.setdefault("USD", {})["USD"] = Decimal("1")
        self.graph.setdefault("EUR", {})["EUR"] = Decimal("1")
        self.graph.setdefault("INR", {})["INR"] = Decimal("1")

    def add_rate(self, source, target, factor):
        source = str(source).upper().strip()
        target = str(target).upper().strip()
        factor = decimal_value(factor)

        if not source or not target or factor is None or factor <= 0:
            return

        # factor = units of target currency per one unit of source.
        self.graph[source][target] = factor
        self.graph[target][source] = Decimal("1") / factor

    def _read_rate_payload(self, payload):
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    self._read_rate_payload(item)
            return

        if not isinstance(payload, dict):
            return

        # Explicit pairwise exchange rate.
        source = (
            payload.get("from_currency")
            or payload.get("from")
            or payload.get("base_currency")
        )
        target = (
            payload.get("to_currency")
            or payload.get("to")
            or payload.get("quote_currency")
        )
        factor = (
            payload.get("exchange_rate")
            or payload.get("conversion_rate")
            or payload.get("rate")
        )

        if source and target and factor is not None:
            self.add_rate(source, target, factor)
            return

        # Standard base + rates object.
        base = (
            payload.get("base")
            or payload.get("base_currency")
            or payload.get("baseCurrency")
        )
        rates = payload.get("rates")

        if isinstance(rates, dict):
            base = str(base or DEFAULT_RATE_BASE).upper()

            self.graph[base][base] = Decimal("1")

            for currency, rate in rates.items():
                factor = decimal_value(rate)
                if factor is not None and factor > 0:
                    self.add_rate(base, currency, factor)

            return

        if isinstance(rates, list):
            self._read_rate_payload(rates)
            return

        # Single currency-rate row.
        currency = (
            payload.get("currency")
            or payload.get("currency_code")
            or payload.get("code")
        )
        rate = (
            payload.get("rate")
            or payload.get("exchange_rate")
            or payload.get("conversion_rate")
        )

        if currency and rate is not None:
            base = str(
                payload.get("base")
                or payload.get("base_currency")
                or DEFAULT_RATE_BASE
            ).upper()
            self.add_rate(base, currency, rate)
            return

        # Flat mapping such as {"USD": 1, "EUR": 0.92, "INR": 83.1}.
        currency_codes = {"USD", "EUR", "INR"}
        numeric_items = {
            str(k).upper(): decimal_value(v)
            for k, v in payload.items()
            if str(k).upper() in currency_codes
        }

        if numeric_items and all(v is not None and v > 0
                                 for v in numeric_items.values()):
            base = str(
                payload.get("base") or DEFAULT_RATE_BASE
            ).upper()

            for currency, rate in numeric_items.items():
                self.add_rate(base, currency, rate)

    def convert(self, amount, source, target="USD"):
        amount = decimal_value(amount)

        if amount is None:
            raise ValueError("Amount is not numeric")

        source = str(source or "USD").upper()
        target = str(target or "USD").upper()

        if source == target:
            return amount

        # Find a conversion path using breadth-first search.
        queue = deque([(source, Decimal("1"))])
        visited = {source}

        while queue:
            currency, factor_so_far = queue.popleft()

            for next_currency, factor in self.graph.get(
                currency, {}
            ).items():
                if next_currency in visited:
                    continue

                new_factor = factor_so_far * factor

                if next_currency == target:
                    return amount * new_factor

                visited.add(next_currency)
                queue.append((next_currency, new_factor))

        raise ValueError(
            f"No conversion path from {source} to {target}. "
            "Check the rates endpoint schema."
        )


# ---------------------------------------------------------
# Load optional export data
# ---------------------------------------------------------

def fetch_export_payload(url):
    if not url:
        return None

    response = fetch_response(url)
    return response.json()
#    content_type = response.headers.get("content-type", "").lower()
#
#   if "json" in content_type:
#      return response.json()
#
#    # Support CSV exports too.
#    text = response.text
#    try:
#        return list(csv.DictReader(io.StringIO(text)))
#    except Exception:
#        logger.warning("Could not parse export response as CSV")
#        return None


def find_refund_records(payload):
    """
    Look for refund records in common export structures.
    Returns None if no explicit refund collection is found.
    """
    if isinstance(payload, list):
        if payload and all(isinstance(x, dict) for x in payload):
            if any(
                "refund" in normalize(k)
                for row in payload
                for k in row.keys()
            ):
                return payload
        return None

    if not isinstance(payload, dict):
        return None

    for key, value in payload.items():
        if "refund" in normalize(key):
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                for subkey in ("records", "items", "results", "data"):
                    if isinstance(value.get(subkey), list):
                        return value[subkey]

    for value in payload.values():
        result = find_refund_records(value)
        if result is not None:
            return result

    return None


def load_data():
    metadata = load_metadata()
    links = metadata["links"]

    raw_orders = fetch_all_orders(links["orders"])
    orders = deduplicate_orders(raw_orders)

    rates_payload = fetch_json(links["rates"])
    converter = CurrencyConverter(rates_payload)

    export_payload = fetch_export_payload(
        links.get("export") or links.get("exports")
    )

    refund_records = find_refund_records(export_payload)

    logger.info(
        "Loaded %d raw order rows; %d current orders",
        len(raw_orders),
        len(orders),
    )

    return {
        "metadata": metadata,
        "orders": orders,
        "converter": converter,
        "export": export_payload,
        "refund_records": refund_records,
    }


DATA = None
LOAD_ERROR = None

try:
    DATA = load_data()
except Exception as exc:
    LOAD_ERROR = str(exc)
    logger.exception("Initial data load failed")


# ---------------------------------------------------------
# Question interpretation
# ---------------------------------------------------------

MONTHS = {
    "january": 1, "february": 2, "march": 3,
    "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}


def get_month(question):
    match = re.search(
        r"\b(january|february|march|april|may|june|july|"
        r"august|september|october|november|december)"
        r"\s+(20\d{2})\b",
        question.lower(),
    )

    if not match:
        match = re.search(r"\b(20\d{2})-(0[1-9]|1[0-2])\b",
                          question)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None

    return int(match.group(2)), MONTHS[match.group(1)]


def get_region(question):
    match = re.search(
        r"\b(?:from|in|for|within)\s+(?:the\s+)?"
        r"(north|south|east|west|central)\s+region\b",
        question.lower(),
    )

    if match:
        return match.group(1).title()

    match = re.search(
        r"\b(north|south|east|west|central)\s+orders?\b",
        question.lower(),
    )

    return match.group(1).title() if match else None


def get_status(question):
    q = question.lower()

    if re.search(r"\b(refunded|refunds?)\b", q):
        return "refunded"

    if re.search(r"\b(void|voided)\b", q):
        return "void"

    if re.search(r"\b(cancelled|canceled)\b", q):
        return "cancelled"

    if re.search(r"\b(unpaid|not paid|pending)\b", q):
        return "unpaid"

    if re.search(r"\bpaid\b", q):
        return "paid"

    return None


def get_currency(question):
    q = question.lower()

    if re.search(r"\b(eur|euros?)\b", q):
        return "EUR"

    if re.search(r"\b(inr|rupees?)\b|₹", q):
        return "INR"

    return "USD"


def get_operation(question):
    q = question.lower()

    if (
        "top selling product" in q
        or "top-selling product" in q
        or "best-selling product" in q
        or "product by revenue" in q
        or "top product" in q
    ):
        return "top_product"

    if (
        "top customer" in q
        or "customer by revenue" in q
        or "highest spending customer" in q
    ):
        return "top_customer"

    if re.search(r"\b(how many|count|number of)\b", q):
        return "count"

    if re.search(r"\b(average|on average|mean)\b", q):
        return "average"

    if re.search(r"\b(highest|maximum|largest)\b", q):
        return "maximum"

    if re.search(r"\b(lowest|minimum|smallest)\b", q):
        return "minimum"

    return "sum"


def get_metric(question):
    q = question.lower()

    if "refund" in q:
        return "refund"

    if "customer" in q and re.search(
        r"\b(how many|count|number of)\b", q
    ):
        return "customer_count"

    if "product" in q and re.search(
        r"\b(how many|count|number of)\b", q
    ):
        return "product_count"

    if "unit" in q or "quantity" in q:
        return "quantity"

    return "revenue"


def matches_month(row, month):
    if month is None:
        return True

    dt = parse_datetime(row.get("created_at"))
    if dt is None:
        return False

    return (dt.year, dt.month) == month


def filter_orders(question):
    rows = DATA["orders"]
    month = get_month(question)
    region = get_region(question)
    status = get_status(question)
    q = question.lower()

    result = []

    for row in rows:
        if not matches_month(row, month):
            continue

        if region and str(row.get("region", "")).lower() != region.lower():
            continue

        row_status = str(row.get("status", "")).lower()

        if status == "paid" and row_status != "paid":
            continue
        elif status == "refunded" and row_status != "refunded":
            continue
        elif status == "void" and row_status != "void":
            continue
        elif status == "cancelled" and row_status not in {
            "cancelled", "canceled"
        }:
            continue
        elif status == "unpaid" and row_status not in {
            "unpaid", "pending", "failed"
        }:
            continue

        # Revenue/sales questions count only paid orders by default.
        if status is None and (
            "revenue" in q or "sales" in q or "top selling" in q
        ):
            if row_status != "paid":
                continue

        # Match a named product if the question explicitly includes it.
        product_match = re.search(
            r"\bproduct\s+['\"]?([a-z0-9 -]+?)['\"]?"
            r"(?=\s+(?:in|from|for|during|with|by)\b|[?.!,]|$)",
            question,
            re.IGNORECASE,
        )

        if product_match:
            product_name = product_match.group(1).strip()
            if str(row.get("product", "")).lower() != product_name.lower():
                continue

        # Match a named customer ID, e.g. customer C1054.
        customer_match = re.search(
            r"\bcustomer\s+(C\d+)\b", question, re.IGNORECASE
        )
        if customer_match:
            if str(row.get("customer", "")).lower() != (
                customer_match.group(1).lower()
            ):
                continue

        result.append(row)

    return result


# ---------------------------------------------------------
# Calculations
# ---------------------------------------------------------

def amount_in_currency(row, currency="USD", metric="revenue"):
    if metric == "quantity":
        return decimal_value(row.get("qty")) or Decimal("0")

    amount = decimal_value(row.get("amount"))

    if amount is None:
        qty = decimal_value(row.get("qty"))
        price = decimal_value(row.get("unit_price"))

        if qty is None or price is None:
            return Decimal("0")

        amount = qty * price

    source_currency = row.get("currency") or "USD"

    return DATA["converter"].convert(
        amount,
        source_currency,
        currency,
    )


def get_refund_rows():
    """
    Prefer explicit refund records from the export if available.
    Otherwise, use orders whose status is refunded.
    """
    explicit = DATA.get("refund_records")

    if explicit is not None:
        return explicit

    return [
        row for row in DATA["orders"]
        if str(row.get("status", "")).lower() == "refunded"
    ]


def aggregate_by(rows, field, currency="USD", metric="revenue"):
    totals = defaultdict(Decimal)

    for row in rows:
        key = row.get(field)
        if key is None:
            continue

        value = amount_in_currency(row, currency, metric)
        totals[str(key)] += value

    return totals


def analyze(question):
    if DATA is None:
        raise RuntimeError(
            f"Data loading failed: {LOAD_ERROR or 'unknown error'}"
        )

    operation = get_operation(question)
    metric = get_metric(question)
    currency = get_currency(question)
    rows = filter_orders(question)

    if metric == "refund":
        rows = [
            row for row in get_refund_rows()
            if isinstance(row, dict)
            and matches_month(row, get_month(question))
            and (
                not get_region(question)
                or str(row.get("region", "")).lower()
                == get_region(question).lower()
            )
        ]

    if metric == "customer_count":
        return str(len({
            row.get("customer")
            for row in rows
            if row.get("customer") is not None
        }))

    if metric == "product_count":
        return str(len({
            row.get("product")
            for row in rows
            if row.get("product") is not None
        }))

    if operation == "count":
        return str(len(rows))

    if operation == "top_product":
        if metric == "quantity":
            totals = aggregate_by(rows, "product", currency, "quantity")
        else:
            # Revenue ranking is based on total converted revenue,
            # not on the number of units sold.
            totals = aggregate_by(rows, "product", currency, "revenue")

        if not totals:
            return "No matching product data found"

        # Stable tie-breaking: product name alphabetically.
        winner = sorted(
            totals.items(),
            key=lambda item: (-item[1], item[0].lower(), item[0]),
        )[0]

        return winner[0]

    if operation == "top_customer":
        totals = aggregate_by(rows, "customer", currency, "revenue")

        if not totals:
            return "No matching customer data found"

        winner = sorted(
            totals.items(),
            key=lambda item: (-item[1], item[0].lower(), item[0]),
        )[0]

        return winner[0]

    if metric == "quantity":
        values = [
            decimal_value(row.get("qty"))
            for row in rows
            if decimal_value(row.get("qty")) is not None
        ]
    else:
        values = [
            amount_in_currency(row, currency, "revenue")
            for row in rows
        ]

    if not values:
        return "No matching data found"

    if operation == "average":
        return money(sum(values, Decimal("0")) / Decimal(len(values)))

    if operation == "maximum":
        return money(max(values))

    if operation == "minimum":
        return money(min(values))

    return money(sum(values, Decimal("0")))


# ---------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok" if DATA is not None else "error",
        "orders_loaded": len(DATA["orders"]) if DATA else 0,
        "error": LOAD_ERROR,
    }


@app.post("/")
def answer_question(request: QuestionRequest):
    try:
        answer = analyze(request.question)
        return {"answer": answer}
    except Exception as exc:
        logger.exception("Question processing failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/reload")
def reload_data():
    global DATA, LOAD_ERROR

    try:
        DATA = load_data()
        LOAD_ERROR = None
        return {
            "status": "ok",
            "orders_loaded": len(DATA["orders"]),
        }
    except Exception as exc:
        LOAD_ERROR = str(exc)
        logger.exception("Reload failed")
        raise HTTPException(status_code=500, detail=LOAD_ERROR)

