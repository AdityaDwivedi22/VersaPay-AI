"""
versapay_llm.py — VersaPay AI Ollama Integration Layer

Provides:
  - check_ollama_status()   : health-check the Ollama service and configured model
  - get_financial_context() : build a compact, privacy-preserving financial summary
  - ask_versapay()          : send a question + history to Ollama; return (reply, source_label)

Configuration (set in your PowerShell session before launching Streamlit):
  $env:VERSAPAY_OLLAMA_MODEL = "versapay-ai"     # default
  $env:VERSAPAY_OLLAMA_HOST  = "http://localhost:11434"   # default
"""

import os
import re
from datetime import date
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — read from environment, never hardcoded
# ---------------------------------------------------------------------------

PREFERRED_MODEL = "versapay-ai"
FALLBACK_MODEL  = "qwen2.5:3b"
DEFAULT_HOST    = "http://localhost:11434"
REQUEST_TIMEOUT = 90          # seconds; generous for first-load of a 3B model
MAX_HISTORY_MESSAGES = 8      # last N messages sent to model to stay within context window

def _configured_model() -> str:
    """Return the model name from env, defaulting to versapay-ai."""
    return os.getenv("VERSAPAY_OLLAMA_MODEL", PREFERRED_MODEL).strip() or PREFERRED_MODEL

def _configured_host() -> str:
    """Return the Ollama host from env."""
    return os.getenv("VERSAPAY_OLLAMA_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST


# ---------------------------------------------------------------------------
# System prompt (kept lean — rich persona already baked into the Modelfile)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are VersaPay AI, an intelligent personal finance assistant.

Help users understand their budgets, spending habits, and savings goals.
Use Indian Rupees (₹) when discussing amounts.
Give practical, personalised, non-judgmental suggestions.
When financial data is provided below, use it accurately — never invent figures.
When information is missing, ask a clear follow-up question.
Never request passwords, OTPs, UPI PINs, or banking credentials.
Never claim to access real bank accounts or execute payments.
Keep replies concise and actionable. Use bullet points or tables when helpful."""


# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------

def check_ollama_status() -> tuple[bool, bool, str]:
    """
    Check whether Ollama is reachable and whether the configured model is available.

    Returns:
        (service_reachable: bool, model_available: bool, human_readable_message: str)
    """
    import urllib.request, urllib.error, json

    host = _configured_host()
    model = _configured_model()

    # 1. Check if the service responds at all.
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        return (
            False, False,
            f"Ollama is not reachable at {host}. "
            f"Start it with: ollama serve\nDetail: {exc}"
        )

    # 2. Check whether the configured model is listed.
    installed_models = [m.get("name", "") for m in data.get("models", [])]
    # Normalize: "versapay-ai:latest" should match "versapay-ai"
    def _base(name: str) -> str:
        return name.split(":")[0].lower()

    model_base = _base(model)
    available = any(_base(m) == model_base for m in installed_models)

    if available:
        return True, True, f"✅  Ollama connected · model **{model}** is ready"

    # Model missing — check if fallback is present.
    fallback_available = any(_base(m) == _base(FALLBACK_MODEL) for m in installed_models)
    installed_display  = ", ".join(installed_models) if installed_models else "none"

    msg = (
        f"⚠️  Ollama is running but model **{model}** is not installed.\n\n"
        f"Installed models: {installed_display}\n\n"
    )
    if fallback_available:
        msg += (
            f"**Fallback model {FALLBACK_MODEL} is available** and will be used.\n\n"
            f"To install {model}, run in PowerShell:\n"
            f"```\nollama create versapay-ai -f Modelfile\n```"
        )
    else:
        msg += (
            f"To install the custom model, run:\n"
            f"```\nollama create versapay-ai -f Modelfile\n```\n\n"
            f"Or to use {FALLBACK_MODEL} directly:\n"
            f"```\nollama pull {FALLBACK_MODEL}\n$env:VERSAPAY_OLLAMA_MODEL = \"{FALLBACK_MODEL}\"\n```"
        )
    return True, False, msg


# ---------------------------------------------------------------------------
# Financial context builder (Python-calculated, privacy-preserving)
# ---------------------------------------------------------------------------

def get_financial_context(df: pd.DataFrame, budgets: dict) -> str:
    """
    Compute a compact financial summary from the user's transaction DataFrame and
    budget dict using Python arithmetic. Only category-level aggregates are included —
    no individual transaction descriptions or dates are sent to the model.
    """
    if df.empty:
        return "No transaction history is recorded yet."

    lines = ["== VersaPay Financial Summary (Python-verified) =="]

    # All-time totals
    income  = float(df.loc[df["transaction_type"] == "Income",  "amount"].sum())
    expense = float(df.loc[df["transaction_type"] == "Expense", "amount"].sum())
    balance = income - expense
    savings_rate = (balance / income * 100) if income > 0 else 0.0
    lines.append(f"All-time income  : ₹{income:,.0f}")
    lines.append(f"All-time expenses: ₹{expense:,.0f}")
    lines.append(f"Net balance      : ₹{balance:,.0f}")
    lines.append(f"Savings rate     : {savings_rate:.1f}%")

    # Current-month expenses
    today       = date.today()
    month_start = date(today.year, today.month, 1)
    import calendar
    month_end   = date(today.year, today.month, calendar.monthrange(today.year, today.month)[1])

    df_date = df.copy()
    df_date["date"] = pd.to_datetime(df_date["date"], errors="coerce")
    this_month = df_date[
        (df_date["transaction_type"] == "Expense") &
        (df_date["date"].dt.date >= month_start) &
        (df_date["date"].dt.date <= month_end)
    ]
    month_total = float(this_month["amount"].sum()) if not this_month.empty else 0.0
    lines.append(f"\nCurrent month ({today.strftime('%B %Y')}) expenses: ₹{month_total:,.0f}")

    # Category breakdown this month + budget utilisation
    if not this_month.empty:
        lines.append("\nCategory spending this month vs. budget:")
        cat_totals = this_month.groupby("category")["amount"].sum().sort_values(ascending=False)
        for cat, spent in cat_totals.items():
            limit = budgets.get(cat, 0)
            if limit > 0:
                pct   = spent / limit * 100
                flag  = " ⚠️ OVER BUDGET" if spent >= limit else (" ⚠️ ~80%" if spent >= limit * 0.8 else "")
                lines.append(f"  {cat}: ₹{spent:,.0f} / ₹{limit:,.0f} ({pct:.0f}%){flag}")
            else:
                lines.append(f"  {cat}: ₹{spent:,.0f} (no budget set)")

    # Total configured budget
    total_budget = sum(budgets.values())
    if total_budget > 0:
        lines.append(f"\nTotal configured monthly budget: ₹{total_budget:,.0f}")
        budget_used_pct = (month_total / total_budget * 100) if total_budget else 0
        lines.append(f"Budget used this month: {budget_used_pct:.1f}%")

    lines.append("\nNote: All figures are based solely on transactions the user has recorded in VersaPay.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main chat function
# ---------------------------------------------------------------------------

def ask_versapay(
    question: str,
    conversation_history: list[dict],
    financial_context: Optional[str] = None,
) -> tuple[str, str]:
    """
    Send a question to Ollama with conversation history and financial context.

    Args:
        question             : The user's latest message.
        conversation_history : List of {"role": "user"|"assistant", "content": str}
                               representing the visible chat history.
        financial_context    : Pre-calculated financial summary string (optional).

    Returns:
        (reply_text: str, source_label: str)
        source_label is shown in the UI so the user knows where the answer came from.
    """
    try:
        import ollama as _ollama
    except ImportError:
        return (
            "The `ollama` Python package is not installed. "
            "Run: pip install ollama>=0.3",
            "Error — missing package"
        )

    model = _configured_model()
    host  = _configured_host()

    # Build message list: system prompt → optional financial context → trimmed history → new question
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if financial_context and financial_context.strip():
        messages.append({
            "role": "system",
            "content": (
                "The following verified financial summary was computed by Python from "
                "the user's recorded transactions. Use it to ground your answer:\n\n"
                + financial_context
            ),
        })

    # Trim history to MAX_HISTORY_MESSAGES to stay within the 4096-token context window.
    # Only include actual user/assistant turns (not system messages).
    trimmed_history = [
        m for m in conversation_history
        if m.get("role") in ("user", "assistant")
    ][-MAX_HISTORY_MESSAGES:]

    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": question})

    # Attempt with configured model, fall back to FALLBACK_MODEL if not found.
    for attempt_model in [model, FALLBACK_MODEL]:
        try:
            client   = _ollama.Client(host=host)
            response = client.chat(
                model=attempt_model,
                messages=messages,
                options={"temperature": 0.4, "num_ctx": 4096},
            )
            reply = response["message"]["content"]
            if attempt_model != model:
                source_label = (
                    f"🟡 VersaPay AI · {attempt_model} (fallback — "
                    f"configured model '{model}' not found)"
                )
            else:
                source_label = f"🟢 VersaPay AI · {attempt_model} via Ollama"
            return reply, source_label

        except Exception as exc:
            err_str = str(exc)
            # If it's a model-not-found error, try the fallback.
            if "not found" in err_str.lower() or "does not exist" in err_str.lower() or "pull" in err_str.lower():
                if attempt_model == FALLBACK_MODEL:
                    # Both models unavailable.
                    break
                continue
            # Any other error (connection refused, timeout, etc.) — don't retry.
            return (
                None,
                f"ollama_error:{err_str}"
            )

    # Both models failed with not-found errors.
    return (
        None,
        f"ollama_error:Neither '{model}' nor '{FALLBACK_MODEL}' is installed. "
        f"Run: ollama create versapay-ai -f Modelfile"
    )


# ---------------------------------------------------------------------------
# Standalone CLI (unchanged from original versapay_llm.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    reachable, model_ok, status_msg = check_ollama_status()
    print("=" * 50)
    print("           VERSAPAY AI — CLI MODE")
    print("       Your Personal Finance Assistant")
    print("=" * 50)
    print(f"Status: {status_msg}\n")

    if not reachable:
        print("Cannot connect to Ollama. Start it with: ollama serve")
        raise SystemExit(1)

    history: list[dict] = []

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nVersaPay AI: Goodbye! Spend wisely.")
            break

        if question.lower() in {"exit", "quit", "bye"}:
            print("VersaPay AI: Goodbye! Spend wisely.")
            break

        if not question:
            continue

        reply, label = ask_versapay(question, history)
        if reply is None:
            print(f"\n[Error] {label}")
        else:
            print(f"\nVersaPay AI [{label}]:\n{reply}")
            history.append({"role": "user",      "content": question})
            history.append({"role": "assistant",  "content": reply})