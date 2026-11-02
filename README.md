# VersaPay AI — upgraded college project

## What is included
1. **Premium fintech UI:** dark dashboard, cash-flow charts, category split, transaction export and improved navigation.
2. **Smarter analytics:** persistent category budgets in SQLite, budget alerts, repeated-category detection, spending-pace comparison and a linear-regression forecast.
3. **AI chat:** works locally without a key for common finance questions. If `OPENAI_API_KEY` is configured, it can use the OpenAI API for conversational responses grounded in the app's recorded transaction summary.

## Important
- This is a student demonstration, not a payment app. It does **not** connect to Paytm, UPI, banks or cards and cannot make payments.
- Transactions and budgets are stored locally in `versapay.db`.
- Keep your existing `versapay.db` file if you already entered transactions. The updated database module adds the budgets table without deleting your transactions.
- Cloud AI sends the finance summary included in the prompt to the configured API provider. Use only test data unless you are comfortable with that data flow.
- Never enter passwords, OTPs, card details or bank credentials.
- Forecasts are estimates and may be inaccurate with limited data.

## Run on Windows PowerShell
From the VersaPay AI project folder:

```powershell
# Activate your existing virtual environment
.\.venv\Scripts\Activate.ps1

# Install/update dependencies
python -m pip install -r requirements.txt

# Start the app
python -m streamlit run app.py
```

If PowerShell blocks activation, you can run Streamlit without activating:
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Optional cloud AI setup
Set an API key in the same PowerShell terminal (do not put the key in app.py):

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
python -m streamlit run app.py
```

Optional model override:
```powershell
$env:OPENAI_MODEL="gpt-4o-mini"
```

The API key only lasts for that terminal session. Do not upload it to GitHub or include it in screenshots/reports.

## Project files
- `app.py` — Streamlit interface, charts, budget logic, forecast and chat.
- `database.py` — SQLite transaction and budget persistence.
- `requirements.txt` — Python dependencies.

## Suggested viva explanation
- The database uses SQLite to persist transactions and category budgets.
- Pandas aggregates transaction data for the dashboard and category analytics.
- Plotly renders interactive charts.
- Scikit-learn's LinearRegression estimates next-month expenses from monthly totals.
- The local assistant uses deterministic logic; when an API key is configured, the chat can use an external language model.
