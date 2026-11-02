import sqlite3
import hashlib
import secrets
from pathlib import Path
import pandas as pd

DB_NAME = Path(__file__).with_name("versapay.db")


def get_connection():
    conn = sqlite3.connect(DB_NAME, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database():
    """Initialize tables and safely migrate existing data."""

    with get_connection() as conn:

        # Users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Transactions table: create if it doesn't exist.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                transaction_type TEXT NOT NULL
                    CHECK(transaction_type IN ('Income', 'Expense')),
                category TEXT NOT NULL,
                description TEXT DEFAULT '',
                amount REAL NOT NULL CHECK(amount > 0)
            )
        """)

        # Add user_id to an existing transactions table if needed.
        transaction_columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(transactions)"
            ).fetchall()
        }

        if "user_id" not in transaction_columns:
            conn.execute("""
                ALTER TABLE transactions
                ADD COLUMN user_id INTEGER REFERENCES users(id)
            """)

        # Create a temporary legacy budget table if necessary.
        budget_columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(budgets)"
            ).fetchall()
        }

        if not budget_columns:
            conn.execute("""
                CREATE TABLE budgets (
                    category TEXT PRIMARY KEY,
                    monthly_limit REAL NOT NULL
                        CHECK(monthly_limit >= 0),
                    updated_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
            """)

        # Recheck the budget table structure.
        budget_columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(budgets)"
            ).fetchall()
        }

        # Migrate the old budget table if it lacks user_id.
        if "user_id" not in budget_columns:

            conn.execute("""
                CREATE TABLE budgets_new (
                    user_id INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    monthly_limit REAL NOT NULL
                        CHECK(monthly_limit >= 0),
                    updated_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, category),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)

            # Preserve existing budgets temporarily.
            # They will be assigned to the first registered user.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS legacy_budgets (
                    category TEXT PRIMARY KEY,
                    monthly_limit REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                INSERT OR REPLACE INTO legacy_budgets
                    (category, monthly_limit, updated_at)
                SELECT category, monthly_limit, updated_at
                FROM budgets
            """)

            conn.execute("DROP TABLE budgets")
            conn.execute(
                "ALTER TABLE budgets_new RENAME TO budgets"
            )

        # Preserve the previous budget-saving API while migrating.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS legacy_budgets (
                category TEXT PRIMARY KEY,
                monthly_limit REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)


def assign_legacy_data_to_user(user_id):
    """
    Assign unowned transactions and legacy budgets to a user.
    Call this only after the user has logged in.
    """

    with get_connection() as conn:

        # Claim only transactions that have no owner.
        conn.execute("""
            UPDATE transactions
            SET user_id = ?
            WHERE user_id IS NULL
        """, (user_id,))

        # Move old budgets to the selected user.
        conn.execute("""
            INSERT INTO budgets
                (user_id, category, monthly_limit, updated_at)
            SELECT ?, category, monthly_limit, updated_at
            FROM legacy_budgets
            WHERE 1
            ON CONFLICT(user_id, category) DO NOTHING
        """, (user_id,))

        conn.execute("DELETE FROM legacy_budgets")


def add_transaction(
    user_id, date, transaction_type, category, description, amount
):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO transactions
                (user_id, date, transaction_type, category,
                 description, amount)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            int(user_id),
            str(date),
            transaction_type,
            category,
            description or "",
            float(amount)
        ))


def get_transactions(user_id):
    with get_connection() as conn:
        return pd.read_sql_query("""
            SELECT id, user_id, date, transaction_type,
                   category, description, amount
            FROM transactions
            WHERE user_id = ?
            ORDER BY date DESC, id DESC
        """, conn, params=(int(user_id),))


def delete_transaction(transaction_id, user_id):
    with get_connection() as conn:
        conn.execute("""
            DELETE FROM transactions
            WHERE id = ? AND user_id = ?
        """, (int(transaction_id), int(user_id)))


def get_budgets(user_id):
    with get_connection() as conn:
        return pd.read_sql_query("""
            SELECT category, monthly_limit
            FROM budgets
            WHERE user_id = ?
            ORDER BY category
        """, conn, params=(int(user_id),))


def save_budget(user_id, category, monthly_limit):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO budgets
                (user_id, category, monthly_limit, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, category) DO UPDATE SET
                monthly_limit = excluded.monthly_limit,
                updated_at = CURRENT_TIMESTAMP
        """, (int(user_id), category, float(monthly_limit)))


def save_budgets(user_id, budget_map):
    with get_connection() as conn:
        conn.executemany("""
            INSERT INTO budgets
                (user_id, category, monthly_limit, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, category) DO UPDATE SET
                monthly_limit = excluded.monthly_limit,
                updated_at = CURRENT_TIMESTAMP
        """, [
            (int(user_id), category, float(limit))
            for category, limit in budget_map.items()
        ])


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        600000
    ).hex()

    return password_hash, salt


def create_user(username, email, password):
    username = username.strip()
    email = email.strip().lower()

    if not username or not email or not password:
        return False, "All fields are required."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    password_hash, salt = hash_password(password)

    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO users
                    (username, email, password_hash, salt)
                VALUES (?, ?, ?, ?)
            """, (username, email, password_hash, salt))

        return True, "Account created successfully."

    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."


def authenticate_user(username_or_email, password):
    identifier = username_or_email.strip()

    with get_connection() as conn:
        user = conn.execute("""
            SELECT id, username, email, password_hash, salt
            FROM users
            WHERE username = ? COLLATE NOCASE
               OR email = ? COLLATE NOCASE
        """, (identifier, identifier)).fetchone()

    if user is None:
        return None

    user_id, username, email, stored_hash, salt = user

    calculated_hash, _ = hash_password(password, salt)

    if secrets.compare_digest(calculated_hash, stored_hash):
        return {
            "id": user_id,
            "username": username,
            "email": email
        }

    return None