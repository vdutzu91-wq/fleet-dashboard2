import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import io
import base64
import calendar
import shutil
import os
import tempfile
import traceback
import errno
import json
import numpy as np
import time
import mimetypes
import requests
from math import isfinite

# -------------------------
# App constants and roles
# -------------------------
ROLE_NAMES = {
    "admin": "Administrator",
    "dispatcher": "Dispatcher",
    "accountant": "Accountant",
    "driver": "Driver",
    "viewer": "Viewer"
}

COMPANY_START = date(2019, 1, 1)

# All available pages in your app
ALL_PAGES = [
    "Dashboard",
    "Trucks",
    "Trailers",
    "Drivers",
    "Dispatchers",
    "Income",
    "Expenses",
    "Reports",
    "Histories",
    "Bulk Upload",
    "Settings",
    "👥 User Management"
]

# -------------------------
# Persistent DB path (works locally and on Streamlit Cloud)
# -------------------------
# On Streamlit Cloud, set DB_DIR="/mount/data" in App → Settings → Secrets.
DB_DIR = st.secrets.get("DB_DIR", ".")
# Avoid trying to create the managed folder on Streamlit Cloud
if DB_DIR not in ("/mount/data", "/app/data"):
    os.makedirs(os.path.expanduser(DB_DIR), exist_ok=True)

DB_FILE = os.path.join(os.path.expanduser(DB_DIR), "fleet_management.db")

# -------------------------
# DB connection helper
# -------------------------
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        DB_FILE,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
    except Exception:
        pass
    return conn

# Enable WAL mode once at startup (safe if called multiple times)
try:
    with get_db_connection() as _c:
        _c.execute("PRAGMA journal_mode=WAL;")
except Exception:
    pass

def close_all_db_connections():
    # No-op now; kept for compatibility
    pass

def close_all_db_connections_if_any():
    """
    Close any long-lived/global DB connections your app might keep open.
    Modify this to close any global variables you use.
    """
    try:
        for name in ('global_conn', 'global_connection', 'db_conn', 'conn'):
            if name in globals():
                obj = globals().get(name)
                if obj:
                    try:
                        obj.close()
                    except Exception:
                        pass
                    try:
                        globals()[name] = None
                    except Exception:
                        pass
    except Exception:
        pass
    try:
        if 'engine' in globals() and globals().get('engine'):
            try:
                globals()['engine'].dispose()
            except Exception:
                pass
    except Exception:
        pass

# -------------------------
# Auth and user helpers
# -------------------------
import hashlib

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    """Verify password against hash"""
    return hash_password(password) == password_hash

def authenticate_user(username, password):
    """Authenticate user and return user data"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT user_id, username, password_hash, full_name, email, role, allowed_pages, is_active
        FROM users
        WHERE username = ? AND is_active = 1
    """, (username,))
    user = cur.fetchone()
    conn.close()
    if user and verify_password(password, user[2]):
        # Parse allowed pages from JSON
        try:
            allowed_pages = json.loads(user[6]) if user[6] else []
        except Exception:
            allowed_pages = []
        return {
            "user_id": user[0],
            "username": user[1],
            "full_name": user[3],
            "email": user[4],
            "role": user[5],
            "allowed_pages": allowed_pages
        }
    return None

def log_session_action(user_id, username, action, ip_address=None):
    """Log user actions for audit trail"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO session_logs (user_id, username, action, ip_address)
            VALUES (?, ?, ?, ?)
        """, (user_id, username, action, ip_address))
        conn.commit()
        conn.close()
    except Exception:
        # Silent fail for logging
        pass

def update_last_login(user_id):
    """Update user's last login timestamp"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE users
        SET last_login = CURRENT_TIMESTAMP
        WHERE user_id = ?
    """, (user_id,))
    conn.commit()
    conn.close()

def can_access_page(page_name):
    """Check if current user can access a specific page"""
    if "user" not in st.session_state or st.session_state.user is None:
        return False
    if st.session_state.user.get("role") == "admin":
        return True
    return page_name in st.session_state.user.get("allowed_pages", [])

def get_user_pages():
    """Get list of pages current user can access"""
    if "user" not in st.session_state or st.session_state.user is None:
        return []
    if st.session_state.user.get("role") == "admin":
        return ALL_PAGES
    return st.session_state.user.get("allowed_pages", [])

def init_users_db():
    """Initialize users and authentication tables"""
    conn = get_db_connection()
    cur = conn.cursor()
    # Users table with page permissions
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            email TEXT,
            role TEXT NOT NULL DEFAULT 'viewer',
            allowed_pages TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    """)
    # Session logs table (for audit trail)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS session_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            action TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    # Check if allowed_pages column exists, if not add it
    cur.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cur.fetchall()]
    if 'allowed_pages' not in columns:
        cur.execute("ALTER TABLE users ADD COLUMN allowed_pages TEXT")
    # Create default admin user if no users exist
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        default_password = "admin123"
        password_hash = hashlib.sha256(default_password.encode()).hexdigest()
        all_pages = json.dumps(ALL_PAGES)
        cur.execute("""
            INSERT INTO users (username, password_hash, full_name, role, allowed_pages, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("admin", password_hash, "System Administrator", "admin", all_pages, 1))
        conn.commit()
    conn.close()

# Call this in your main initialization
init_users_db()

# -------------------------
# Export helpers (Excel + PDF table)
# -------------------------
def export_to_excel(df, filename_prefix="report"):
    """Return an Excel download button for a given dataframe."""
    if df is None or df.empty:
        st.warning("No data available to export.")
        return None
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    st.download_button(
        label=f"📊 Download {filename_prefix}.xlsx",
        data=buffer.getvalue(),
        file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# If you are using pdfkit, make sure pdfkit_config is defined elsewhere as you had it
def export_to_pdf_table(df, title="Report"):
    """Exports a DataFrame to PDF using pdfkit (wkhtmltopdf backend)."""
    if df is None or df.empty:
        st.warning("No data available to export.")
        return
    html = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
          body {{
            font-family: Arial, sans-serif;
            margin: 25px;
          }}
          h2 {{
            text-align: center;
            margin-bottom: 20px;
          }}
          table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 13px;
          }}
          th, td {{
            border: 1px solid #ccc;
            padding: 6px 10px;
            text-align: center;
          }}
          th {{
            background-color: #f2f2f2;
            font-weight: bold;
          }}
        </style>
      </head>
      <body>
        <h2>{title}</h2>
        {df.to_html(index=False, justify="center")}
      </body>
    </html>
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        # pdfkit_config must exist in your file
        pdfkit.from_string(html, tmpfile.name, configuration=pdfkit_config)
        tmpfile.seek(0)
        st.download_button(
            label=f"📄 Download {title}.pdf",
            data=tmpfile.read(),
            file_name=f"{title}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )

def export_buttons(df, base_name="report", title="Report"):
    """Render Excel + PDF export buttons side by side."""
    if df is None or df.empty:
        st.info("No data to export.")
        return
    c1, c2 = st.columns(2)
    with c1:
        export_to_excel(df, base_name)
    with c2:
        export_to_pdf_table(df, title)

# -------------------------
# Date helpers
# -------------------------
def to_date(v):
    """
    Convert various date-like values to a python date.
    Accepts: None, str (YYYY-MM-DD or other parseable), datetime.date, datetime.datetime, pandas Timestamp.
    Returns: date or None.
    """
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    try:
        return pd.to_datetime(v).date()
    except Exception:
        pass
    try:
        return datetime.strptime(str(v), '%Y-%m-%d').date()
    except Exception:
        for fmt in ('%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d'):
            try:
                return datetime.strptime(str(v), fmt).date()
            except Exception:
                continue
    return None

def get_preset_date_range(preset):
    """Return (start_date, end_date) for common presets."""
    today = date.today()
    if preset == "Today":
        return today, today
    elif preset == "Last 7 days":
        return today - timedelta(days=7), today
    elif preset == "Last 30 days":
        return today - timedelta(days=30), today
    elif preset == "This Month":
        return today.replace(day=1), today
    elif preset == "Last Month":
        first_this_month = today.replace(day=1)
        last_month_end = first_this_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return last_month_start, last_month_end
    elif preset == "This Quarter":
        quarter_start_month = ((today.month - 1) // 3) * 3 + 1
        return today.replace(month=quarter_start_month, day=1), today
    elif preset == "Last Quarter":
        quarter_start_month = ((today.month - 1) // 3) * 3 + 1
        this_q_start = today.replace(month=quarter_start_month, day=1)
        last_q_end = this_q_start - timedelta(days=1)
        last_q_start_month = ((last_q_end.month - 1) // 3) * 3 + 1
        last_q_start = last_q_end.replace(month=last_q_start_month, day=1)
        return last_q_start, last_q_end
    elif preset == "Year-To-Date":
        return date(today.year, 1, 1), today
    elif preset == "Last Year":
        return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)
    elif preset == "All Time":
        return COMPANY_START, today
    else:
        return today.replace(month=1, day=1), today  # default YTD

# -------------------------
# OpenRouteService API Helper
# -------------------------
ORS_API_KEY = st.secrets.get("ORS_API_KEY", "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImI4M2U0MWVhZDc1NjQ2ZDVhMWNlOTlhNmNiNWQ3MjI4IiwiaCI6Im11cm11cjY0In0=")

def geocode_address(address: str) -> tuple:
    """
    Geocode an address to (longitude, latitude) using OpenRouteService.
    Returns (None, None) if geocoding fails.
    """
    if not address or not address.strip():
        return None, None
    try:
        url = "https://api.openrouteservice.org/geocode/search"
        headers = {"Authorization": ORS_API_KEY, "Accept": "application/json, application/geo+json"}
        params = {"text": address.strip(), "size": 1, "boundary.country": "US,CA,MX"}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code != 200:
            return None, None
        data = response.json()
        if data.get("features"):
            lon, lat = data["features"][0]["geometry"]["coordinates"]
            if -170 <= lon <= -50 and 15 <= lat <= 75:
                return lon, lat
        return None, None
    except Exception:
        return None, None

def calculate_distance_miles(start_address: str, end_address: str, debug=False) -> float:
    """
    Calculate driving distance in miles between two addresses using OpenRouteService.
    Returns 0.0 if calculation fails.
    """
    if not start_address or not end_address:
        if debug:
            st.warning("Missing start or end address")
        return 0.0
    start_lon, start_lat = geocode_address(start_address)
    end_lon, end_lat = geocode_address(end_address)
    if start_lon is None or end_lon is None:
        if debug:
            st.warning(f"Geocoding failed for: {start_address} → {end_address}")
        return 0.0
    # Quick distance check using Haversine formula
    from math import radians, sin, cos, sqrt, atan2
    R = 3958.8  # Earth radius in miles
    lat1, lon1 = radians(start_lat), radians(start_lon)
    lat2, lon2 = radians(end_lat), radians(end_lon)
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    straight_line_distance = 2 * atan2(sqrt(a), sqrt(1-a)) * R
    if debug:
        st.info(f"📏 Straight-line distance: {straight_line_distance:.2f} miles")
    if straight_line_distance > 4000:
        if debug:
            st.warning(f"⚠️ Distance too large ({straight_line_distance:.0f} mi). Check addresses.")
        return 0.0
    try:
        url = "https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {
            "Authorization": ORS_API_KEY,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json"
        }
        body = {"coordinates": [[start_lon, start_lat], [end_lon, end_lat]], "units": "mi", "instructions": False}
        if debug:
            st.json({"request": body})
        response = requests.post(url, headers=headers, json=body, timeout=15)
        if response.status_code in (400, 404):
            if debug:
                st.error(f"API {response.status_code} Error: {response.text}")
            return 0.0
        if response.status_code != 200:
            if debug:
                st.error(f"API Error {response.status_code}: {response.text}")
            return 0.0
        data = response.json()
        if debug:
            st.json({"response": data})
        if not data.get("routes"):
            if debug:
                st.warning("No routes found in API response")
            return 0.0
        summary = data["routes"][0].get("summary", {})
        distance_value = summary.get("distance", 0)
        if distance_value > 0:
            # The API typically returns meters; convert if value is too large to be miles
            expected_driving_miles = straight_line_distance * 1.3
            if abs(distance_value - expected_driving_miles) < expected_driving_miles * 0.5:
                distance_miles = distance_value
            elif distance_value > 1000 or (distance_value / 1609.344) < expected_driving_miles * 2:
                distance_miles = distance_value / 1609.344
            else:
                distance_miles = distance_value * 0.621371
            return round(distance_miles, 2)
        return 0.0
    except requests.exceptions.RequestException as e:
        if debug:
            st.error(f"Request error: {e}")
        return 0.0
    except Exception as e:
        if debug:
            st.error(f"Unexpected error: {e}")
        return 0.0

# -------------------------
# Attachment helpers
# -------------------------
def save_uploaded_file_to_base64(uploaded_file):
    """Convert uploaded file to base64 string with metadata."""
    try:
        file_bytes = uploaded_file.read()
        b64 = base64.b64encode(file_bytes).decode('utf-8')
        mime_type = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0] or "application/octet-stream"
        return {
            "filename": uploaded_file.name,
            "mime_type": mime_type,
            "size": len(file_bytes),
            "data": b64
        }
    except Exception as e:
        st.error(f"Failed to encode file: {e}")
        return None

def get_attachments_for_expense(expense_id):
    """Retrieve attachments for a given expense."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT attachments FROM expenses WHERE expense_id = ?", (expense_id,))
        row = cur.fetchone()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except Exception:
                return []
        return []
    finally:
        conn.close()

def add_attachment_to_expense(expense_id, attachment_dict):
    """Add an attachment to an expense."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        attachments = get_attachments_for_expense(expense_id)
        attachments.append(attachment_dict)
        cur.execute("UPDATE expenses SET attachments = ? WHERE expense_id = ?", (json.dumps(attachments), expense_id))
        conn.commit()
    finally:
        conn.close()

def remove_attachment_from_expense(expense_id, attachment_index):
    """Remove an attachment by index."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        attachments = get_attachments_for_expense(expense_id)
        if 0 <= attachment_index < len(attachments):
            attachments.pop(attachment_index)
            cur.execute("UPDATE expenses SET attachments = ? WHERE expense_id = ?", (json.dumps(attachments), expense_id))
            conn.commit()
    finally:
        conn.close()

# -------------------------
# Expense categories and metadata
# -------------------------
def ensure_expense_categories_table():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS expense_categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                schema_json TEXT NOT NULL,
                default_apply_mode TEXT NOT NULL
            )
        """)
        # ensure expenses table has metadata and apply_mode columns
        try:
            cur.execute("ALTER TABLE expenses ADD COLUMN metadata TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            cur.execute("ALTER TABLE expenses ADD COLUMN apply_mode TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()
    finally:
        if conn:
            conn.close()

def get_expense_categories():
    ensure_expense_categories_table()
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT name, schema_json, default_apply_mode FROM expense_categories ORDER BY name")
        rows = cur.fetchall()
        cats = []
        for name, schema_json, default_apply_mode in rows:
            try:
                schema = json.loads(schema_json)
            except Exception:
                schema = []
            cats.append({"name": name, "schema": schema, "default_apply_mode": default_apply_mode})
        return cats
    finally:
        conn.close()

def add_or_update_expense_category(name, field_list, default_apply_mode="individual"):
    """
    field_list: list of dicts, each dict: {"key": "card_number", "label": "Card Number", "type": "text"}
    """
    ensure_expense_categories_table()
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        schema_json = json.dumps(field_list)
        try:
            cur.execute("INSERT INTO expense_categories (name, schema_json, default_apply_mode) VALUES (?, ?, ?)",
                        (name, schema_json, default_apply_mode))
        except sqlite3.IntegrityError:
            cur.execute("UPDATE expense_categories SET schema_json=?, default_apply_mode=? WHERE name=?",
                        (schema_json, default_apply_mode, name))
        conn.commit()
    finally:
        conn.close()

def ensure_default_expense_categories():
    fuel_schema = [
        {"key": "card_number", "label": "Card Number", "type": "text"},
        {"key": "transaction_date", "label": "Transaction Date", "type": "date"},
        {"key": "location", "label": "Location", "type": "text"},
        {"key": "amount_paid", "label": "Amount Paid", "type": "number"},
        {"key": "discount_amount", "label": "Discount Amount", "type": "number"}
    ]
    add_or_update_expense_category("Fuel", fuel_schema, default_apply_mode="individual")
    tolls_schema = [
        {"key": "toll_agency", "label": "Toll Agency", "type": "text"},
        {"key": "date_occurred", "label": "Date Occurred", "type": "date"},
        {"key": "toll_amount", "label": "Toll Amount", "type": "number"}
    ]
    add_or_update_expense_category("Tolls", tolls_schema, default_apply_mode="individual")

def ensure_maintenance_category():
    maintenance_schema = [
        {"key": "service_type", "label": "Service Type", "type": "text"},
        {"key": "parts_cost", "label": "Parts Cost", "type": "number"},
        {"key": "labor_cost", "label": "Labor Cost", "type": "number"},
        {"key": "location", "label": "Location", "type": "text"}
    ]
    add_or_update_expense_category("Maintenance", maintenance_schema, default_apply_mode="individual")

def ensure_expenses_attachments():
    """Add attachments column to expenses table if it doesn't exist, and create indexes for performance."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA table_info(expenses)")
        columns = [row[1] for row in cur.fetchall()]
        if "attachments" not in columns:
            cur.execute("ALTER TABLE expenses ADD COLUMN attachments TEXT")
            conn.commit()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_expenses_date ON expenses(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_expenses_category ON expenses(category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_expenses_truck_id ON expenses(truck_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_expenses_date_category ON expenses(date, category)")
        conn.commit()
    finally:
        conn.close()

# -------------------------
# Core DB creation & migration
# -------------------------
def init_database():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    # trucks
    cur.execute('''
    CREATE TABLE IF NOT EXISTS trucks (
        truck_id INTEGER PRIMARY KEY AUTOINCREMENT,
        number TEXT UNIQUE,
        make TEXT,
        model TEXT,
        year INTEGER,
        plate TEXT,
        vin TEXT,
        status TEXT DEFAULT 'Active',
        trailer_id INTEGER,
        driver_id INTEGER,
        loan_amount REAL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # trailers
    cur.execute('''
    CREATE TABLE IF NOT EXISTS trailers (
        trailer_id INTEGER PRIMARY KEY AUTOINCREMENT,
        number TEXT UNIQUE,
        type TEXT,
        year INTEGER,
        plate TEXT,
        vin TEXT,
        status TEXT DEFAULT 'Active',
        loan_amount REAL DEFAULT 0,
        truck_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # drivers
    cur.execute('''
    CREATE TABLE IF NOT EXISTS drivers (
        driver_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        license_number TEXT,
        phone TEXT,
        email TEXT,
        hire_date DATE,
        status TEXT DEFAULT 'Active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # expenses
    cur.execute('''
    CREATE TABLE IF NOT EXISTS expenses (
        expense_id INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATE NOT NULL,
        category TEXT NOT NULL,
        amount REAL NOT NULL,
        truck_id INTEGER,
        description TEXT,
        location TEXT,
        service_type TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (truck_id) REFERENCES trucks (truck_id)
    )
    ''')
    # income
    cur.execute('''
    CREATE TABLE IF NOT EXISTS income (
        income_id INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATE NOT NULL,
        source TEXT NOT NULL,
        amount REAL NOT NULL,
        truck_id INTEGER,
        description TEXT,
        pickup_date DATE,
        pickup_address TEXT,
        delivery_date DATE,
        delivery_address TEXT,
        job_reference TEXT,
        empty_miles REAL,
        loaded_miles REAL,
        rpm REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (truck_id) REFERENCES trucks (truck_id)
    )
    ''')
    # extra income columns (idempotent)
    for alter_sql in [
        "ALTER TABLE income ADD COLUMN driver_name TEXT",
        "ALTER TABLE income ADD COLUMN broker_number TEXT",
        "ALTER TABLE income ADD COLUMN tonu TEXT DEFAULT 'N'",
        "ALTER TABLE income ADD COLUMN pickup_city TEXT",
        "ALTER TABLE income ADD COLUMN pickup_state TEXT",
        "ALTER TABLE income ADD COLUMN pickup_zip TEXT",
        "ALTER TABLE income ADD COLUMN delivery_city TEXT",
        "ALTER TABLE income ADD COLUMN delivery_state TEXT",
        "ALTER TABLE income ADD COLUMN delivery_zip TEXT",
        "ALTER TABLE income ADD COLUMN stops INTEGER",
        "ALTER TABLE income ADD COLUMN pickup_full_address TEXT",
        "ALTER TABLE income ADD COLUMN delivery_full_address TEXT",
    ]:
        try:
            cur.execute(alter_sql)
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()

# call initial DB creation
init_database()

def ensure_truck_and_trailer_extra_columns():
    conn = get_db_connection()
    cur = conn.cursor()
    # trucks.notes
    try:
        cur.execute("ALTER TABLE trucks ADD COLUMN notes TEXT")
    except sqlite3.OperationalError:
        pass
    # trucks.color (optional if UI references it)
    try:
        cur.execute("ALTER TABLE trucks ADD COLUMN color TEXT")
    except sqlite3.OperationalError:
        pass
    # trucks.owner (optional if UI references it)
    try:
        cur.execute("ALTER TABLE trucks ADD COLUMN owner TEXT")
    except sqlite3.OperationalError:
        pass

    # trailers.trailer_number (your query references it)
    try:
        cur.execute("ALTER TABLE trailers ADD COLUMN trailer_number TEXT")
    except sqlite3.OperationalError:
        pass

    # Also ensure dispatcher link table if your query uses it
    cur.execute("""
        CREATE TABLE IF NOT EXISTS truck_dispatcher_link (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            truck_id INTEGER NOT NULL,
            dispatcher_id INTEGER NOT NULL,
            start_date DATE NOT NULL DEFAULT (DATE('now')),
            end_date DATE,
            FOREIGN KEY (truck_id) REFERENCES trucks(truck_id) ON DELETE CASCADE,
            FOREIGN KEY (dispatcher_id) REFERENCES dispatchers(dispatcher_id) ON DELETE CASCADE
        )
    """)

    # Helpful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tdl_truck ON truck_dispatcher_link(truck_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tdl_dispatcher ON truck_dispatcher_link(dispatcher_id)")

    conn.commit()
    conn.close()

# Call it after init_database()
ensure_truck_and_trailer_extra_columns()

def ensure_income_extra_columns():
    conn = get_db_connection()
    cur = conn.cursor()
    # Time fields
    try:
        cur.execute("ALTER TABLE income ADD COLUMN pickup_time TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE income ADD COLUMN delivery_time TEXT")
    except sqlite3.OperationalError:
        pass

    # Ensure all columns referenced by the Income page exist (idempotent guards)
    for sql in [
        "ALTER TABLE income ADD COLUMN driver_name TEXT",
        "ALTER TABLE income ADD COLUMN broker_number TEXT",
        "ALTER TABLE income ADD COLUMN tonu TEXT DEFAULT 'N'",
        "ALTER TABLE income ADD COLUMN empty_miles REAL",
        "ALTER TABLE income ADD COLUMN loaded_miles REAL",
        "ALTER TABLE income ADD COLUMN rpm REAL",
        "ALTER TABLE income ADD COLUMN pickup_city TEXT",
        "ALTER TABLE income ADD COLUMN pickup_state TEXT",
        "ALTER TABLE income ADD COLUMN pickup_zip TEXT",
        "ALTER TABLE income ADD COLUMN pickup_address TEXT",
        "ALTER TABLE income ADD COLUMN delivery_city TEXT",
        "ALTER TABLE income ADD COLUMN delivery_state TEXT",
        "ALTER TABLE income ADD COLUMN delivery_zip TEXT",
        "ALTER TABLE income ADD COLUMN delivery_address TEXT",
        "ALTER TABLE income ADD COLUMN stops INTEGER",
        "ALTER TABLE income ADD COLUMN pickup_full_address TEXT",
        "ALTER TABLE income ADD COLUMN delivery_full_address TEXT",
        # If you renamed "Description" to "Load number", ensure that column name exists in DB
        # (your UI might still use i.description, so we keep it)
    ]:
        try:
            cur.execute(sql)
        except sqlite3.OperationalError:
            pass

    # Helpful indexes for date-range queries
    try:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_income_date ON income(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_income_truck ON income(truck_id)")
    except Exception:
        pass

    conn.commit()
    conn.close()

# Call this during startup, after init_database()
ensure_income_extra_columns()

def ensure_expenses_fuel_columns():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE expenses ADD COLUMN gallons REAL")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE expenses ADD COLUMN unit_price REAL")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

ensure_expenses_fuel_columns()

# -------------------------
# History tables: loans + assignments
# -------------------------
def init_history_tables():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS loans_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_type TEXT NOT NULL,
        entity_id INTEGER NOT NULL,
        monthly_amount REAL NOT NULL,
        start_date DATE NOT NULL,
        end_date DATE,
        note TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS trailer_truck_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trailer_id INTEGER NOT NULL,
        old_truck_id INTEGER,
        new_truck_id INTEGER,
        start_date DATE NOT NULL,
        end_date DATE,
        note TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS driver_assignment_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        driver_id INTEGER NOT NULL,
        truck_id INTEGER,
        trailer_id INTEGER,
        start_date DATE NOT NULL,
        end_date DATE,
        note TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

def migrate_trailer_history_add_truck_id(conn):
    """
    Ensures trailer_truck_history has a truck_id column, backfills it, and adds indexes.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(trailer_truck_history);")
    cols = [r[1] for r in cur.fetchall()]
    if "truck_id" not in cols:
        cur.execute("ALTER TABLE trailer_truck_history ADD COLUMN truck_id INTEGER;")
    cur.execute("""
        UPDATE trailer_truck_history AS h
        SET truck_id = (
            SELECT trl.truck_id FROM trailers trl WHERE trl.trailer_id = h.trailer_id
        )
        WHERE h.truck_id IS NULL
          AND (h.end_date IS NULL OR h.end_date = '' OR DATE(h.end_date) > DATE('now'))
          AND EXISTS (SELECT 1 FROM trailers trl WHERE trl.trailer_id = h.trailer_id AND trl.truck_id IS NOT NULL);
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tth_trailer_id ON trailer_truck_history(trailer_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tth_truck_id   ON trailer_truck_history(truck_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tth_dates      ON trailer_truck_history(start_date, end_date)")
    conn.commit()

init_history_tables()
# Run one-time migration: add truck_id to trailer history and backfill
try:
    _conn_mig = get_db_connection()
    migrate_trailer_history_add_truck_id(_conn_mig)
    _conn_mig.close()
except Exception as _mig_err:
    try:
        st.warning(f"Trailer history migration check failed: {_mig_err}")
    except Exception:
        pass

# Now that DB helpers and schema exist, create default categories and attachments/indexes
ensure_expense_categories_table()
ensure_default_expense_categories()
ensure_maintenance_category()
ensure_expenses_attachments()

# -------------------------
# Dispatchers tables and link
# -------------------------
def ensure_dispatcher_tables():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dispatchers (
            dispatcher_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            phone TEXT,
            email TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dispatcher_trucks (
            dispatcher_id INTEGER NOT NULL,
            truck_id INTEGER NOT NULL,
            PRIMARY KEY (dispatcher_id, truck_id),
            FOREIGN KEY (dispatcher_id) REFERENCES dispatchers(dispatcher_id) ON DELETE CASCADE,
            FOREIGN KEY (truck_id) REFERENCES trucks(truck_id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()

def ensure_truck_dispatcher_link():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE trucks ADD COLUMN dispatcher_id INTEGER")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

ensure_dispatcher_tables()
ensure_truck_dispatcher_link()

def add_dispatcher(name, phone=None, email=None, notes=None):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO dispatchers (name, phone, email, notes) VALUES (?, ?, ?, ?)",
                (name.strip(), phone, email, notes))
    conn.commit()
    dispatcher_id = cur.lastrowid
    conn.close()
    return dispatcher_id

def update_dispatcher(dispatcher_id, name, phone=None, email=None, notes=None):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE dispatchers SET name=?, phone=?, email=?, notes=? WHERE dispatcher_id=?",
                (name.strip(), phone, email, notes, dispatcher_id))
    conn.commit()
    conn.close()

def delete_dispatcher(dispatcher_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM dispatcher_trucks WHERE dispatcher_id=?", (dispatcher_id,))
    cur.execute("DELETE FROM dispatchers WHERE dispatcher_id=?", (dispatcher_id,))
    conn.commit()
    conn.close()

def get_all_dispatchers():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT dispatcher_id, name, phone, email, notes, created_at FROM dispatchers ORDER BY name")
    rows = cur.fetchall()
    conn.close()
    return [{"dispatcher_id": r[0], "name": r[1], "phone": r[2], "email": r[3], "notes": r[4], "created_at": r[5]} for r in rows]

def get_trucks_for_dispatcher(dispatcher_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT truck_id FROM dispatcher_trucks WHERE dispatcher_id = ?", (dispatcher_id,))
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]

def assign_trucks_to_dispatcher(dispatcher_id, truck_id_list):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM dispatcher_trucks WHERE dispatcher_id = ?", (dispatcher_id,))
    for tid in set([int(t) for t in truck_id_list if t is not None]):
        cur.execute("INSERT OR IGNORE INTO dispatcher_trucks (dispatcher_id, truck_id) VALUES (?, ?)", (dispatcher_id, tid))
    conn.commit()
    conn.close()

# -------------------------
# Safe rerun helper
# -------------------------
def safe_rerun():
    if hasattr(st, "experimental_rerun"):
        try:
            st.experimental_rerun()
            return
        except Exception:
            pass
    st.session_state["_rerun_trigger"] = not st.session_state.get("_rerun_trigger", False)
    st.stop()

# ========== Safe-delete helpers ==========
def safe_delete_all_rows(table_name: str):
    """
    Deletes all rows from table_name even if another table has a bad foreign key.
    Temporarily disables FK checks only inside this connection/transaction.
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = OFF;")
        cur.execute(f"DELETE FROM {table_name};")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            cur.execute("PRAGMA foreign_keys = ON;")
        except Exception:
            pass
        conn.close()

def safe_delete_one(table_name: str, key_col: str, key_val):
    """
    Deletes one row by key. If FK blocks it, retry with FK OFF for this op.
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute(f"DELETE FROM {table_name} WHERE {key_col} = ?;", (key_val,))
        except Exception:
            cur.execute("PRAGMA foreign_keys = OFF;")
            cur.execute(f"DELETE FROM {table_name} WHERE {key_col} = ?;", (key_val,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            cur.execute("PRAGMA foreign_keys = ON;")
        except Exception:
            pass
        conn.close()
# ========================================

# -------------------------
# Preset date range (already above) – kept single definition
# -------------------------

# -------------------------
# Loan proration helpers (per-month prorating)
# -------------------------
def iterate_month_starts(start_date: date, end_date: date):
    y, m = start_date.year, start_date.month
    while (y, m) <= (end_date.year, end_date.month):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1

def prorated_monthly_amount_for_range(monthly_amount: float, start_date: date, end_date: date) -> float:
    if not monthly_amount or monthly_amount == 0:
        return 0.0
    total = 0.0
    for y, m in iterate_month_starts(start_date, end_date):
        days_in_month = calendar.monthrange(y, m)[1]
        month_start = date(y, m, 1)
        month_end = date(y, m, days_in_month)
        seg_start = max(start_date, month_start)
        seg_end = min(end_date, month_end)
        if seg_start > seg_end:
            continue
        covered_days = (seg_end - seg_start).days + 1
        total += (monthly_amount / days_in_month) * covered_days
    return total

# -------------------------
# Loan-history helpers
# -------------------------
def get_loan_history(entity_type: str, entity_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, monthly_amount, start_date, end_date, note
        FROM loans_history
        WHERE entity_type=? AND entity_id=?
        ORDER BY DATE(start_date) ASC
    """, (entity_type, entity_id))
    rows = cur.fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            'id': r[0],
            'monthly_amount': float(r[1]),
            'start_date': to_date(r[2]) if r[2] else None,
            'end_date': to_date(r[3]) if r[3] else None,
            'note': r[4]
        })
    return result

def set_loan_history(entity_type: str, entity_id: int, monthly_amount: float, start_date: date=None, note: str=None):
    if start_date is None:
        start_date = date.today()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, start_date FROM loans_history
        WHERE entity_type=? AND entity_id=? AND end_date IS NULL
        ORDER BY DATE(start_date) DESC LIMIT 1
    """, (entity_type, entity_id))
    prev = cur.fetchone()
    if prev:
        prev_id = prev[0]
        prev_end = start_date - timedelta(days=1)
        cur.execute("UPDATE loans_history SET end_date=? WHERE id=?", (prev_end, prev_id))
    cur.execute("""
        INSERT INTO loans_history (entity_type, entity_id, monthly_amount, start_date, note)
        VALUES (?, ?, ?, ?, ?)
    """, (entity_type, entity_id, monthly_amount, start_date, note))
    conn.commit()
    conn.close()

def get_prorated_loan_for_entity(entity_type: str, entity_id: int, range_start: date, range_end: date) -> float:
    rows = get_loan_history(entity_type, entity_id)
    if not rows:
        conn = get_db_connection()
        cur = conn.cursor()
        if entity_type == 'truck':
            cur.execute("SELECT loan_amount FROM trucks WHERE truck_id=?", (entity_id,))
        else:
            cur.execute("SELECT loan_amount FROM trailers WHERE trailer_id=?", (entity_id,))
        r = cur.fetchone()
        conn.close()
        monthly = float(r[0]) if r and r[0] is not None else 0.0
        return prorated_monthly_amount_for_range(monthly, range_start, range_end)
    total = 0.0
    for row in rows:
        seg_start = max(range_start, row['start_date'])
        seg_end = min(range_end, row['end_date'] or range_end)
        if seg_start > seg_end:
            continue
        total += prorated_monthly_amount_for_range(row['monthly_amount'], seg_start, seg_end)
    return total

def upsert_current_loan(entity_type: str, entity_id: int, monthly_amount: float, start_date: date, end_date: date | None):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, monthly_amount, DATE(start_date) AS s, end_date
        FROM loans_history
        WHERE entity_type = ? AND entity_id = ? AND (end_date IS NULL OR end_date = '')
        ORDER BY DATE(start_date) ASC
        LIMIT 1
    """, (entity_type, entity_id))
    row = cur.fetchone()
    new_amt = float(monthly_amount or 0.0)
    new_s = start_date
    new_e = end_date
    def ds(d):
        return None if d is None else d.isoformat()
    today_d = date.today()
    if row:
        open_id, old_amt, old_s_str, old_e = row
        old_s = date.fromisoformat(old_s_str)
        if new_e:
            cur.execute("UPDATE loans_history SET end_date = ? WHERE id = ?", (ds(new_e), open_id))
            conn.commit()
            conn.close()
            return
        if new_amt <= 0:
            cur.execute("UPDATE loans_history SET end_date = ? WHERE id = ?", (ds(today_d), open_id))
            conn.commit()
            conn.close()
            return
        if new_s > old_s:
            end_prev = new_s - timedelta(days=1)
            cur.execute("UPDATE loans_history SET end_date = ? WHERE id = ?", (ds(end_prev), open_id))
            cur.execute("""
                INSERT INTO loans_history (entity_type, entity_id, monthly_amount, start_date, end_date)
                VALUES (?, ?, ?, ?, NULL)
            """, (entity_type, entity_id, new_amt, ds(new_s)))
            conn.commit()
            conn.close()
            return
        if new_s == old_s:
            if float(old_amt) != new_amt:
                cur.execute("UPDATE loans_history SET monthly_amount = ? WHERE id = ?", (new_amt, open_id))
                conn.commit()
                conn.close()
            else:
                conn.close()
            return
        cur.execute("UPDATE loans_history SET start_date = ?, monthly_amount = ? WHERE id = ?", (ds(new_s), new_amt, open_id))
        conn.commit()
        conn.close()
        return
    if new_amt <= 0:
        conn.close()
        return
    cur.execute("""
        INSERT INTO loans_history (entity_type, entity_id, monthly_amount, start_date, end_date)
        VALUES (?, ?, ?, ?, ?)
    """, (entity_type, entity_id, new_amt, ds(new_s), ds(new_e)))
    conn.commit()
    conn.close()

# -------------------------
# Assignment history helpers
# -------------------------
def record_trailer_assignment(trailer_id: int, new_truck_id: int, start_date: date=None, note: str=None):
    if start_date is None:
        start_date = date.today()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, new_truck_id FROM trailer_truck_history
        WHERE trailer_id=? AND end_date IS NULL
        ORDER BY start_date DESC LIMIT 1
    """, (trailer_id,))
    prev = cur.fetchone()
    old_truck_id = prev[1] if prev else None
    if prev:
        prev_id = prev[0]
        prev_end = start_date - timedelta(days=1)
        cur.execute("UPDATE trailer_truck_history SET end_date=? WHERE id=?", (prev_end, prev_id))
    cur.execute("""
        INSERT INTO trailer_truck_history (trailer_id, old_truck_id, new_truck_id, truck_id, start_date, note)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (trailer_id, old_truck_id, new_truck_id, new_truck_id, start_date, note))
    cur.execute("UPDATE trailers SET truck_id=? WHERE trailer_id=?", (new_truck_id, trailer_id))
    conn.commit()
    conn.close()

def record_driver_assignment(driver_id: int, truck_id: int=None, trailer_id: int=None, start_date: date=None, note: str=None):
    if start_date is None:
        start_date = date.today()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id FROM driver_assignment_history
        WHERE driver_id=? AND end_date IS NULL
        ORDER BY start_date DESC LIMIT 1
    """, (driver_id,))
    prev = cur.fetchone()
    if prev:
        prev_id = prev[0]
        prev_end = start_date - timedelta(days=1)
        cur.execute("UPDATE driver_assignment_history SET end_date=? WHERE id=?", (prev_end, prev_id))
    cur.execute("""
        INSERT INTO driver_assignment_history (driver_id, truck_id, trailer_id, start_date, note)
        VALUES (?, ?, ?, ?, ?)
    """, (driver_id, truck_id, trailer_id, start_date, note))
    if truck_id:
        cur.execute("UPDATE trucks SET driver_id=? WHERE truck_id=?", (driver_id, truck_id))
    conn.commit()
    conn.close()

# -------------------------
# Simple getters
# -------------------------
def get_trucks():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM trucks ORDER BY number", conn)
    conn.close()
    return df

def get_trailers():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM trailers ORDER BY number", conn)
    conn.close()
    return df

def get_drivers():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM drivers ORDER BY name", conn)
    conn.close()
    return df

def get_expenses():
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT e.*, t.number as truck_number 
        FROM expenses e 
        LEFT JOIN trucks t ON e.truck_id = t.truck_id 
        ORDER BY e.date DESC
    """, conn)
    conn.close()
    return df

def get_income():
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT i.*, t.number as truck_number 
        FROM income i 
        LEFT JOIN trucks t ON i.truck_id = t.truck_id 
        ORDER BY i.date DESC
    """, conn)
    conn.close()
    return df

def delete_record(table, id_column, record_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {table} WHERE {id_column} = ?", (record_id,))
    conn.commit()
    conn.close()

def get_current_dispatcher_for_truck(conn, truck_id: int) -> str:
    """
    Returns the current/active dispatcher name for a truck.
    Priority:
      1) trucks.dispatcher_id if set -> dispatchers.name
      2) fallback to legacy mapping table dispatcher_trucks if present
    """
    try:
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT d.name
                FROM trucks t
                LEFT JOIN dispatchers d ON d.dispatcher_id = t.dispatcher_id
                WHERE t.truck_id = ?
            """, (truck_id,))
            r = cur.fetchone()
            if r and r[0]:
                return r[0]
        except Exception:
            pass
        try:
            cur.execute("""
                SELECT d.name
                FROM dispatcher_trucks dt
                JOIN dispatchers d ON d.dispatcher_id = dt.dispatcher_id
                WHERE dt.truck_id = ?
                ORDER BY d.name ASC
                LIMIT 1
            """, (truck_id,))
            r2 = cur.fetchone()
            if r2 and r2[0]:
                return r2[0]
        except Exception:
            pass
        return ""
    except Exception:
        return ""

def safe_read_sql(query, conn, params=None):
    """
    Execute a SQL query safely and return a DataFrame.
    - Returns empty DataFrame on any error.
    """
    try:
        if params is None:
            return pd.read_sql_query(query, conn)
        else:
            return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        try:
            st.warning(f"Query failed (handled): {e}")
        except Exception:
            pass
        return pd.DataFrame()

# -------------------------
# Aggregated loan totals per truck (truck + linked trailers) using history
# -------------------------
def get_truck_loans_in_range(start_date: date, end_date: date):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT truck_id, number FROM trucks")
    trucks = cur.fetchall()
    cur.execute("SELECT trailer_id, number, truck_id FROM trailers")
    trailers = cur.fetchall()
    conn.close()
    trailer_map_by_truck = {}
    for trailer_id, tr_number, tr_truck_id in trailers:
        if tr_truck_id:
            trailer_map_by_truck.setdefault(tr_truck_id, []).append(trailer_id)
    result = {}
    for truck_id, number in trucks:
        truck_prorated = get_prorated_loan_for_entity('truck', truck_id, start_date, end_date)
        trailer_prorated_total = 0.0
        for tr_id in trailer_map_by_truck.get(truck_id, []):
            trailer_prorated_total += get_prorated_loan_for_entity('trailer', tr_id, start_date, end_date)
        result[number] = {
            'truck_id': truck_id,
            'truck_prorated_loan': truck_prorated,
            'trailer_prorated_loan': trailer_prorated_total,
            'total_loan': truck_prorated + trailer_prorated_total
        }
    return result

# -------------------------
# Seed existing loans (one-time safe)
# -------------------------
def seed_existing_loans_start():
    conn = get_db_connection()
    cur = conn.cursor()
    # trucks
    cur.execute("SELECT truck_id, loan_amount FROM trucks WHERE loan_amount IS NOT NULL AND loan_amount>0")
    for truck_id, loan in cur.fetchall():
        cur.execute("SELECT COUNT(*) FROM loans_history WHERE entity_type='truck' AND entity_id=?", (truck_id,))
        if cur.fetchone()[0] == 0:
            cur.execute("""
                INSERT INTO loans_history (entity_type, entity_id, monthly_amount, start_date, note)
                VALUES (?, ?, ?, ?, ?)
            """, ('truck', truck_id, loan, '2000-01-01', 'seed from existing data'))
    # trailers
    cur.execute("SELECT trailer_id, loan_amount FROM trailers WHERE loan_amount IS NOT NULL AND loan_amount>0")
    for trailer_id, loan in cur.fetchall():
        cur.execute("SELECT COUNT(*) FROM loans_history WHERE entity_type='trailer' AND entity_id=?", (trailer_id,))
        if cur.fetchone()[0] == 0:
            cur.execute("""
                INSERT INTO loans_history (entity_type, entity_id, monthly_amount, start_date, note)
                VALUES (?, ?, ?, ?, ?)
            """, ('trailer', trailer_id, loan, '2000-01-01', 'seed from existing data'))
    conn.commit()
    conn.close()

seed_existing_loans_start()

# -------------------------
# Streamlit setup
# -------------------------
st.set_page_config(page_title="Fleet Management System", page_icon="🚛", layout="wide")
st.title("🚛 Fleet Management System")
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        text-align: left;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #2C2C2C;
        color: white;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        text-align: center;
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None

def login_page():
    """Display login page"""
    st.title("🚛 Fleet Management System")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("🔐 Login")
        
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        update_last_login(user["user_id"])
                        log_session_action(user["user_id"], user["username"], "login")
                        st.success(f"Welcome, {user['full_name'] or user['username']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                        log_session_action(None, username, "failed_login_attempt")
        
        st.info("**Default credentials:**\n\nUsername: `admin`\nPassword: `admin123`\n\n⚠️ Change password immediately after first login!")

def logout():
    """Logout current user"""
    if st.session_state.user:
        log_session_action(st.session_state.user["user_id"], st.session_state.user["username"], "logout")
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()

# Check authentication
if not st.session_state.authenticated:
    login_page()
    st.stop()

# User is authenticated - show user info in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown(f"👤 **{st.session_state.user['full_name'] or st.session_state.user['username']}**")
    st.caption(f"Role: {st.session_state.user['role'].capitalize()}")
    
    if st.button("🚪 Logout", use_container_width=True):
        logout()
    
    # ADD THIS NEW SECTION:
    with st.expander("🔑 Change Password"):
        with st.form("change_password_form"):
            current_pw = st.text_input("Current Password", type="password", key="current_pw")
            new_pw = st.text_input("New Password", type="password", key="new_pw")
            confirm_pw = st.text_input("Confirm New Password", type="password", key="confirm_pw")
            
            if st.form_submit_button("Change Password"):
                if not current_pw or not new_pw or not confirm_pw:
                    st.error("All fields are required")
                elif new_pw != confirm_pw:
                    st.error("New passwords don't match")
                elif len(new_pw) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    # Verify current password
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("SELECT password_hash FROM users WHERE user_id = ?", (st.session_state.user["user_id"],))
                    result = cur.fetchone()
                    
                    if result and verify_password(current_pw, result[0]):
                        new_hash = hash_password(new_pw)
                        conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", 
                                   (new_hash, st.session_state.user["user_id"]))
                        conn.commit()
                        conn.close()
                        log_session_action(st.session_state.user["user_id"], st.session_state.user["username"], "changed_password")
                        st.success("Password changed successfully!")
                    else:
                        st.error("Current password is incorrect")
                        conn.close()
    
    st.markdown("---")

# Sidebar navigation
st.sidebar.title("📋 Navigation")

# Get pages user can access
pages = get_user_pages()

# Keep the selected page in session state to persist selection
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Dashboard" if "Dashboard" in pages else (pages[0] if pages else "Dashboard")

# If current page is not in available pages, reset to first available
if st.session_state["current_page"] not in pages:
    st.session_state["current_page"] = pages[0] if pages else "Dashboard"

# Create a vertical list of buttons
for p in pages:
    # Highlight currently active page by color / emoji
    if st.sidebar.button(
        f"➡️ {p}" if st.session_state["current_page"] != p else f"✅ {p}",
        key=f"nav_{p}"
    ):
        st.session_state["current_page"] = p
        st.rerun()

page = st.session_state["current_page"]

# Check if user has access to current page
if not can_access_page(page):
    st.error("⛔ Access Denied: You don't have permission to view this page")
    st.stop()

# -------------------------
# Dashboard
# -------------------------
if page == "Dashboard":
    st.header("📊 Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    trucks_df = get_trucks()
    trailers_df = get_trailers()
    drivers_df = get_drivers()
    expenses_df = get_expenses()
    income_df = get_income()

    with col1:
        st.metric("Total Trucks", len(trucks_df))
    with col2:
        st.metric("Total Trailers", len(trailers_df))
    with col3:
        st.metric("Total Drivers", len(drivers_df))
    with col4:
        total_expenses = expenses_df['amount'].sum() if not expenses_df.empty else 0.0
        st.metric("Total Expenses", f"${total_expenses:,.2f}")

    col1, col2 = st.columns(2)
    with col1:
        if not expenses_df.empty:
            expense_by_category = expenses_df.groupby('category')['amount'].sum().reset_index()
            fig = px.pie(expense_by_category, values='amount', names='category', title='Expenses by Category')
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if not income_df.empty:
            income_by_truck = income_df.groupby('truck_number')['amount'].sum().reset_index()
            fig = px.bar(income_by_truck, x='truck_number', y='amount', title='Income by Truck')
            st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Trucks Management
# -------------------------
elif page == "Trucks":
    st.header("🚛 Trucks Management")
    tab1, tab2 = st.tabs(["View Trucks", "Add New Truck"])

    # Helper: current trailer per truck, based on trailers.truck_id
    def get_trucks_with_names_by_trailer_link():
        conn = get_db_connection()
        try:
            sql = """
            SELECT
                t.truck_id,
                t.number AS truck_number,
                t.make,
                t.model,
                t.year,
                t.vin,
                t.plate,
                t.status,
                t.notes,
                d.name AS dispatcher_name,
                tr.trailer_number AS linked_trailer_number
            FROM trucks t
            LEFT JOIN truck_dispatcher_link tdl
                   ON tdl.truck_id = t.truck_id
                  AND (tdl.end_date IS NULL OR tdl.end_date = '')
            LEFT JOIN dispatchers d
                   ON d.dispatcher_id = tdl.dispatcher_id
            LEFT JOIN trailers tr
                   ON tr.truck_id = t.truck_id
            ORDER BY t.truck_id DESC;
            """
            df = pd.read_sql_query(sql, conn)
            return df
        except Exception as e:
            import traceback
            st.error(f"Trucks query failed: {e}")
            st.code(sql, language="sql")
            st.text(traceback.format_exc())
            return pd.DataFrame()
        finally:
            conn.close()

    # Helper: list of trailers for selection
    def get_all_trailers():
        conn = get_db_connection()
        tdf = pd.read_sql_query(
            """
            SELECT trailer_id, number, truck_id
            FROM trailers
            ORDER BY number
            """,
            conn,
        )
        conn.close()
        return tdf

    with tab1:
        trucks_df = get_trucks_with_names_by_trailer_link()

        if not trucks_df.empty:
            display_df = trucks_df[[
                "truck_id",
                "truck_number",
                "make",
                "model",
                "year",
                "plate",
                "vin",
                "status",
                "trailer_number",
                "driver_name",
                "dispatcher_name",
                "loan_amount",
                "loan_start_date",
                "loan_term_months",
                "created_at",
            ]].copy()

            display_df.rename(columns={
                "truck_number": "Truck #",
                "trailer_number": "Trailer",
                "driver_name": "Driver",
                "dispatcher_name": "Dispatcher",
                "loan_amount": "Loan Amount",
                "loan_start_date": "Loan Start",
                "loan_term_months": "Loan Term (months)",
                "created_at": "Created At",
            }, inplace=True)

            display_df["Loan Amount"] = display_df["Loan Amount"].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00"
            )

            st.dataframe(display_df, use_container_width=True)

            st.subheader("Edit or Delete Truck")
            selected_truck = st.selectbox(
                "Select Truck",
                options=trucks_df['truck_id'].tolist(),
                format_func=lambda x: f"{trucks_df.loc[trucks_df['truck_id'] == x, 'truck_number'].iloc[0]}",
            )

            if selected_truck:
                row = trucks_df[trucks_df['truck_id'] == selected_truck].iloc[0]

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✏️ Edit Truck", key="edit_truck"):
                        st.session_state.editing_truck = selected_truck
                with col2:
                    if st.button("🗑️ Delete Truck", key="delete_truck"):
                        delete_record("trucks", "truck_id", selected_truck)
                        st.success("Truck deleted successfully!")
                        safe_rerun()

                if st.session_state.get('editing_truck') == selected_truck:
                    st.subheader("Edit Truck Details")

                    # Get raw current IDs for selectbox defaults
                    conn_pref_ids = get_db_connection()
                    raw_ids = pd.read_sql_query(
                        "SELECT driver_id, dispatcher_id, loan_amount, year FROM trucks WHERE truck_id = ?",
                        conn_pref_ids,
                        params=(int(selected_truck),)
                    )
                    conn_pref_ids.close()
                    raw_driver_id = int(raw_ids.iloc[0]["driver_id"]) if not raw_ids.empty and pd.notna(raw_ids.iloc[0]["driver_id"]) else None
                    raw_dispatcher_id = int(raw_ids.iloc[0]["dispatcher_id"]) if not raw_ids.empty and pd.notna(raw_ids.iloc[0]["dispatcher_id"]) else None
                    raw_loan_amount = float(raw_ids.iloc[0]["loan_amount"]) if not raw_ids.empty and pd.notna(raw_ids.iloc[0]["loan_amount"]) else 0.0
                    raw_year = int(raw_ids.iloc[0]["year"]) if not raw_ids.empty and pd.notna(raw_ids.iloc[0]["year"]) else 2020

                    # Current trailer by link (from joined row)
                    current_trailer_id = int(row["trailer_id"]) if "trailer_id" in row and pd.notna(row["trailer_id"]) else None

                    with st.form(f"edit_truck_form_{selected_truck}"):
                        new_number = st.text_input("Truck Number", value=row['truck_number'] or "")
                        new_make = st.text_input("Make", value=row['make'] or "")
                        new_model = st.text_input("Model", value=row['model'] or "")
                        new_year = st.number_input("Year", value=raw_year, min_value=1900, max_value=2100)
                        new_plate = st.text_input("Plate Number", value=row['plate'] or "")
                        new_vin = st.text_input("VIN", value=row['vin'] or "")
                        new_status = st.selectbox(
                            "Status",
                            ["Active", "Inactive", "Maintenance"],
                            index=["Active", "Inactive", "Maintenance"].index(row['status'])
                                  if row['status'] in ["Active", "Inactive", "Maintenance"]
                                  else 0,
                        )
                        new_loan = st.number_input("Loan Amount (monthly)", value=raw_loan_amount, min_value=0.0)

                        # Loan history widgets (unchanged)
                        conn_pref = get_db_connection()
                        df_open = pd.read_sql_query(
                            """
                            SELECT id, monthly_amount, DATE(start_date) AS s
                            FROM loans_history
                            WHERE entity_type='truck' AND entity_id=? AND (end_date IS NULL OR end_date = '')
                            ORDER BY DATE(start_date) DESC
                            LIMIT 1
                            """,
                            conn_pref,
                            params=(int(selected_truck),),
                        )
                        conn_pref.close()

                        today_d = date.today()
                        pref_start = today_d
                        if df_open is not None and not df_open.empty:
                            raw_s = str(df_open.iloc[0].get("s")) if "s" in df_open.columns else None
                            try:
                                if raw_s and raw_s != "NaT":
                                    pref_start = date.fromisoformat(raw_s)
                            except Exception:
                                pref_start = today_d

                        pref_start = max(pref_start, COMPANY_START)
                        max_allowed = date(2100, 12, 31)

                        loan_start_input = st.date_input(
                            "Truck Loan Start Date",
                            value=pref_start,
                            min_value=COMPANY_START,
                            max_value=max_allowed,
                            key=f"truck_loan_start_{selected_truck}",
                        )

                        end_checked = st.checkbox(
                            "End this truck loan",
                            value=False,
                            key=f"truck_loan_end_chk_{selected_truck}",
                        )

                        loan_end_input = None
                        if end_checked:
                            default_end = max(loan_start_input, today_d)
                            loan_end_input = st.date_input(
                                "Truck Loan End Date",
                                value=default_end,
                                min_value=loan_start_input,
                                max_value=max_allowed,
                                key=f"truck_loan_end_{selected_truck}",
                            )

                        if loan_start_input < COMPANY_START:
                            loan_start_input = COMPANY_START
                        if loan_end_input and loan_end_input < loan_start_input:
                            st.warning("Loan end date cannot be before start date.")
                            loan_end_input = loan_start_input

                        # Driver selector
                        drivers_df = get_drivers()
                        driver_options = ["No Driver Assigned"] + [
                            f"{r['name']} ({r['license_number'] or ''})" for _, r in drivers_df.iterrows()
                        ]
                        driver_ids = [None] + drivers_df['driver_id'].tolist()
                        current_driver_idx = 0
                        if raw_driver_id:
                            try:
                                current_driver_idx = driver_ids.index(raw_driver_id)
                            except ValueError:
                                current_driver_idx = 0
                        selected_driver_idx = st.selectbox(
                            "Assigned Driver",
                            range(len(driver_options)),
                            format_func=lambda x: driver_options[x],
                            index=current_driver_idx,
                        )
                        new_driver_id = driver_ids[selected_driver_idx]

                        # Dispatcher selector
                        dispatchers = get_all_dispatchers()
                        dispatcher_options = ["No Dispatcher Assigned"] + [d['name'] for d in dispatchers]
                        dispatcher_ids = [None] + [d['dispatcher_id'] for d in dispatchers]
                        current_dispatcher_idx = 0
                        if raw_dispatcher_id:
                            try:
                                current_dispatcher_idx = dispatcher_ids.index(raw_dispatcher_id)
                            except ValueError:
                                current_dispatcher_idx = 0
                        selected_dispatcher_idx = st.selectbox(
                            "Assigned Dispatcher",
                            range(len(dispatcher_options)),
                            format_func=lambda x: dispatcher_options[x],
                            index=current_dispatcher_idx,
                        )
                        new_dispatcher_id = dispatcher_ids[selected_dispatcher_idx]

                        # Trailer selector (now editable): based on trailers.truck_id link
                        trailers_df = get_all_trailers()
                        trailer_options = ["No Trailer Assigned"] + trailers_df['number'].astype(str).tolist()
                        trailer_ids = [None] + trailers_df['trailer_id'].tolist()
                        current_trailer_idx = 0
                        if current_trailer_id:
                            try:
                                current_trailer_idx = trailer_ids.index(current_trailer_id)
                            except ValueError:
                                current_trailer_idx = 0
                        selected_trailer_idx = st.selectbox(
                            "Assigned Trailer",
                            range(len(trailer_options)),
                            format_func=lambda x: trailer_options[x],
                            index=current_trailer_idx,
                        )
                        new_trailer_id = trailer_ids[selected_trailer_idx]

                        col1, col2 = st.columns(2)
                        with col1:
                            saved = st.form_submit_button("💾 Save Changes")
                        with col2:
                            cancelled = st.form_submit_button("❌ Cancel")

                    # Handle form actions
                    if 'editing_truck' in st.session_state and st.session_state.editing_truck == selected_truck:
                        if cancelled:
                            del st.session_state.editing_truck
                            safe_rerun()

                        if saved:
                            try:
                                # Update truck core fields
                                conn = get_db_connection()
                                cur = conn.cursor()
                                cur.execute(
                                    "SELECT loan_amount, driver_id FROM trucks WHERE truck_id=?",
                                    (selected_truck,),
                                )
                                prev = cur.fetchone()
                                prev_driver_id = prev[1] if prev else None

                                cur.execute(
                                    """
                                    UPDATE trucks
                                    SET number=?, make=?, model=?, year=?, plate=?, vin=?, status=?, loan_amount=?, driver_id=?, dispatcher_id=?
                                    WHERE truck_id=?
                                    """,
                                    (
                                        new_number,
                                        new_make,
                                        new_model,
                                        new_year,
                                        new_plate,
                                        new_vin,
                                        new_status,
                                        new_loan,
                                        new_driver_id,
                                        new_dispatcher_id,
                                        selected_truck,
                                    ),
                                )
                                conn.commit()

                                # Update trailer linkage in trailers table
                                # 1) Unassign this truck from any trailers if "No Trailer Assigned" chosen
                                if new_trailer_id is None:
                                    cur.execute("UPDATE trailers SET truck_id = NULL WHERE truck_id = ?", (selected_truck,))
                                else:
                                    # 2) Ensure selected trailer points to this truck
                                    #    First, clear it from any previous truck
                                    cur.execute("UPDATE trailers SET truck_id = NULL WHERE trailer_id = ?", (new_trailer_id,))
                                    #    Then link to this truck
                                    cur.execute("UPDATE trailers SET truck_id = ? WHERE trailer_id = ?", (selected_truck, new_trailer_id))
                                    # 3) Also ensure no other trailers remain linked to this truck (single trailer per truck policy)
                                    cur.execute("UPDATE trailers SET truck_id = NULL WHERE truck_id = ? AND trailer_id <> ?", (selected_truck, new_trailer_id))
                                conn.commit()
                                conn.close()

                                # Loan history upsert
                                try:
                                    upsert_current_loan(
                                        "truck",
                                        int(selected_truck),
                                        float(new_loan or 0.0),
                                        loan_start_input,
                                        loan_end_input,
                                    )
                                except Exception as e:
                                    st.warning(f"Truck loan history update warning: {e}")

                                # Driver assignment history
                                if new_driver_id and new_driver_id != prev_driver_id:
                                    record_driver_assignment(
                                        new_driver_id,
                                        truck_id=selected_truck,
                                        start_date=date.today(),
                                        note="Assigned via UI",
                                    )

                                st.success("Truck updated.")
                                del st.session_state.editing_truck
                                safe_rerun()
                            except Exception as e:
                                st.error(f"Truck save failed: {e}")
        else:
            st.info("No trucks found. Add some trucks to get started!")

    with tab2:
        st.subheader("Add New Truck")
        with st.form("add_truck"):
            number = st.text_input("Truck Number*", placeholder="e.g., T001")
            make = st.text_input("Make")
            model = st.text_input("Model")
            year = st.number_input("Year", min_value=1900, max_value=2100, value=2020)
            plate = st.text_input("Plate Number")
            vin = st.text_input("VIN")
            status = st.selectbox("Status", ["Active", "Inactive", "Maintenance"])
            loan_amount = st.number_input("Loan Amount (monthly)", min_value=0.0, value=0.0)

            # driver selection
            drivers_df = get_drivers()
            driver_options = ["No Driver Assigned"] + [
                f"{r['name']} ({r['license_number'] or ''})" for _, r in drivers_df.iterrows()
            ]
            driver_ids = [None] + drivers_df['driver_id'].tolist()
            selected_driver_idx = st.selectbox(
                "Assigned Driver", range(len(driver_options)), format_func=lambda x: driver_options[x]
            )
            driver_id = driver_ids[selected_driver_idx]

            # dispatcher selection
            dispatchers = get_all_dispatchers()
            dispatcher_options = ["No Dispatcher Assigned"] + [d['name'] for d in dispatchers]
            dispatcher_ids = [None] + [d['dispatcher_id'] for d in dispatchers]
            selected_dispatcher_idx = st.selectbox(
                "Assigned Dispatcher", range(len(dispatcher_options)), format_func=lambda x: dispatcher_options[x]
            )
            dispatcher_id = dispatcher_ids[selected_dispatcher_idx]

            if st.form_submit_button("Add Truck"):
                if not number:
                    st.error("Truck number is required.")
                else:
                    try:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        cur.execute(
                            """
                            INSERT INTO trucks (number, make, model, year, plate, vin, status, loan_amount, driver_id, dispatcher_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (number, make, model, year, plate, vin, status, loan_amount, driver_id, dispatcher_id),
                        )
                        truck_id = cur.lastrowid
                        conn.commit()
                        conn.close()

                        if loan_amount and loan_amount > 0:
                            try:
                                upsert_current_loan(
                                    "truck",
                                    int(truck_id),
                                    float(loan_amount),
                                    date.today(),
                                    None,
                                )
                            except Exception as e:
                                st.warning(f"Initial loan history warning: {e}")

                        if driver_id:
                            record_driver_assignment(
                                driver_id, truck_id=truck_id, start_date=date.today(), note="Assigned on create"
                            )

                        st.success("Truck added successfully!")
                        safe_rerun()
                    except sqlite3.IntegrityError:
                        st.error("Truck number already exists.")

# -------------------------
# Trailers Management
# -------------------------
elif page == "Trailers":
    st.header("🚚 Trailers Management")
    tab1, tab2 = st.tabs(["View Trailers", "Add New Trailer"])

    with tab1:
        trailers_df = get_trailers()
        if not trailers_df.empty:
            st.dataframe(trailers_df, use_container_width=True)
            st.subheader("Edit or Delete Trailer")

            selected_trailer = st.selectbox(
                "Select Trailer",
                options=trailers_df['trailer_id'].tolist(),
                format_func=lambda x: f"{trailers_df[trailers_df['trailer_id'] == x]['number'].iloc[0]}",
            )

            if selected_trailer:
                trailer_data = trailers_df[trailers_df['trailer_id'] == selected_trailer].iloc[0]
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✏️ Edit Trailer", key="edit_trailer"):
                        st.session_state.editing_trailer = selected_trailer
                with col2:
                    if st.button("🗑️ Delete Trailer", key="delete_trailer"):
                        delete_record("trailers", "trailer_id", selected_trailer)
                        st.success("Trailer deleted successfully!")
                        safe_rerun()

                if st.session_state.get('editing_trailer') == selected_trailer:
                    st.subheader("Edit Trailer Details")
                    with st.form(f"edit_trailer_form_{selected_trailer}"):
                        new_number = st.text_input("Trailer Number", value=trailer_data['number'] or "")
                        new_type = st.text_input("Type", value=trailer_data['type'] or "")
                        new_year = st.number_input(
                            "Year",
                            value=int(trailer_data['year']) if trailer_data['year'] else 2020,
                            min_value=1900,
                            max_value=2100,
                        )
                        new_plate = st.text_input("Plate", value=trailer_data['plate'] or "")
                        new_vin = st.text_input("VIN", value=trailer_data['vin'] or "")
                        new_status = st.selectbox(
                            "Status",
                            ["Active", "Inactive", "Maintenance"],
                            index=["Active", "Inactive", "Maintenance"].index(trailer_data['status'])
                                  if trailer_data['status'] in ["Active", "Inactive", "Maintenance"]
                                  else 0,
                        )
                        new_loan = st.number_input(
                            "Loan Amount (monthly)",
                            value=float(trailer_data['loan_amount']) if trailer_data['loan_amount'] else 0.0,
                            min_value=0.0,
                        )

                        # --- Trailer loan start/end controls ---
                        conn_pref_tr = get_db_connection()
                        df_open_tr = pd.read_sql_query(
                            """
                            SELECT id, monthly_amount, DATE(start_date) AS s
                            FROM loans_history
                            WHERE entity_type='trailer' AND entity_id=? AND (end_date IS NULL OR end_date = '')
                            ORDER BY DATE(start_date) DESC
                            LIMIT 1
                            """,
                            conn_pref_tr,
                            params=(int(selected_trailer),),
                        )
                        conn_pref_tr.close()

                        # Safe prefill
                        today_d = date.today()
                        pref_start_tr = today_d
                        if df_open_tr is not None and not df_open_tr.empty:
                            raw_s_tr = str(df_open_tr.iloc[0].get("s")) if "s" in df_open_tr.columns else None
                            try:
                                if raw_s_tr and raw_s_tr != "NaT":
                                    pref_start_tr = date.fromisoformat(raw_s_tr)
                            except Exception:
                                pref_start_tr = today_d

                        pref_start_tr = max(pref_start_tr, COMPANY_START)
                        max_allowed = date(2100, 12, 31)

                        loan_start_input_tr = st.date_input(
                            "Trailer Loan Start Date",
                            value=pref_start_tr,
                            min_value=COMPANY_START,
                            max_value=max_allowed,
                            key=f"trailer_loan_start_{selected_trailer}",
                        )

                        end_checked_tr = st.checkbox(
                            "End this trailer loan",
                            value=False,
                            key=f"trailer_loan_end_chk_{selected_trailer}",
                        )

                        loan_end_input_tr = None
                        if end_checked_tr:
                            default_end_tr = max(loan_start_input_tr, today_d)
                            loan_end_input_tr = st.date_input(
                                "Trailer Loan End Date",
                                value=default_end_tr,
                                min_value=loan_start_input_tr,
                                max_value=max_allowed,
                                key=f"trailer_loan_end_{selected_trailer}",
                            )

                        # Defensive clamps
                        if loan_start_input_tr < COMPANY_START:
                            loan_start_input_tr = COMPANY_START
                        if loan_end_input_tr and loan_end_input_tr < loan_start_input_tr:
                            st.warning("Loan end date cannot be before start date.")
                            loan_end_input_tr = loan_start_input_tr

                        # Truck assignment
                        trucks_df = get_trucks()
                        truck_options = ["No Truck Assigned"] + [
                            f"{r['number']} - {r['make'] or ''} {r['model'] or ''}".strip()
                            for _, r in trucks_df.iterrows()
                        ]
                        truck_ids = [None] + trucks_df['truck_id'].tolist()
                        current_truck_idx = 0
                        if trailer_data['truck_id']:
                            try:
                                current_truck_idx = truck_ids.index(trailer_data['truck_id'])
                            except ValueError:
                                current_truck_idx = 0
                        selected_truck_idx = st.selectbox(
                            "Assigned Truck",
                            range(len(truck_options)),
                            format_func=lambda x: truck_options[x],
                            index=current_truck_idx,
                        )
                        new_truck_id = truck_ids[selected_truck_idx]

                        col1, col2 = st.columns(2)
                        with col1:
                            saved = st.form_submit_button("💾 Save Changes")
                        with col2:
                            cancelled = st.form_submit_button("❌ Cancel")

                    # Handle form actions
                    if 'editing_trailer' in st.session_state and st.session_state.editing_trailer == selected_trailer:
                        if cancelled:
                            del st.session_state.editing_trailer
                            safe_rerun()

                        if saved:
                            try:
                                # Update trailer row
                                conn = get_db_connection()
                                cur = conn.cursor()
                                cur.execute(
                                    "SELECT loan_amount, truck_id FROM trailers WHERE trailer_id=?",
                                    (selected_trailer,),
                                )
                                prev = cur.fetchone()
                                prev_loan = float(prev[0]) if prev and prev[0] is not None else 0.0
                                prev_truck_id = prev[1] if prev else None

                                cur.execute(
                                    """
                                    UPDATE trailers
                                    SET number=?, type=?, year=?, plate=?, vin=?, status=?, loan_amount=?, truck_id=?
                                    WHERE trailer_id=?
                                    """,
                                    (
                                        new_number,
                                        new_type,
                                        new_year,
                                        new_plate,
                                        new_vin,
                                        new_status,
                                        new_loan,
                                        new_truck_id,
                                        selected_trailer,
                                    ),
                                )
                                conn.commit()
                                conn.close()

                                # Always upsert loan interval so date-only changes persist
                                try:
                                    upsert_current_loan(
                                        "trailer",
                                        int(selected_trailer),
                                        float(new_loan or 0.0),
                                        loan_start_input_tr,
                                        loan_end_input_tr,
                                    )
                                except Exception as e:
                                    st.warning(f"Trailer loan history update warning: {e}")

                                # Trailer assignment history
                                if new_truck_id != prev_truck_id:
                                    record_trailer_assignment(
                                        selected_trailer,
                                        new_truck_id,
                                        start_date=date.today(),
                                        note="Assigned via UI",
                                    )

                                # Requery to confirm
                                try:
                                    conn_chk = get_db_connection()
                                    df_chk = pd.read_sql_query(
                                        """
                                        SELECT DATE(start_date) AS s, monthly_amount,
                                               CASE WHEN end_date IS NULL OR end_date='' THEN NULL ELSE DATE(end_date) END AS e
                                        FROM loans_history
                                        WHERE entity_type='trailer' AND entity_id=?
                                        ORDER BY (CASE WHEN e IS NULL THEN 0 ELSE 1 END), DATE(s) DESC
                                        LIMIT 1
                                        """,
                                        conn_chk,
                                        params=(int(selected_trailer),),
                                    )
                                    conn_chk.close()
                                    if df_chk is not None and not df_chk.empty:
                                        st.success(
                                            f"Trailer updated. Current loan -> start={df_chk.iloc[0]['s']}, "
                                            f"amount={df_chk.iloc[0]['monthly_amount']}, end={df_chk.iloc[0]['e']}"
                                        )
                                    else:
                                        st.success("Trailer updated.")
                                except Exception as e2:
                                    st.success("Trailer updated.")
                                    st.info(f"(Post-save check issue: {e2})")

                                # Exit edit mode
                                del st.session_state.editing_trailer
                                safe_rerun()
                            except Exception as e:
                                st.error(f"Trailer save failed: {e}")
        else:
            st.info("No trailers found. Add some trailers to get started!")

    with tab2:
        st.subheader("Add New Trailer")
        with st.form("add_trailer"):
            number = st.text_input("Trailer Number*", placeholder="e.g., TR001")
            trailer_type = st.text_input("Type")
            year = st.number_input("Year", min_value=1900, max_value=2100, value=2020)
            plate = st.text_input("Plate Number")
            vin = st.text_input("VIN")
            status = st.selectbox("Status", ["Active", "Inactive", "Maintenance"])
            loan_amount = st.number_input("Loan Amount (monthly)", min_value=0.0, value=0.0)

            trucks_df = get_trucks()
            truck_options = ["No Truck Assigned"] + [
                f"{r['number']} - {r['make'] or ''} {r['model'] or ''}".strip() for _, r in trucks_df.iterrows()
            ]
            truck_ids = [None] + trucks_df['truck_id'].tolist()
            selected_truck_idx = st.selectbox(
                "Assigned Truck", range(len(truck_options)), format_func=lambda x: truck_options[x]
            )
            truck_id = truck_ids[selected_truck_idx]

            if st.form_submit_button("Add Trailer"):
                if not number:
                    st.error("Trailer number is required.")
                else:
                    try:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        cur.execute(
                            """
                            INSERT INTO trailers (number, type, year, plate, vin, status, loan_amount, truck_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (number, trailer_type, year, plate, vin, status, loan_amount, truck_id),
                        )
                        trailer_id = cur.lastrowid
                        conn.commit()
                        conn.close()

                        if loan_amount and loan_amount > 0:
                            # Start loan today by default on create
                            try:
                                upsert_current_loan(
                                    "trailer",
                                    int(trailer_id),
                                    float(loan_amount),
                                    date.today(),
                                    None,
                                )
                            except Exception as e:
                                st.warning(f"Initial trailer loan history warning: {e}")

                        if truck_id:
                            record_trailer_assignment(
                                trailer_id, truck_id, start_date=date.today(), note="Assigned on create"
                            )

                        st.success("Trailer added successfully!")
                        safe_rerun()
                    except sqlite3.IntegrityError:
                        st.error("Trailer number already exists.")

# -------------------------
# Drivers Management
# -------------------------
elif page == "Drivers":
    st.header("👨‍💼 Drivers Management")
    tab1, tab2 = st.tabs(["View Drivers", "Add New Driver"])
    with tab1:
        drivers_df = get_drivers()
        if not drivers_df.empty:
            st.dataframe(drivers_df, use_container_width=True)
            st.subheader("Edit or Delete Driver")
            selected_driver = st.selectbox("Select Driver", options=drivers_df['driver_id'].tolist(),
                                           format_func=lambda x: drivers_df[drivers_df['driver_id']==x]['name'].iloc[0])
            if selected_driver:
                driver_data = drivers_df[drivers_df['driver_id'] == selected_driver].iloc[0]
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✏️ Edit Driver", key="edit_driver"):
                        st.session_state.editing_driver = selected_driver
                with col2:
                    if st.button("🗑️ Delete Driver", key="delete_driver"):
                        safe_delete_one("drivers", "driver_id", selected_driver)
                        st.success("Driver deleted successfully!")
                        safe_rerun()
                if st.session_state.get('editing_driver') == selected_driver:
                    with st.form("edit_driver_form"):
                        new_name = st.text_input("Name", value=driver_data['name'])
                        new_license = st.text_input("License Number", value=driver_data['license_number'] or "")
                        new_phone = st.text_input("Phone", value=driver_data['phone'] or "")
                        new_email = st.text_input("Email", value=driver_data['email'] or "")
                        new_hire_date = st.date_input(
                            "Hire Date",
                            value=to_date(driver_data.get('hire_date')) or date.today()
                        )
                        new_status = st.selectbox("Status", ["Active", "Inactive"], index=["Active", "Inactive"].index(driver_data['status']) if driver_data['status'] in ["Active", "Inactive"] else 0)
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.form_submit_button("💾 Save Changes"):
                                conn = get_db_connection()
                                cur = conn.cursor()
                                cur.execute("""
                                    UPDATE drivers SET name=?, license_number=?, phone=?, email=?, hire_date=?, status=?
                                    WHERE driver_id=?
                                """, (new_name, new_license, new_phone, new_email, new_hire_date, new_status, selected_driver))
                                conn.commit()
                                conn.close()
                                st.success("Driver updated successfully!")
                                del st.session_state.editing_driver
                                safe_rerun()
                        with col2:
                            if st.form_submit_button("❌ Cancel"):
                                del st.session_state.editing_driver
                                safe_rerun()
        else:
            st.info("No drivers found.")
    with tab2:
        st.subheader("Add New Driver")
        with st.form("add_driver"):
            name = st.text_input("Name*", placeholder="Driver's full name")
            license_number = st.text_input("License Number")
            phone = st.text_input("Phone")
            email = st.text_input("Email")
            hire_date = st.date_input("Hire Date", value=date.today())
            status = st.selectbox("Status", ["Active", "Inactive"])
            if st.form_submit_button("Add Driver"):
                if not name:
                    st.error("Name required.")
                else:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO drivers (name, license_number, phone, email, hire_date, status)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (name, license_number, phone, email, hire_date, status))
                    conn.commit()
                    conn.close()
                    st.success("Driver added successfully!")
                    safe_rerun()

# -------------------------
# Expenses page (with bulk operations, attachments, preset filters, optimized queries)
# -------------------------
elif page == "Expenses":
    st.header("💸 Expenses")

    # Ensure infra
    ensure_expense_categories_table()
    ensure_default_expense_categories()
    ensure_expenses_attachments()

    # ===== ONE-TIME SETUP: Add gallons column =====
    st.divider()
    with st.expander("🔧 Database Setup - Add Gallons Column (One-time)", expanded=False):
        st.caption("Click this button once to add fuel gallons tracking to your expenses table")
        if st.button("Add Gallons Column to Expenses Table"):
            try:
                conn_update = get_db_connection()
                cur_update = conn_update.cursor()
                
                # Check if column already exists
                cur_update.execute("PRAGMA table_info(expenses)")
                columns = [col[1] for col in cur_update.fetchall()]
                
                if 'gallons' not in columns:
                    cur_update.execute("ALTER TABLE expenses ADD COLUMN gallons REAL DEFAULT 0")
                    conn_update.commit()
                    st.success("✅ Gallons column added successfully!")
                else:
                    st.info("ℹ️ Gallons column already exists")
                
                conn_update.close()
            except Exception as e:
                st.error(f"Error: {e}")

    # Categories
    categories = get_expense_categories()
    category_names = [c["name"] for c in categories]
    category_by_name = {c["name"]: c for c in categories}

    # Trucks lookup
    trucks_df = get_trucks()
    def _norm(s):
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return None
        return str(s).strip()
    truck_display = ["[None]"] + [f"{_norm(r['number'])} (ID:{r['truck_id']})" for _, r in trucks_df.iterrows()]
    truck_id_map = {f"{_norm(r['number'])} (ID:{r['truck_id']})": r['truck_id'] for _, r in trucks_df.iterrows()}
    truck_id_reverse_map = {r['truck_id']: f"{_norm(r['number'])} (ID:{r['truck_id']})" for _, r in trucks_df.iterrows()}

    # Tabs/sections (top buttons)
    top_buttons = ["All Expenses", "Maintenance"] + [c for c in category_names if c != "Maintenance"] + ["Manage Categories", "+ New Category"]
    cols_tabs = st.columns(len(top_buttons))
    chosen = None
    for i, name in enumerate(top_buttons):
        if cols_tabs[i].button(name, key=f"tab_{name.replace(' ', '_')}"):
            chosen = name
    current_tab = chosen or st.session_state.get("expenses_current_tab", "All Expenses")
    st.session_state["expenses_current_tab"] = current_tab

    # Toolbar (filters)
    st.markdown("#### Filters")
    preset_options = ["Custom", "Today", "Last 7 days", "Last 30 days", "This Month", "Last Month",
                      "This Quarter", "Last Quarter", "Year-To-Date", "Last Year", "All Time"]
    colf1, colf2, colf3, colf4 = st.columns([1.2, 1.2, 1, 1.6])
    with colf1:
        selected_preset = st.selectbox("Date Preset", preset_options, index=preset_options.index("Year-To-Date"), key="date_preset")
    if selected_preset == "Custom":
        with colf2:
            start_date = st.date_input("Start", value=date.today().replace(month=1, day=1))
        with colf3:
            end_date = st.date_input("End", value=date.today())
        with colf4:
            per_page = st.selectbox("Rows/page", [10, 25, 50, 100], index=1)
    else:
        start_date, end_date = get_preset_date_range(selected_preset)
        with colf2:
            st.caption(f"{start_date} → {end_date}")
        with colf3:
            per_page = st.selectbox("Rows/page", [10, 25, 50, 100], index=1)
        with colf4:
            st.info(f"📅 {selected_preset}")
    if start_date > end_date:
        st.error("Start must be before end.")
        st.stop()

    # Top metrics (selected range + YTD)
    conn_tot = get_db_connection()
    try:
        df_range = pd.read_sql_query("SELECT amount FROM expenses WHERE DATE(date) BETWEEN DATE(?) AND DATE(?)", conn_tot, params=(start_date, end_date))
        df_ytd = pd.read_sql_query("SELECT amount FROM expenses WHERE DATE(date) BETWEEN DATE(?) AND DATE(?)", conn_tot, params=(date(date.today().year, 1, 1), date.today()))
    finally:
        conn_tot.close()
    total_range = df_range['amount'].sum() if not df_range.empty else 0.0
    total_ytd = df_ytd['amount'].sum() if not df_ytd.empty else 0.0
    m1, m2 = st.columns(2)
    m1.metric("Total (Selected Range)", f"${total_range:,.2f}")
    m2.metric("Year-to-Date (YTD)", f"${total_ytd:,.2f}")

    # Quick Add launcher
    st.markdown("#### Quick Add Expense")
    cadd1, cadd2 = st.columns([2, 1])
    with cadd1:
        add_category = st.selectbox("Select category", options=category_names, key="add_expense_category_select")
    with cadd2:
        if st.button("➕ Add Expense"):
            st.session_state["show_add_expense_form_for"] = add_category

    # New Category
    if current_tab == "+ New Category":
        st.markdown("### ➕ Create a new expense category")
        new_name = st.text_input("Category name")
        st.caption("Truck Number is implicit. Add user-visible fields below.")
        if "new_category_fields" not in st.session_state:
            st.session_state["new_category_fields"] = [{"label": "", "type": "text"}]
        for i, field in enumerate(st.session_state["new_category_fields"]):
            c1, c2, c3 = st.columns([3, 2, 0.6])
            with c1:
                st.session_state["new_category_fields"][i]["label"] = st.text_input(f"Field {i+1} Label", value=field["label"], key=f"new_field_label_{i}")
            with c2:
                st.session_state["new_category_fields"][i]["type"] = st.selectbox(
                    f"Field {i+1} Type", ["text", "number", "date"],
                    index=["text", "number", "date"].index(field["type"]), key=f"new_field_type_{i}"
                )
            with c3:
                if st.button("❌", key=f"remove_field_{i}"):
                    st.session_state["new_category_fields"].pop(i)
                    safe_rerun()
        if st.button("➕ Add Field"):
            st.session_state["new_category_fields"].append({"label": "", "type": "text"})
        default_apply = st.selectbox("Default apply mode", ["individual", "divide", "exclude"], index=0)
        if st.button("Save category"):
            if not new_name.strip():
                st.error("Please enter a category name.")
            else:
                parsed = []
                for f in st.session_state["new_category_fields"]:
                    label = (f["label"] or "").strip()
                    if label:
                        key = label.lower().replace(" ", "_")
                        parsed.append({"key": key, "label": label, "type": f["type"]})
                try:
                    add_or_update_expense_category(new_name.strip(), parsed, default_apply_mode=default_apply)
                    st.success(f"Category '{new_name}' saved.")
                    st.session_state["new_category_fields"] = [{"label": "", "type": "text"}]
                    safe_rerun()
                except Exception as e:
                    st.error(f"Could not save category: {e}")

    # Add Expense Form (category-aware)
    if st.session_state.get("show_add_expense_form_for"):
        cat_name = st.session_state["show_add_expense_form_for"]
        cat = category_by_name.get(cat_name)
        st.markdown(f"### Add Expense — {cat_name}")

        # Build multiselect options
        tdf = get_trucks()
        def _tlabel(row):
            return f"{str(row.get('number') or '').strip()} (ID:{row.get('truck_id')})"
        all_labels = [] if tdf.empty else [_tlabel(r) for _, r in tdf.iterrows()]
        label_to_id = { _tlabel(r): r["truck_id"] for _, r in tdf.iterrows() } if not tdf.empty else {}

        with st.form(key=f"add_expense_form_{cat_name}"):
            ad_date = st.date_input("Date", value=date.today())
            ad_amount = st.number_input("Amount (total if dividing)", min_value=0.0, value=0.0, step=0.01)
            
            # ===== ADD GALLONS FIELD FOR FUEL =====
            ad_gallons = None
            if cat_name == "Fuel":
                ad_gallons = st.number_input("Gallons", min_value=0.0, value=0.0, step=0.1)
            
            ad_apply = st.selectbox("Apply mode", ["individual", "divide", "exclude"], index=["individual", "divide", "exclude"].index(cat.get("default_apply_mode", "individual")))

            # Divide state keys
            sess_key_all = f"divide_all_trucks__{cat_name}"
            sess_key_sel = f"divide_selected_labels__{cat_name}"
            if sess_key_all not in st.session_state:
                st.session_state[sess_key_all] = True
            if sess_key_sel not in st.session_state:
                st.session_state[sess_key_sel] = all_labels.copy()

            selected_truck_id = None
            if ad_apply == "divide":
                st.info("Divide mode splits the amount equally across selected trucks.")
                all_now = st.checkbox("All trucks", value=st.session_state[sess_key_all])
                if all_now != st.session_state[sess_key_all]:
                    st.session_state[sess_key_all] = all_now
                    if all_now:
                        st.session_state[sess_key_sel] = []
                selected_labels_now = st.multiselect(
                    "Select trucks (ignored if All is checked)",
                    options=all_labels,
                    default=st.session_state[sess_key_sel]
                )
                if selected_labels_now != st.session_state[sess_key_sel]:
                    st.session_state[sess_key_sel] = selected_labels_now
            else:
                ad_truck_label = st.selectbox("Truck", ["[None]"] + all_labels, index=0)
                selected_truck_id = None if ad_truck_label == "[None]" else label_to_id.get(ad_truck_label)

            # Category fields -> metadata
            meta = {}
            for f in cat.get("schema", []):
                t = f.get("type", "text"); lab = f.get("label", ""); key = f.get("key") or lab.lower().replace(" ", "_")
                if t == "number":
                    meta[key] = st.number_input(lab, value=0.0, step=0.01, key=f"add_{cat_name}_{key}")
                elif t == "date":
                    meta[key] = str(st.date_input(lab, value=date.today(), key=f"add_{cat_name}_{key}"))
                else:
                    meta[key] = st.text_input(lab, value="", key=f"add_{cat_name}_{key}")

            # Attachments
            uploaded_files = st.file_uploader("Attachments (optional)", accept_multiple_files=True, key=f"add_attachments_{cat_name}")

            submitted = st.form_submit_button("Add Expense")
            if submitted:
                try:
                    conn_a = get_db_connection()
                    cur = conn_a.cursor()

                    attachments = []
                    if uploaded_files:
                        for uf in uploaded_files:
                            att = save_uploaded_file_to_base64(uf)
                            if att:
                                attachments.append(att)

                    if ad_apply == "divide":
                        divide_all = st.session_state[sess_key_all]
                        chosen_labels = st.session_state[sess_key_sel]
                        if divide_all:
                            target_ids = list(tdf["truck_id"].values) if not tdf.empty else []
                        else:
                            target_ids = [label_to_id.get(lbl) for lbl in chosen_labels]
                        target_ids = [int(t) for t in target_ids if t is not None]
                        uniq = sorted(set(target_ids))
                        if not uniq:
                            conn_a.close()
                            st.error("Please select at least one truck (or choose All trucks).")
                            st.stop()

                        per = round(ad_amount / len(uniq), 2)
                        amounts = [per] * len(uniq)
                        diff = round(ad_amount - sum(amounts), 2)
                        if diff != 0 and amounts:
                            amounts[-1] = round(amounts[-1] + diff, 2)

                        # Divide gallons too if Fuel
                        gallons_per = None
                        if cat_name == "Fuel" and ad_gallons:
                            gallons_per = round(ad_gallons / len(uniq), 2)

                        for i, tid in enumerate(uniq):
                            cur.execute("""
                                INSERT INTO expenses (date, category, amount, truck_id, description, metadata, apply_mode, attachments, gallons)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (ad_date, cat_name, amounts[i], tid, "", json.dumps(meta), "divide", json.dumps(attachments), gallons_per))
                    else:
                        cur.execute("""
                            INSERT INTO expenses (date, category, amount, truck_id, description, metadata, apply_mode, attachments, gallons)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (ad_date, cat_name, ad_amount, selected_truck_id, "", json.dumps(meta), ad_apply, json.dumps(attachments), ad_gallons))

                    conn_a.commit()
                    conn_a.close()
                    st.success("Expense added.")
                    # reset state
                    st.session_state.pop(sess_key_all, None)
                    st.session_state.pop(sess_key_sel, None)
                    st.session_state.pop("show_add_expense_form_for", None)
                    safe_rerun()
                except Exception as e:
                    st.error(f"Failed to add: {e}")

        # Cancel Add
        if st.session_state.get("show_add_expense_form_for"):
            if st.button("Cancel add"):
                st.session_state.pop(f"divide_all_trucks__{cat_name}", None)
                st.session_state.pop(f"divide_selected_labels__{cat_name}", None)
                st.session_state.pop("show_add_expense_form_for", None)
                safe_rerun()

    # Manage Categories
    if current_tab == "Manage Categories":
        st.markdown("### Manage Expense Categories")
        for cat in categories:
            with st.expander(f"📁 {cat['name']}"):
                st.write(f"**Default apply mode:** {cat.get('default_apply_mode', 'individual')}")
                st.write("**Schema:**")
                if cat.get("schema"):
                    for f in cat["schema"]:
                        st.write(f"- {f.get('label')} ({f.get('type')})")
                else:
                    st.write("No custom fields.")
                
                # Edit category
                with st.form(key=f"edit_cat_{cat['name']}"):
                    st.markdown("#### Edit Category")
                    new_name = st.text_input("Category name", value=cat['name'])
                    new_apply = st.selectbox("Default apply mode", ["individual", "divide", "exclude"], 
                                            index=["individual", "divide", "exclude"].index(cat.get('default_apply_mode', 'individual')))
                    
                    # Edit fields
                    if f"edit_fields_{cat['name']}" not in st.session_state:
                        st.session_state[f"edit_fields_{cat['name']}"] = cat.get("schema", []).copy()
                    
                    for i, field in enumerate(st.session_state[f"edit_fields_{cat['name']}"]):
                        c1, c2, c3 = st.columns([3, 2, 0.6])
                        with c1:
                            st.session_state[f"edit_fields_{cat['name']}"][i]["label"] = st.text_input(
                                f"Field {i+1} Label", value=field.get("label", ""), key=f"edit_{cat['name']}_field_label_{i}")
                        with c2:
                            st.session_state[f"edit_fields_{cat['name']}"][i]["type"] = st.selectbox(
                                f"Field {i+1} Type", ["text", "number", "date"],
                                index=["text", "number", "date"].index(field.get("type", "text")), 
                                key=f"edit_{cat['name']}_field_type_{i}")
                        with c3:
                            if st.form_submit_button("❌", help=f"Remove field {i+1}"):
                                st.session_state[f"edit_fields_{cat['name']}"].pop(i)
                                safe_rerun()
                    
                    if st.form_submit_button("➕ Add Field"):
                        st.session_state[f"edit_fields_{cat['name']}"].append({"label": "", "type": "text"})
                        safe_rerun()
                    
                    if st.form_submit_button("Save Changes"):
                        parsed = []
                        for f in st.session_state[f"edit_fields_{cat['name']}"]:
                            label = (f.get("label") or "").strip()
                            if label:
                                key = label.lower().replace(" ", "_")
                                parsed.append({"key": key, "label": label, "type": f.get("type", "text")})
                        try:
                            add_or_update_expense_category(new_name.strip(), parsed, default_apply_mode=new_apply)
                            st.success(f"Category '{new_name}' updated.")
                            st.session_state.pop(f"edit_fields_{cat['name']}", None)
                            safe_rerun()
                        except Exception as e:
                            st.error(f"Could not update category: {e}")
                
                # Delete category
                if st.button(f"🗑️ Delete '{cat['name']}'", key=f"delete_cat_{cat['name']}"):
                    try:
                        conn_d = get_db_connection()
                        cur = conn_d.cursor()
                        cur.execute("DELETE FROM expense_categories WHERE name = ?", (cat['name'],))
                        conn_d.commit()
                        conn_d.close()
                        st.success(f"Category '{cat['name']}' deleted.")
                        safe_rerun()
                    except Exception as e:
                        st.error(f"Could not delete category: {e}")

    # View tables (All Expenses or specific category)
    if current_tab not in ["Manage Categories", "+ New Category"]:
        filter_cat = None if current_tab == "All Expenses" else current_tab
        st.markdown(f"### {current_tab}")

        # Build query
        query = "SELECT * FROM expenses WHERE DATE(date) BETWEEN DATE(?) AND DATE(?)"
        params = [start_date, end_date]
        if filter_cat:
            query += " AND category = ?"
            params.append(filter_cat)
        query += " ORDER BY date DESC"

        conn_v = get_db_connection()
        try:
            df_expenses = pd.read_sql_query(query, conn_v, params=params)
        finally:
            conn_v.close()

        if df_expenses.empty:
            st.info("No expenses found for this filter.")
        else:
            # Pagination
            total_rows = len(df_expenses)
            total_pages = (total_rows + per_page - 1) // per_page
            if "expenses_page" not in st.session_state:
                st.session_state["expenses_page"] = 1
            current_page = st.session_state["expenses_page"]
            if current_page > total_pages:
                current_page = total_pages
                st.session_state["expenses_page"] = current_page

            start_idx = (current_page - 1) * per_page
            end_idx = start_idx + per_page
            df_page = df_expenses.iloc[start_idx:end_idx]

            # Display table
            st.markdown(f"**Showing {start_idx+1}-{min(end_idx, total_rows)} of {total_rows} expenses**")
            
            # Build display dataframe
            display_data = []
            for _, row in df_page.iterrows():
                truck_label = "[None]"
                if pd.notna(row.get("truck_id")):
                    truck_label = truck_id_reverse_map.get(int(row["truck_id"]), f"ID:{int(row['truck_id'])}")
                
                meta = {}
                try:
                    meta = json.loads(row.get("metadata") or "{}")
                except:
                    pass
                
                # Get category schema
                cat_schema = []
                if row.get("category"):
                    cat_obj = category_by_name.get(row["category"])
                    if cat_obj:
                        cat_schema = cat_obj.get("schema", [])
                
                # Build metadata display
                meta_display = []
                for f in cat_schema:
                    key = f.get("key")
                    label = f.get("label")
                    if key in meta:
                        meta_display.append(f"{label}: {meta[key]}")
                
                # Attachments count
                attachments = []
                try:
                    attachments = json.loads(row.get("attachments") or "[]")
                except:
                    pass
                
                display_row = {
                    "ID": row["expense_id"],
                    "Date": row["date"],
                    "Category": row["category"],
                    "Amount": f"${row['amount']:.2f}",
                    "Truck": truck_label,
                    "Apply Mode": row.get("apply_mode", "individual"),
                    "Details": " | ".join(meta_display) if meta_display else "-",
                    "Attachments": len(attachments)
                }
                
                # Add gallons if Fuel category
                if row.get("category") == "Fuel" and pd.notna(row.get("gallons")):
                    display_row["Gallons"] = f"{row['gallons']:.2f}"
                
                display_data.append(display_row)
            
            df_display = pd.DataFrame(display_data)
            st.dataframe(df_display, use_container_width=True)

            # Pagination controls
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            with col1:
                if st.button("⏮️ First", disabled=(current_page == 1)):
                    st.session_state["expenses_page"] = 1
                    safe_rerun()
            with col2:
                if st.button("◀️ Prev", disabled=(current_page == 1)):
                    st.session_state["expenses_page"] = current_page - 1
                    safe_rerun()
            with col3:
                st.markdown(f"<div style='text-align: center; padding-top: 8px;'>Page {current_page} of {total_pages}</div>", unsafe_allow_html=True)
            with col4:
                if st.button("Next ▶️", disabled=(current_page == total_pages)):
                    st.session_state["expenses_page"] = current_page + 1
                    safe_rerun()
            with col5:
                if st.button("Last ⏭️", disabled=(current_page == total_pages)):
                    st.session_state["expenses_page"] = total_pages
                    safe_rerun()

            # Bulk operations
            st.markdown("---")
            st.markdown("#### Bulk Operations")
            bulk_col1, bulk_col2 = st.columns(2)
            
            with bulk_col1:
                st.markdown("##### Export")
                export_format = st.selectbox("Format", ["CSV", "Excel"], key="export_format_expenses")
                if st.button("📥 Export Current View"):
                    try:
                        if export_format == "CSV":
                            csv = df_expenses.to_csv(index=False)
                            st.download_button("Download CSV", csv, f"expenses_{current_tab}_{start_date}_{end_date}.csv", "text/csv")
                        else:
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df_expenses.to_excel(writer, index=False, sheet_name='Expenses')
                            st.download_button("Download Excel", output.getvalue(), 
                                             f"expenses_{current_tab}_{start_date}_{end_date}.xlsx", 
                                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
            
            with bulk_col2:
                st.markdown("##### Delete")
                delete_ids = st.text_input("Enter expense IDs to delete (comma-separated)", key="bulk_delete_ids")
                if st.button("🗑️ Delete Selected", type="primary"):
                    if delete_ids.strip():
                        try:
                            ids = [int(x.strip()) for x in delete_ids.split(",") if x.strip().isdigit()]
                            if ids:
                                conn_del = get_db_connection()
                                cur = conn_del.cursor()
                                placeholders = ",".join(["?"] * len(ids))
                                cur.execute(f"DELETE FROM expenses WHERE expense_id IN ({placeholders})", ids)
                                conn_del.commit()
                                conn_del.close()
                                st.success(f"Deleted {len(ids)} expense(s).")
                                safe_rerun()
                            else:
                                st.warning("No valid IDs provided.")
                        except Exception as e:
                            st.error(f"Delete failed: {e}")

            # Edit expense
            st.markdown("---")
            st.markdown("#### Edit Expense")
            edit_id = st.number_input("Enter Expense ID to edit", min_value=1, step=1, key="edit_expense_id")
            if st.button("Load Expense for Editing"):
                st.session_state["editing_expense_id"] = int(edit_id)
                safe_rerun()

            if st.session_state.get("editing_expense_id"):
                eid = st.session_state["editing_expense_id"]
                conn_e = get_db_connection()
                try:
                    df_edit = pd.read_sql_query("SELECT * FROM expenses WHERE expense_id = ?", conn_e, params=(eid,))
                finally:
                    conn_e.close()

                if df_edit.empty:
                    st.error(f"Expense ID {eid} not found.")
                    st.session_state.pop("editing_expense_id", None)
                else:
                    row = df_edit.iloc[0]
                    st.markdown(f"##### Editing Expense #{eid}")

                    with st.form(key=f"edit_expense_form_{eid}"):
                        ed_date = st.date_input("Date", value=pd.to_datetime(row["date"]).date())
                        ed_category = st.selectbox("Category", category_names, 
                                                   index=category_names.index(row["category"]) if row["category"] in category_names else 0)
                        ed_amount = st.number_input("Amount", value=float(row["amount"]), step=0.01)
                        
                        # ===== EDIT GALLONS FOR FUEL =====
                        ed_gallons = None
                        if ed_category == "Fuel":
                            current_gallons = row.get("gallons", 0.0)
                            if pd.isna(current_gallons):
                                current_gallons = 0.0
                            ed_gallons = st.number_input("Gallons", value=float(current_gallons), step=0.1)
                        
                        # Truck selection
                        current_truck_id = row.get("truck_id")
                        if pd.isna(current_truck_id):
                            current_truck_label = "[None]"
                        else:
                            current_truck_label = truck_id_reverse_map.get(int(current_truck_id), "[None]")
                        
                        ed_truck_label = st.selectbox("Truck", truck_display, 
                                                     index=truck_display.index(current_truck_label) if current_truck_label in truck_display else 0)
                        ed_truck_id = None if ed_truck_label == "[None]" else truck_id_map.get(ed_truck_label)
                        
                        ed_apply = st.selectbox("Apply mode", ["individual", "divide", "exclude"],
                                               index=["individual", "divide", "exclude"].index(row.get("apply_mode", "individual")))

                        # Metadata
                        cat_obj = category_by_name.get(ed_category)
                        current_meta = {}
                        try:
                            current_meta = json.loads(row.get("metadata") or "{}")
                        except:
                            pass
                        
                        new_meta = {}
                        if cat_obj:
                            for f in cat_obj.get("schema", []):
                                t = f.get("type", "text")
                                lab = f.get("label", "")
                                key = f.get("key") or lab.lower().replace(" ", "_")
                                current_val = current_meta.get(key, "")
                                
                                if t == "number":
                                    new_meta[key] = st.number_input(lab, value=float(current_val) if current_val else 0.0, 
                                                                   step=0.01, key=f"edit_{eid}_{key}")
                                elif t == "date":
                                    try:
                                        date_val = pd.to_datetime(current_val).date() if current_val else date.today()
                                    except:
                                        date_val = date.today()
                                    new_meta[key] = str(st.date_input(lab, value=date_val, key=f"edit_{eid}_{key}"))
                                else:
                                    new_meta[key] = st.text_input(lab, value=str(current_val), key=f"edit_{eid}_{key}")

                        # Attachments
                        current_attachments = []
                        try:
                            current_attachments = json.loads(row.get("attachments") or "[]")
                        except:
                            pass
                        
                        if current_attachments:
                            st.markdown("**Current Attachments:**")
                            for i, att in enumerate(current_attachments):
                                st.write(f"{i+1}. {att.get('filename', 'Unknown')}")
                        
                        new_files = st.file_uploader("Add new attachments", accept_multiple_files=True, key=f"edit_attachments_{eid}")

                        submitted = st.form_submit_button("Save Changes")
                        if submitted:
                            try:
                                # Merge attachments
                                all_attachments = current_attachments.copy()
                                if new_files:
                                    for uf in new_files:
                                        att = save_uploaded_file_to_base64(uf)
                                        if att:
                                            all_attachments.append(att)

                                conn_u = get_db_connection()
                                cur = conn_u.cursor()
                                cur.execute("""
                                    UPDATE expenses 
                                    SET date = ?, category = ?, amount = ?, truck_id = ?, 
                                        metadata = ?, apply_mode = ?, attachments = ?, gallons = ?
                                    WHERE expense_id = ?
                                """, (ed_date, ed_category, ed_amount, ed_truck_id, 
                                     json.dumps(new_meta), ed_apply, json.dumps(all_attachments), ed_gallons, eid))
                                conn_u.commit()
                                conn_u.close()
                                
                                st.success(f"Expense #{eid} updated.")
                                st.session_state.pop("editing_expense_id", None)
                                safe_rerun()
                            except Exception as e:
                                st.error(f"Update failed: {e}")

                    if st.button("Cancel Edit"):
                        st.session_state.pop("editing_expense_id", None)
                        safe_rerun()

# -------------------------
# Income Management
# -------------------------
if page == "Income":
    st.header("💵 Income Management")
    tab1, tab2 = st.tabs(["View Income", "Add New Income"])

    st.divider()
    st.subheader("🔍 Diagnose Address Data")
    st.caption("Check what addresses are actually stored in the database")

    if st.button("Show Sample Address Data"):
        conn = get_db_connection()
        cur = conn.cursor()
    
        cur.execute("""
            SELECT 
                income_id,
                truck_id,
                pickup_city,
                pickup_state, 
                pickup_zip,
                pickup_address,
                pickup_full_address,
                delivery_city,
                delivery_state,
                delivery_zip,
                delivery_address,
                delivery_full_address,
                empty_miles
            FROM income
            WHERE pickup_date IS NOT NULL
            ORDER BY pickup_date DESC
            LIMIT 10
        """)
    
        rows = cur.fetchall()
        conn.close()
    
        if rows:
            df_diag = pd.DataFrame(rows, columns=[
                'ID', 'Truck', 'P_City', 'P_State', 'P_Zip', 
                'P_Address', 'P_Full_Address',
                'D_City', 'D_State', 'D_Zip',
                'D_Address', 'D_Full_Address',
                'Empty_Miles'
            ])
            st.dataframe(df_diag, use_container_width=True)
        
            # Check which columns have data
            st.write("**Column Status:**")
            for col in ['P_Address', 'P_Full_Address', 'D_Address', 'D_Full_Address']:
                non_null = df_diag[col].notna().sum()
                st.write(f"- {col}: {non_null}/10 records have data")
        else:
            st.info("No records found")

    # One-time cleanup button
    with st.expander("🔧 Database Cleanup Tools"):
        st.caption("Run this once to fix ZIP code decimals in existing data")
        if st.button("Fix ZIP Code Decimals (.0 issue)"):
            conn = get_db_connection()
            cur = conn.cursor()
            
            try:
                # Remove .0 from ZIP codes
                cur.execute("""
                    UPDATE income 
                    SET pickup_zip = CAST(CAST(pickup_zip AS INTEGER) AS TEXT)
                    WHERE pickup_zip LIKE '%.0'
                """)
                
                cur.execute("""
                    UPDATE income 
                    SET delivery_zip = CAST(CAST(delivery_zip AS INTEGER) AS TEXT)
                    WHERE delivery_zip LIKE '%.0'
                """)
                
                # Rebuild pickup addresses without .0
                cur.execute("""
                    UPDATE income
                    SET pickup_address = 
                        CASE 
                            WHEN pickup_city IS NOT NULL OR pickup_state IS NOT NULL OR pickup_zip IS NOT NULL
                            THEN (
                                COALESCE(pickup_city, '') || 
                                CASE WHEN pickup_city IS NOT NULL AND pickup_state IS NOT NULL THEN ', ' ELSE '' END ||
                                COALESCE(pickup_state, '') ||
                                CASE WHEN (pickup_city IS NOT NULL OR pickup_state IS NOT NULL) AND pickup_zip IS NOT NULL THEN ' ' ELSE '' END ||
                                COALESCE(pickup_zip, '')
                            )
                            ELSE pickup_address
                        END
                    WHERE pickup_address LIKE '%.0'
                """)
                
                # Rebuild delivery addresses without .0
                cur.execute("""
                    UPDATE income
                    SET delivery_address = 
                        CASE 
                            WHEN delivery_city IS NOT NULL OR delivery_state IS NOT NULL OR delivery_zip IS NOT NULL
                            THEN (
                                COALESCE(delivery_city, '') || 
                                CASE WHEN delivery_city IS NOT NULL AND delivery_state IS NOT NULL THEN ', ' ELSE '' END ||
                                COALESCE(delivery_state, '') ||
                                CASE WHEN (delivery_city IS NOT NULL OR delivery_state IS NOT NULL) AND delivery_zip IS NOT NULL THEN ' ' ELSE '' END ||
                                COALESCE(delivery_zip, '')
                            )
                            ELSE delivery_address
                        END
                    WHERE delivery_address LIKE '%.0'
                """)
                
                rows_affected = cur.rowcount
                conn.commit()
                conn.close()
                
                st.success(f"✅ Fixed ZIP codes! Updated {rows_affected} records.")
                st.info("You can now close this section - cleanup is complete.")
                
            except Exception as e:
                st.error(f"Error during cleanup: {e}")
                conn.close()

    with tab1:
        # -------------------------
        # Date range filter
        # -------------------------

        today_d = date.today()
        default_start = COMPANY_START if "COMPANY_START" in globals() else date(2019, 1, 1)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            start_date = st.date_input("From", value=default_start, key="income_from")
        with c2:
            end_date = st.date_input("To", value=today_d, key="income_to")
        with c3:
            page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1, key="income_page_size")

        if start_date > end_date:
            st.warning("Start date cannot be after end date. Adjusting.")
            start_date, end_date = end_date, start_date

        # Fetch and filter income by date on the DB side for efficiency
        conn = get_db_connection()
        try:
            q = """
                SELECT 
                    i.income_id, 
                    i.date, 
                    i.source, 
                    i.amount, 
                    i.truck_id,
                    t.number as truck_number,
                    i.description,
                    i.driver_name,
                    i.broker_number,
                    i.tonu,
                    i.empty_miles, 
                    i.loaded_miles, 
                    i.rpm,
                    i.pickup_date, 
                    i.pickup_time,
                    i.pickup_city,
                    i.pickup_state,
                    i.pickup_zip,
                    i.pickup_address,
                    i.delivery_date, 
                    i.delivery_time,
                    i.delivery_city,
                    i.delivery_state,
                    i.delivery_zip,
                    i.delivery_address,
                    i.stops
                FROM income i
                LEFT JOIN trucks t ON i.truck_id = t.truck_id
                WHERE DATE(i.date) BETWEEN DATE(?) AND DATE(?)
                ORDER BY DATE(i.date) DESC, i.delivery_time DESC, i.income_id DESC
            """
            income_df = pd.read_sql_query(q, conn, params=(start_date.isoformat(), end_date.isoformat()))
        except Exception as e:
            st.error(f"Failed to load income: {e}")
            income_df = pd.DataFrame()
        finally:
            conn.close()

        if income_df is None or income_df.empty:
            st.info("No income records in selected range.")
        else:
            # Show Load number instead of Description, and Truck Number instead of truck_id
            view_df = income_df.copy()
        
            # Rename columns for display
            if "description" in view_df.columns:
                view_df.rename(columns={"description": "Load number"}, inplace=True)
        
            # Drop truck_id and keep only truck_number for display
            display_columns = [col for col in view_df.columns if col != 'truck_id']
            view_df = view_df[display_columns]
        
            # Rename truck_number to Truck for cleaner display
            if "truck_number" in view_df.columns:
                view_df.rename(columns={"truck_number": "Truck"}, inplace=True)
        
            # Reorder columns to put Truck near the front
            cols = view_df.columns.tolist()
            if 'Truck' in cols:
                # Move Truck to position after date
                cols.remove('Truck')
                date_idx = cols.index('date') if 'date' in cols else 0
                cols.insert(date_idx + 1, 'Truck')
                view_df = view_df[cols]

            # Pagination state
            if "income_page_index" not in st.session_state:
                st.session_state.income_page_index = 0

            total_rows = len(view_df)
            total_pages = max(1, (total_rows + page_size - 1) // page_size)
            current_page = min(st.session_state.income_page_index, total_pages - 1)
            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, total_rows)

            st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_rows} loads")

            st.dataframe(view_df.iloc[start_idx:end_idx], use_container_width=True)

            col_prev, col_pnum, col_next = st.columns([1, 2, 1])
            with col_prev:
                if st.button("⬅️ Prev", disabled=(current_page == 0), key="income_prev"):
                    st.session_state.income_page_index = max(0, current_page - 1)
                    st.rerun()
            with col_pnum:
                st.text_input("Page", value=f"{current_page + 1} / {total_pages}", disabled=True)
            with col_next:
                if st.button("Next ➡️", disabled=(current_page >= total_pages - 1), key="income_next"):
                    st.session_state.income_page_index = min(total_pages - 1, current_page + 1)
                    st.rerun()

            # Export buttons
            st.subheader("Export")
            col_exp_csv, col_exp_xlsx = st.columns(2)
            csv_bytes = view_df.to_csv(index=False).encode('utf-8')
            col_exp_csv.download_button(
                "Export Income to CSV",
                data=csv_bytes,
                file_name=f"income_{start_date}_{end_date}.csv",
                mime="text/csv"
            )

            excel_buffer = io.BytesIO()
            excel_data = None
            for engine in ("xlsxwriter", "openpyxl"):
                try:
                    with pd.ExcelWriter(excel_buffer, engine=engine) as writer:
                        view_df.to_excel(writer, index=False, sheet_name='Income')
                    excel_data = excel_buffer.getvalue()
                    break
                except Exception:
                    excel_buffer = io.BytesIO()

            if excel_data:
                col_exp_xlsx.download_button(
                    "Export Income to Excel",
                    data=excel_data,
                    file_name=f"income_{start_date}_{end_date}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                col_exp_xlsx.error("Install xlsxwriter or openpyxl for Excel export:\n`pip install xlsxwriter openpyxl`")

            # Bulk Empty Miles Calculator
            st.subheader("🗺️ Bulk Calculate Empty Miles")
            st.caption("Calculate empty miles for non-TONU loads using OpenRouteService API")

            if st.button("Calculate Empty Miles for All Loads (Rate Limited)"):
                conn_bulk = get_db_connection()
                cur_bulk = conn_bulk.cursor()
    
                # Get all non-TONU loads that need empty miles calculated
                # Use COALESCE to check both address columns
                cur_bulk.execute("""
                    SELECT income_id, truck_id, 
                           pickup_date, pickup_time, 
                           COALESCE(pickup_full_address, pickup_address) as pickup_addr,
                           delivery_date, delivery_time,
                           COALESCE(delivery_full_address, delivery_address) as delivery_addr,
                           tonu
                    FROM income
                    WHERE DATE(date) BETWEEN DATE(?) AND DATE(?)
                      AND (tonu IS NULL OR tonu = 'N' OR tonu = '')
                      AND COALESCE(pickup_full_address, pickup_address) IS NOT NULL
                      AND COALESCE(pickup_full_address, pickup_address) != ''
                      AND (empty_miles IS NULL OR empty_miles = 0 OR empty_miles < 0.5)
                    ORDER BY truck_id, 
                             COALESCE(delivery_date, pickup_date) ASC, 
                             CASE WHEN delivery_time IS NULL THEN '23:59:59' ELSE delivery_time END ASC
                """, (start_date.isoformat(), end_date.isoformat()))
    
                loads_to_calc = cur_bulk.fetchall()
                conn_bulk.close()
    
                if not loads_to_calc:
                    st.info("No non-TONU loads need empty miles calculation in this date range.")
                else:
                    st.info(f"Found {len(loads_to_calc)} non-TONU loads to calculate. This may take a few minutes...")
                    st.caption("⏱️ Processing with 2-second delays to respect API rate limits (40/min)")
        
                    progress_bar = st.progress(0)
                    status_text = st.empty()
        
                    success_count = 0
                    error_count = 0
                    skipped_count = 0
        
                    for idx, load in enumerate(loads_to_calc):
                        (income_id, truck_id, 
                         pickup_date, pickup_time, current_pickup_addr,
                         delivery_date, delivery_time, current_delivery_addr,
                         tonu) = load
            
                        if not current_pickup_addr or len(current_pickup_addr.strip()) < 5:
                            skipped_count += 1
                            continue
            
                        # Find previous load's delivery for this truck
                        conn_prev = get_db_connection()
                        cur_prev = conn_prev.cursor()
            
                        # Use delivery time to find chronologically previous load
                        if delivery_date and delivery_time:
                            cur_prev.execute("""
                                SELECT COALESCE(delivery_full_address, delivery_address) as prev_delivery
                                FROM income
                                WHERE truck_id = ? 
                                  AND (tonu IS NULL OR tonu = 'N' OR tonu = '')
                                  AND COALESCE(delivery_full_address, delivery_address) IS NOT NULL
                                  AND COALESCE(delivery_full_address, delivery_address) != ''
                                  AND (
                                      COALESCE(delivery_date, pickup_date) < ? 
                                      OR (COALESCE(delivery_date, pickup_date) = ? AND COALESCE(delivery_time, '00:00:00') < ?)
                                  )
                                ORDER BY COALESCE(delivery_date, pickup_date) DESC, 
                                         CASE WHEN delivery_time IS NULL THEN '00:00:00' ELSE delivery_time END DESC
                                LIMIT 1
                            """, (truck_id, delivery_date, delivery_date, delivery_time))
                        elif pickup_date:
                            # Fallback: use pickup date if delivery date is missing
                            cur_prev.execute("""
                                SELECT COALESCE(delivery_full_address, delivery_address) as prev_delivery
                                FROM income
                                WHERE truck_id = ? 
                                  AND (tonu IS NULL OR tonu = 'N' OR tonu = '')
                                  AND COALESCE(delivery_full_address, delivery_address) IS NOT NULL
                                  AND COALESCE(delivery_full_address, delivery_address) != ''
                                  AND COALESCE(delivery_date, pickup_date) < ?
                                ORDER BY COALESCE(delivery_date, pickup_date) DESC, 
                                         CASE WHEN delivery_time IS NULL THEN '00:00:00' ELSE delivery_time END DESC
                                LIMIT 1
                            """, (truck_id, pickup_date))
                        else:
                            # No date info, skip
                            conn_prev.close()
                            skipped_count += 1
                            continue
            
                        prev_load = cur_prev.fetchone()
                        conn_prev.close()
            
                        if prev_load and prev_load[0]:
                            prev_delivery = prev_load[0]
                
                            status_text.text(f"Processing load {idx+1}/{len(loads_to_calc)}... ({success_count} ✓, {error_count} ✗, {skipped_count} skipped)")
                
                            calculated_miles = calculate_distance_miles(prev_delivery, current_pickup_addr)
                
                            if calculated_miles > 0:
                                conn_update = get_db_connection()
                                cur_update = conn_update.cursor()
                                cur_update.execute("""
                                    UPDATE income SET empty_miles = ? WHERE income_id = ?
                                """, (calculated_miles, income_id))
                                conn_update.commit()
                                conn_update.close()
                                success_count += 1
                            else:
                                error_count += 1
                
                            # 2-second delay between API calls to stay under 40/min limit
                            time.sleep(2)
                        else:
                            # No previous load found (first load for this truck in date range)
                            skipped_count += 1
            
                        progress_bar.progress((idx + 1) / len(loads_to_calc))
        
                    progress_bar.empty()
                    status_text.empty()
        
                    st.success(f"✅ Calculated empty miles for {success_count} loads")
                    if error_count > 0:
                        st.warning(f"⚠️ {error_count} loads failed (check addresses)")
                    if skipped_count > 0:
                        st.info(f"ℹ️ {skipped_count} loads skipped (no previous load or missing address)")
        
                    safe_rerun()

            st.divider()
            st.subheader("🔬 Debug Single Load Empty Miles")
            st.caption("See exactly what addresses are being compared and why the result is wrong")

            # Let user pick a load to debug
            conn_debug = get_db_connection()
            cur_debug = conn_debug.cursor()

            cur_debug.execute("""
                SELECT income_id, truck_id, description,
                       pickup_date, pickup_time,
                       COALESCE(pickup_full_address, pickup_address) as pickup_addr,
                       delivery_date, delivery_time,
                       COALESCE(delivery_full_address, delivery_address) as delivery_addr,
                       empty_miles
                FROM income
                WHERE DATE(date) BETWEEN DATE(?) AND DATE(?)
                ORDER BY truck_id, pickup_date, pickup_time
            """, (start_date.isoformat(), end_date.isoformat()))

            all_loads = cur_debug.fetchall()
            conn_debug.close()

            if all_loads:
                load_options = [
                    f"ID {row[0]} | Truck {row[1]} | {row[2]} | Empty: {row[9] if row[9] else 'None'}"
                    for row in all_loads
                ]
    
                selected_debug = st.selectbox("Select load to debug:", load_options)
                selected_idx = load_options.index(selected_debug)
                selected_load = all_loads[selected_idx]
    
                if st.button("🔍 Debug This Load"):
                    (income_id, truck_id, description,
                     pickup_date, pickup_time, current_pickup_addr,
                     delivery_date, delivery_time, current_delivery_addr,
                     current_empty_miles) = selected_load
        
                    st.write("---")
                    st.write(f"**Current Load:** ID {income_id}, Truck {truck_id}")
                    st.write(f"**Load Number:** {description}")
                    st.write(f"**Pickup:** {pickup_date} {pickup_time or '(no time)'}")
                    st.write(f"**Pickup Address:** `{current_pickup_addr}`")
                    st.write(f"**Delivery:** {delivery_date} {delivery_time or '(no time)'}")
                    st.write(f"**Delivery Address:** `{current_delivery_addr}`")
                    st.write(f"**Current Empty Miles:** {current_empty_miles}")
        
                    st.write("---")
                    st.write("**Finding Previous Load...**")
        
                    # Find previous load using same logic as bulk calculator
                    conn_prev = get_db_connection()
                    cur_prev = conn_prev.cursor()
        
                    if delivery_date and delivery_time:
                        cur_prev.execute("""
                            SELECT income_id, description,
                                   pickup_date, pickup_time,
                                   COALESCE(pickup_full_address, pickup_address) as prev_pickup,
                                   delivery_date, delivery_time,
                                   COALESCE(delivery_full_address, delivery_address) as prev_delivery
                            FROM income
                            WHERE truck_id = ? 
                              AND (tonu IS NULL OR tonu = 'N' OR tonu = '')
                              AND COALESCE(delivery_full_address, delivery_address) IS NOT NULL
                              AND COALESCE(delivery_full_address, delivery_address) != ''
                              AND (
                                  COALESCE(delivery_date, pickup_date) < ? 
                                  OR (COALESCE(delivery_date, pickup_date) = ? AND COALESCE(delivery_time, '00:00:00') < ?)
                              )
                            ORDER BY COALESCE(delivery_date, pickup_date) DESC, 
                                     CASE WHEN delivery_time IS NULL THEN '00:00:00' ELSE delivery_time END DESC
                            LIMIT 1
                        """, (truck_id, delivery_date, delivery_date, delivery_time))
                    elif pickup_date:
                        cur_prev.execute("""
                            SELECT income_id, description,
                                   pickup_date, pickup_time,
                                   COALESCE(pickup_full_address, pickup_address) as prev_pickup,
                                   delivery_date, delivery_time,
                                   COALESCE(delivery_full_address, delivery_address) as prev_delivery
                            FROM income
                            WHERE truck_id = ? 
                              AND (tonu IS NULL OR tonu = 'N' OR tonu = '')
                              AND COALESCE(delivery_full_address, delivery_address) IS NOT NULL
                              AND COALESCE(delivery_full_address, delivery_address) != ''
                              AND COALESCE(delivery_date, pickup_date) < ?
                            ORDER BY COALESCE(delivery_date, pickup_date) DESC, 
                                     CASE WHEN delivery_time IS NULL THEN '00:00:00' ELSE delivery_time END DESC
                            LIMIT 1
                        """, (truck_id, pickup_date))
                    else:
                        st.error("❌ No date info for this load!")
                        conn_prev.close()
                        st.stop()
        
                    prev_load = cur_prev.fetchone()
                    conn_prev.close()
        
                    if prev_load:
                        (prev_id, prev_desc, prev_pickup_date, prev_pickup_time,
                         prev_pickup_addr, prev_delivery_date, prev_delivery_time, prev_delivery_addr) = prev_load
            
                        st.write(f"**Previous Load:** ID {prev_id}, Truck {truck_id}")
                        st.write(f"**Load Number:** {prev_desc}")
                        st.write(f"**Pickup:** {prev_pickup_date} {prev_pickup_time or '(no time)'}")
                        st.write(f"**Pickup Address:** `{prev_pickup_addr}`")
                        st.write(f"**Delivery:** {prev_delivery_date} {prev_delivery_time or '(no time)'}")
                        st.write(f"**Delivery Address:** `{prev_delivery_addr}`")
            
                        st.write("---")
                        st.write("**Calculating Empty Miles:**")
                        st.info(f"From: **{prev_delivery_addr}**\n\nTo: **{current_pickup_addr}**")
            
                        calculated = calculate_distance_miles(prev_delivery_addr, current_pickup_addr, debug=True)
            
                        st.write("---")
                        if calculated > 0:
                            st.success(f"✅ **Calculated: {calculated} miles**")
                            if current_empty_miles and abs(current_empty_miles - calculated) > 1:
                                st.warning(f"⚠️ Database has {current_empty_miles} miles (difference: {abs(current_empty_miles - calculated):.2f})")
                        else:
                            st.error("❌ Calculation failed!")
                    else:
                        st.warning("⚠️ No previous load found for this truck before this date/time")
                        st.info("This is likely the first load for this truck in your date range, so empty miles = 0 is correct")

            st.subheader("Edit or Delete Income")
            # Include Load number in the selectbox display
            def fmt_income_option(xid: int):
                row = income_df[income_df['income_id'] == xid].iloc[0]
                load_num = row['description'] or ""
                try:
                    amt = float(row['amount'] or 0.0)
                except Exception:
                    amt = 0.0
                return f"{row['date']} - ${amt:,.2f}" + (f" - Load {load_num}" if load_num else "")

            selected_income = st.selectbox(
                "Select Income",
                options=income_df['income_id'].tolist(),
                format_func=fmt_income_option
            )

            if selected_income:
                income_data = income_df[income_df['income_id'] == selected_income].iloc[0]
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✏️ Edit Income", key=f"edit_income_{selected_income}"):
                        st.session_state.editing_income = selected_income
                with col2:
                    if st.button("🗑️ Delete Income", key=f"delete_income_{selected_income}"):
                        delete_record("income", "income_id", selected_income)
                        st.success("Income deleted!")
                        safe_rerun()

                if st.session_state.get('editing_income') == selected_income:
                    with st.form("edit_income_form"):
                        # Edit form
                        new_date = st.date_input(
                            "Date",
                            value=to_date(income_data['date'])
                        )
                        new_source = st.text_input("Source", value=income_data['source'] or "")
                        new_amount = st.number_input("Amount", value=float(income_data['amount']), min_value=0.0)

                        # truck selection
                        trucks_df = get_trucks()
                        truck_options = [f"{r['number']} - {r['make'] or ''} {r['model'] or ''}".strip()
                                         for _, r in trucks_df.iterrows()]
                        truck_ids = trucks_df['truck_id'].tolist()
                        current_truck_idx = 0
                        if income_data['truck_id']:
                            try:
                                current_truck_idx = truck_ids.index(income_data['truck_id'])
                            except ValueError:
                                current_truck_idx = 0
                        selected_truck_idx = st.selectbox(
                            "Truck",
                            range(len(truck_options)),
                            format_func=lambda x: truck_options[x],
                            index=current_truck_idx
                        )
                        new_truck_id = truck_ids[selected_truck_idx]

                        # Load number
                        new_load_number = st.text_input("Load number", value=income_data['description'] or "")

                        # Pickup/Delivery section
                        st.markdown("---")
                        st.subheader("📍 Pickup & Delivery")
                        
                        col_p1, col_p2 = st.columns(2)
                        with col_p1:
                            new_pickup_date = st.date_input(
                                "Pickup Date",
                                value=to_date(income_data['pickup_date']) if income_data['pickup_date'] else None
                            )
                        with col_p2:
                            new_pickup_time = st.time_input(
                                "Pickup Time",
                                value=datetime.strptime(income_data['pickup_time'], "%H:%M:%S").time() if income_data['pickup_time'] else None
                            )
                        
                        new_pickup_address = st.text_input(
                            "Pickup Address",
                            value=income_data['pickup_address'] or ""
                        )
                        
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            new_delivery_date = st.date_input(
                                "Delivery Date",
                                value=to_date(income_data['delivery_date']) if income_data['delivery_date'] else None
                            )
                        with col_d2:
                            new_delivery_time = st.time_input(
                                "Delivery Time",
                                value=datetime.strptime(income_data['delivery_time'], "%H:%M:%S").time() if income_data['delivery_time'] else None
                            )
                        
                        new_delivery_address = st.text_input(
                            "Delivery Address",
                            value=income_data['delivery_address'] or ""
                        )

                        # Mileage fields
                        st.markdown("---")
                        st.subheader("📏 Mileage & Rate")
                        
                        new_rpm = st.number_input(
                            "Rate Per Mile (RPM)",
                            value=float(income_data['rpm']) if income_data['rpm'] else 0.0,
                            min_value=0.0,
                            step=0.01,
                            help="Revenue per mile"
                        )
                        
                        col_em1, col_em2 = st.columns([3, 1])
                        with col_em1:
                            new_empty_miles = st.number_input(
                                "Empty Miles",
                                value=float(income_data['empty_miles']) if income_data['empty_miles'] else 0.0,
                                min_value=0.0,
                                help="Miles driven empty (deadhead)"
                            )
                        with col_em2:
                            st.write("")
                            st.write("")
                            calc_empty = st.form_submit_button("🗺️ Calculate", help="Calculate empty miles using API")
                        
                        # Auto-calculate loaded miles if RPM is provided
                        auto_loaded = None
                        if new_rpm and new_rpm > 0:
                            auto_loaded = new_amount / new_rpm
                        
                        new_loaded_miles = st.number_input(
                            "Loaded Miles",
                            value=auto_loaded if auto_loaded else (float(income_data['loaded_miles']) if income_data['loaded_miles'] else 0.0),
                            min_value=0.0,
                            help="Miles driven loaded (auto-calculated from Amount ÷ RPM if RPM > 0)"
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            save_btn = st.form_submit_button("💾 Save Changes")
                        with col2:
                            cancel_btn = st.form_submit_button("❌ Cancel")
                        
                        if calc_empty:
                            # Calculate empty miles using API
                            conn_calc = get_db_connection()
                            cur_calc = conn_calc.cursor()
                            
                            # Find previous load using pickup datetime
                            if income_data['pickup_date'] and income_data['pickup_time']:
                                cur_calc.execute("""
                                    SELECT delivery_address
                                    FROM income
                                    WHERE truck_id = ? 
                                      AND delivery_address IS NOT NULL
                                      AND (
                                          pickup_date < ? 
                                          OR (pickup_date = ? AND pickup_time < ?)
                                      )
                                    ORDER BY pickup_date DESC, 
                                             CASE WHEN pickup_time IS NULL THEN '00:00:00' ELSE pickup_time END DESC
                                    LIMIT 1
                                """, (income_data['truck_id'], income_data['pickup_date'], income_data['pickup_date'], income_data['pickup_time']))
                            else:
                                cur_calc.execute("""
                                    SELECT delivery_address
                                    FROM income
                                    WHERE truck_id = ? 
                                      AND delivery_address IS NOT NULL
                                      AND date < ?
                                    ORDER BY date DESC
                                    LIMIT 1
                                """, (income_data['truck_id'], income_data['date']))
                            
                            prev_load = cur_calc.fetchone()
                            conn_calc.close()
                            
                            if prev_load and prev_load[0] and income_data['pickup_address']:
                                with st.spinner(f"Calculating distance..."):
                                    calculated_miles = calculate_distance_miles(prev_load[0], income_data['pickup_address'])
                                    if calculated_miles > 0:
                                        st.success(f"✅ Calculated: {calculated_miles} miles")
                                        new_empty_miles = calculated_miles
                                    else:
                                        st.error("Could not calculate distance. Check addresses.")
                            else:
                                st.warning("Missing previous delivery address or current pickup address.")
                        
                        if save_btn:
                            conn = get_db_connection()
                            cur = conn.cursor()
                            cur.execute(
                                """
                                UPDATE income
                                SET date=?, source=?, amount=?, truck_id=?, description=?,
                                    rpm=?, empty_miles=?, loaded_miles=?,
                                    pickup_date=?, pickup_time=?, pickup_address=?,
                                    delivery_date=?, delivery_time=?, delivery_address=?
                                WHERE income_id=?
                                """,
                                (new_date, new_source, new_amount, new_truck_id, new_load_number,
                                 new_rpm, new_empty_miles, new_loaded_miles,
                                 new_pickup_date, str(new_pickup_time) if new_pickup_time else None, new_pickup_address,
                                 new_delivery_date, str(new_delivery_time) if new_delivery_time else None, new_delivery_address,
                                 selected_income)
                            )
                            conn.commit()
                            conn.close()
                            st.success("Income updated!")
                            del st.session_state.editing_income
                            safe_rerun()
                        
                        if cancel_btn:
                            del st.session_state.editing_income
                            safe_rerun()

    with tab2:
        st.subheader("Add New Income")
        trucks_df = get_trucks()
        if not trucks_df.empty:
            with st.form("add_income"):
                income_date = st.date_input("Date", value=date.today())
                source = st.text_input("Source*", placeholder="e.g., Load Payment")
                amount = st.number_input("Amount*", min_value=0.0, value=0.0)

                truck_options = [f"{r['number']} - {r['make'] or ''} {r['model'] or ''}".strip()
                                 for _, r in trucks_df.iterrows()]
                truck_ids = trucks_df['truck_id'].tolist()
                selected_truck_idx = st.selectbox(
                    "Truck*",
                    range(len(truck_options)),
                    format_func=lambda x: truck_options[x]
                )
                truck_id = truck_ids[selected_truck_idx]

                load_number = st.text_input("Load number", placeholder="e.g., 123456 or BOL #")

                # Pickup/Delivery section
                st.markdown("---")
                st.subheader("📍 Pickup & Delivery")
                
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    pickup_date = st.date_input("Pickup Date")
                with col_p2:
                    pickup_time = st.time_input("Pickup Time")
                
                pickup_address = st.text_input("Pickup Address")
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    delivery_date = st.date_input("Delivery Date")
                with col_d2:
                    delivery_time = st.time_input("Delivery Time")
                
                delivery_address = st.text_input("Delivery Address")

                # Mileage fields
                st.markdown("---")
                st.subheader("📏 Mileage & Rate")
                
                rpm = st.number_input(
                    "Rate Per Mile (RPM)",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    help="Revenue per mile"
                )
                
                empty_miles = st.number_input(
                    "Empty Miles",
                    min_value=0.0,
                    value=0.0,
                    help="Miles driven empty (deadhead)"
                )
                
                # Auto-calculate loaded miles if RPM is provided
                auto_loaded = None
                if rpm and rpm > 0 and amount > 0:
                    auto_loaded = amount / rpm
                
                loaded_miles = st.number_input(
                    "Loaded Miles",
                    min_value=0.0,
                    value=auto_loaded if auto_loaded else 0.0,
                    help="Miles driven loaded (auto-calculated from Amount ÷ RPM if RPM > 0)"
                )

                if st.form_submit_button("Add Income"):
                    if not source or amount <= 0:
                        st.error("Source and positive amount required.")
                    else:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        cur.execute(
                            """
                            INSERT INTO income (
                                    date, source, amount, truck_id, description,
                                    driver_name, broker_number, tonu, stops,
                                    pickup_date, pickup_time, pickup_city, pickup_state, pickup_zip, pickup_address, pickup_full_address,
                                    delivery_date, delivery_time, delivery_city, delivery_state, delivery_zip, delivery_address, delivery_full_address,
                                    rpm, empty_miles, loaded_miles
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                date_val, source_val, amount_val, truck_id, load_number_val,
                                driver_val, broker_val, tonu_val, stops_val,
                                pickup_date_val, pickup_time_val, pickup_city_val, pickup_state_val, pickup_zip_val, pickup_full_address_val, pickup_full_address_val,
                                delivery_date_val, delivery_time_val, delivery_city_val, delivery_state_val, delivery_zip_val, delivery_full_address_val, delivery_full_address_val,
                                rpm_val, empty_miles_val, loaded_miles_val
                            ))
                        conn.commit()
                        conn.close()
                        st.success("Income added!")
                        safe_rerun()
        else:
            st.warning("No trucks available. Please add trucks first.")

# -------------------------
# Bulk Upload page
# -------------------------
elif page == "Bulk Upload":
    st.header("📥 Bulk Upload for Trucks, Trailers, Expenses, and Income")
    tab_trucks, tab_trailers, tab_expenses, tab_income = st.tabs(["Trucks", "Trailers", "Expenses", "Income"])

    # Trucks upload
    with tab_trucks:
        st.subheader("Upload Trucks CSV or Excel (optional assignments & loans)")
        uploaded_file = st.file_uploader("Upload Trucks file", type=['csv', 'xlsx'], key="upload_trucks")

        if uploaded_file:
            # Safe file read
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
                df = None

            if df is not None:
                st.write("Preview:")
                st.dataframe(df.head())
                columns = df.columns.tolist()
                number_col = st.selectbox("Truck Number Column", columns, key="truck_number_col")
                make_col = st.selectbox("Make Column", columns, key="truck_make_col")
                model_col = st.selectbox("Model Column", columns, key="truck_model_col")
                year_col = st.selectbox("Year Column", columns, key="truck_year_col")
                plate_col = st.selectbox("Plate Column", columns, key="truck_plate_col")
                vin_col = st.selectbox("VIN Column", columns, key="truck_vin_col")
                status_col = st.selectbox("Status Column", columns, key="truck_status_col")
                loan_col = st.selectbox("Loan Amount Column", [None] + columns, key="truck_loan_col")
                driver_col = st.selectbox("Driver Identifier Column (optional)", [None] + columns, key="truck_driver_col")
                create_missing_drivers = st.checkbox("Auto-create missing drivers when linking", value=False)

                # New: dispatcher column mapping
                dispatcher_col = st.selectbox("Dispatcher Column (optional) - can be name or numeric id", [None] + columns, key="truck_dispatcher_col")
                create_missing_dispatchers = st.checkbox("Auto-create missing dispatchers when linking by name", value=False)

                if st.button("Upload Trucks Data"):
                    import traceback
                    # DB upload in try/except/finally
                    try:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        errors = []
                        success_count = 0
                        created_drivers = 0
                        created_dispatchers = 0

                        for idx, row in df.iterrows():
                            try:
                                number = row[number_col] if number_col else None
                                make = row[make_col] if make_col and pd.notna(row[make_col]) else None
                                model = row[model_col] if model_col and pd.notna(row[model_col]) else None
                                year = int(row[year_col]) if year_col and pd.notna(row[year_col]) else None
                                plate = row[plate_col] if plate_col and pd.notna(row[plate_col]) else None
                                vin = row[vin_col] if vin_col and pd.notna(row[vin_col]) else None
                                status = row[status_col] if status_col and pd.notna(row[status_col]) else "Active"
                                loan_amount = float(row[loan_col]) if loan_col and pd.notna(row[loan_col]) else 0.0

                                driver_id = None
                                if driver_col and pd.notna(row[driver_col]):
                                    ident = str(row[driver_col]).strip()
                                    cur2 = conn.cursor()
                                    cur2.execute("SELECT driver_id FROM drivers WHERE name=? OR license_number=?", (ident, ident))
                                    found = cur2.fetchone()
                                    if found:
                                        driver_id = found[0]
                                    elif create_missing_drivers and ident:
                                        # create a minimal driver record
                                        cur2.execute("INSERT INTO drivers (name, license_number, status) VALUES (?, ?, ?)", (ident, None, "Active"))
                                        driver_id = cur2.lastrowid
                                        created_drivers += 1

                                dispatcher_id = None
                                if dispatcher_col and pd.notna(row[dispatcher_col]):
                                    val = row[dispatcher_col]
                                    # try numeric ID first
                                    try:
                                        dispatcher_id = int(val)
                                    except Exception:
                                        # treat as name
                                        cur2 = conn.cursor()
                                        cur2.execute("SELECT dispatcher_id FROM dispatchers WHERE name=?", (str(val).strip(),))
                                        found = cur2.fetchone()
                                        if found:
                                            dispatcher_id = found[0]
                                        elif create_missing_dispatchers and str(val).strip():
                                            dispatcher_id = add_dispatcher(str(val).strip())
                                            created_dispatchers += 1

                                cur.execute("""
                                    INSERT INTO trucks (number, make, model, year, plate, vin, status, loan_amount, driver_id, dispatcher_id)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (number, make, model, year, plate, vin, status, loan_amount, driver_id, dispatcher_id))
                                truck_id = cur.lastrowid

                                if loan_amount and loan_amount > 0:
                                    # record loan history entry for the truck
                                    set_loan_history('truck', truck_id, float(loan_amount), start_date=date.today(), note="Bulk upload")

                                if driver_id:
                                    # record driver assignment history
                                    record_driver_assignment(driver_id, truck_id=truck_id, start_date=date.today(), note="Assigned via bulk upload")

                                success_count += 1
                            except Exception as row_e:
                                errors.append(f"Row {idx+1}: {row_e}")

                        conn.commit()
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
                        st.text(traceback.format_exc())
                    finally:
                        try:
                            cur.close()
                        except Exception:
                            pass
                        try:
                            conn.close()
                        except Exception:
                            pass

                    st.success(f"Uploaded {success_count} trucks.")
                    if created_drivers:
                        st.info(f"Auto-created {created_drivers} drivers for linking.")
                    if created_dispatchers:
                        st.info(f"Auto-created {created_dispatchers} dispatchers for linking.")
                    if errors:
                        st.error("Some rows failed:")
                        for err in errors:
                            st.write(err)

    # Trailers upload
    with tab_trailers:
        st.subheader("Upload Trailers CSV or Excel (optional link to truck)")
        uploaded_file = st.file_uploader("Upload Trailers file", type=['csv', 'xlsx'], key="upload_trailers")

        if uploaded_file:
            # Safe file read
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
                df = None

            if df is not None:
                st.write("Preview:")
                st.dataframe(df.head())
                columns = df.columns.tolist()
                number_col = st.selectbox("Trailer Number Column", columns, key="trailer_number_col")
                type_col = st.selectbox("Type Column", columns, key="trailer_type_col")
                year_col = st.selectbox("Year Column", columns, key="trailer_year_col")
                plate_col = st.selectbox("Plate Column", columns, key="trailer_plate_col")
                vin_col = st.selectbox("VIN Column", columns, key="trailer_vin_col")
                status_col = st.selectbox("Status Column", columns, key="trailer_status_col")
                loan_col = st.selectbox("Loan Amount Column", [None] + columns, key="trailer_loan_col")
                st.markdown("Optional: Truck identifier column in your file to link trailer to a truck")
                truck_identifier_col = st.selectbox("Truck Identifier Column (optional)", [None] + columns, key="trailer_truck_identifier_col")
                truck_id_type = st.selectbox("Truck Identifier Type", ["Number", "Plate", "VIN"], key="trailer_truck_id_type")
                create_missing_trucks = st.checkbox("Auto-create missing trucks for links", value=False)

                if st.button("Upload Trailers Data"):
                    import traceback
                    try:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        trucks_df = get_trucks()

                        def normalize(s):
                            if s is None or (isinstance(s, float) and pd.isna(s)):
                                return None
                            return str(s).strip().lower()

                        number_map = {normalize(r['number']): r['truck_id'] for _, r in trucks_df.iterrows() if normalize(r['number'])}
                        plate_map  = {normalize(r['plate']): r['truck_id'] for _, r in trucks_df.iterrows() if normalize(r['plate'])}
                        vin_map    = {normalize(r['vin']): r['truck_id'] for _, r in trucks_df.iterrows() if normalize(r['vin'])}
                        errors = []
                        success_count = 0
                        created_trucks = 0

                        for idx, row in df.iterrows():
                            try:
                                trailer_number = row[number_col] if number_col else None
                                trailer_type = row[type_col] if type_col and pd.notna(row[type_col]) else None
                                trailer_year = int(row[year_col]) if year_col and pd.notna(row[year_col]) else None
                                trailer_plate = row[plate_col] if plate_col and pd.notna(row[plate_col]) else None
                                trailer_vin = row[vin_col] if vin_col and pd.notna(row[vin_col]) else None
                                trailer_status = row[status_col] if status_col and pd.notna(row[status_col]) else "Active"
                                trailer_loan = float(row[loan_col]) if loan_col and pd.notna(row[loan_col]) else 0.0

                                linked_truck_id = None
                                if truck_identifier_col and pd.notna(row[truck_identifier_col]):
                                    ident_raw = row[truck_identifier_col]
                                    ident = normalize(ident_raw)
                                    if truck_id_type == "Number":
                                        linked_truck_id = number_map.get(ident)
                                    elif truck_id_type == "Plate":
                                        linked_truck_id = plate_map.get(ident)
                                    else:
                                        linked_truck_id = vin_map.get(ident)

                                    # fallback to other maps if not found
                                    if linked_truck_id is None:
                                        linked_truck_id = number_map.get(ident) or plate_map.get(ident) or vin_map.get(ident)

                                    if linked_truck_id is None and create_missing_trucks and ident:
                                        new_number = ident if truck_id_type == "Number" else f"IMPORT-{int(datetime.utcnow().timestamp())}"
                                        new_plate = ident if truck_id_type == "Plate" else None
                                        new_vin = ident if truck_id_type == "VIN" else None
                                        cur.execute("""
                                            INSERT INTO trucks (number, make, model, year, plate, vin, status, loan_amount)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                        """, (new_number, None, None, None, new_plate, new_vin, "Active", 0.0))
                                        linked_truck_id = cur.lastrowid
                                        created_trucks += 1
                                        # refresh maps
                                        trucks_df = get_trucks()
                                        number_map = {normalize(r['number']): r['truck_id'] for _, r in trucks_df.iterrows() if normalize(r['number'])}
                                        plate_map  = {normalize(r['plate']): r['truck_id'] for _, r in trucks_df.iterrows() if normalize(r['plate'])}
                                        vin_map    = {normalize(r['vin']): r['truck_id'] for _, r in trucks_df.iterrows() if normalize(r['vin'])}

                                cur.execute("""
                                    INSERT INTO trailers (number, type, year, plate, vin, status, loan_amount, truck_id)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (trailer_number, trailer_type, trailer_year, trailer_plate, trailer_vin, trailer_status, trailer_loan, linked_truck_id))
                                trailer_id = cur.lastrowid

                                if trailer_loan and trailer_loan > 0:
                                    set_loan_history('trailer', trailer_id, float(trailer_loan), start_date=date.today(), note="Bulk upload")
                                if linked_truck_id:
                                    record_trailer_assignment(trailer_id, linked_truck_id, start_date=date.today(), note="Bulk upload")

                                success_count += 1
                            except Exception as row_e:
                                errors.append(f"Row {idx+1}: {row_e}")

                        conn.commit()

                    except Exception as e:
                        st.error(f"Upload failed: {e}")
                        st.text(traceback.format_exc())
                    finally:
                        try:
                            cur.close()
                        except Exception:
                            pass
                        try:
                            conn.close()
                        except Exception:
                            pass

                    st.success(f"Uploaded {success_count} trailers.")
                    if created_trucks:
                        st.info(f"Auto-created {created_trucks} trucks for linking.")
                    if errors:
                        st.error("Some rows failed:")
                        for err in errors:
                            st.write(err)

    # Expenses upload (category-aware)
        with tab_expenses:
            st.subheader("Upload Expenses CSV or Excel (category-aware)")

            # Ensure categories & defaults exist
            ensure_expense_categories_table()
            ensure_default_expense_categories()
            cats = get_expense_categories()
            cat_names = [c["name"] for c in cats]

            selected_category = st.selectbox("Select Expense Category for this upload", cat_names, index=cat_names.index("Maintenance") if "Maintenance" in cat_names else 0)
            uploaded_file = st.file_uploader(f"Upload {selected_category} file", type=['csv', 'xlsx'], key="upload_expenses_category")

            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    df = None

                if df is not None:
                    st.write("Preview:")
                    st.dataframe(df.head())
                    columns = df.columns.tolist()

                    # get category schema
                    cat = next((c for c in cats if c["name"] == selected_category), None)
                    schema = cat["schema"] if cat else []

                    # For Fuel and Tolls enforce required mapping labels
                    required_map = []
                    if selected_category == "Fuel":
                        required_map = [("truck_number_col", "Truck Number"), ("card_col", "Card Number"),
                                        ("tx_date_col", "Transaction Date"), ("location_col", "Location"),
                                        ("amount_col", "Amount Paid"), ("discount_col", "Discount Amount"),
                                        ("gallons_col", "Gallons")]
                    elif selected_category == "Tolls":
                        required_map = [("truck_number_col", "Truck Number"), ("agency_col", "Toll Agency"),
                                        ("date_col", "Date Occurred"), ("amount_col", "Toll Amount")]
                    else:
                        # Generic: allow user to map Truck Number and any defined schema fields
                        required_map = [("truck_number_col", "Truck Number")]
                        for f in schema:
                            required_map.append((f"col_{f['key']}", f["label"]))

                    # Build mapping selectors
                    mapping = {}
                    for key, label in required_map:
                        mapping[key] = st.selectbox(f"Map column for: {label}", [None] + columns, index=0, key=f"map_{selected_category}_{key}")

                    create_missing_trucks = st.checkbox("Automatically create missing trucks when identifier not found", value=False)

                    if st.button("Process Category Upload"):
                        conn = get_db_connection()
                        cur = conn.cursor()
                        trucks_df = get_trucks()
                        def normalize(s):
                            if s is None or (isinstance(s, float) and pd.isna(s)):
                                return None
                            return str(s).strip().lower()
                        number_map = {normalize(r['number']): r['truck_id'] for _, r in trucks_df.iterrows() if normalize(r['number'])}
                        plate_map  = {normalize(r['plate']): r['truck_id'] for _, r in trucks_df.iterrows() if normalize(r['plate'])}
                        vin_map    = {normalize(r['vin']): r['truck_id'] for _, r in trucks_df.iterrows() if normalize(r['vin'])}

                        errors = []
                        success_count = 0
                        created_trucks = 0

                        for idx, row in df.iterrows():
                            try:
                                # truck resolution
                                truck_id = None
                                if mapping.get("truck_number_col") and pd.notna(row[mapping["truck_number_col"]]):
                                    ident = normalize(row[mapping["truck_number_col"]])
                                    truck_id = number_map.get(ident) or plate_map.get(ident) or vin_map.get(ident)
                                    if truck_id is None and create_missing_trucks and ident:
                                        cur.execute("INSERT INTO trucks (number, make, model, year, plate, vin, status, loan_amount) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                                    (ident, None, None, None, None, None, "Active", 0.0))
                                        truck_id = cur.lastrowid
                                        created_trucks += 1

                                # parse category-specific fields to metadata and primary amount/date
                                metadata = {}
                                amount_val = None
                                date_val = None
                                gallons_val = None

                                if selected_category == "Fuel":
                                    # look up each mapped column
                                    card = mapping.get("card_col")
                                    tx_date = mapping.get("tx_date_col")
                                    location = mapping.get("location_col")
                                    amount_col = mapping.get("amount_col")
                                    discount_col = mapping.get("discount_col")
                                    gallons_col = mapping.get("gallons_col")
                                
                                    metadata = {
                                        "card_number": row[card] if card and pd.notna(row[card]) else None,
                                        "transaction_date": str(pd.to_datetime(row[tx_date]).date()) if tx_date and pd.notna(row[tx_date]) else None,
                                        "location": row[location] if location and pd.notna(row[location]) else None,
                                        "discount_amount": float(row[discount_col]) if discount_col and pd.notna(row[discount_col]) else 0.0
                                    }
                                    amount_val = float(row[amount_col]) if amount_col and pd.notna(row[amount_col]) else 0.0
                                    date_val = metadata.get("transaction_date")
                                
                                    # Parse gallons
                                    if gallons_col and pd.notna(row[gallons_col]):
                                        try:
                                            gallons_val = float(row[gallons_col])
                                        except:
                                            gallons_val = 0.0
                                
                                elif selected_category == "Tolls":
                                    agency = mapping.get("agency_col")
                                    date_col = mapping.get("date_col")
                                    amount_col = mapping.get("amount_col")
                                    metadata = {
                                        "toll_agency": row[agency] if agency and pd.notna(row[agency]) else None,
                                        "date_occurred": str(pd.to_datetime(row[date_col]).date()) if date_col and pd.notna(row[date_col]) else None
                                    }
                                    amount_val = float(row[amount_col]) if amount_col and pd.notna(row[amount_col]) else 0.0
                                    date_val = metadata.get("date_occurred")
                                else:
                                    # generic cat, collect each mapped schema field into metadata
                                    for f in schema:
                                        map_key = f"col_{f['key']}"
                                        colname = mapping.get(map_key)
                                        if colname and pd.notna(row[colname]):
                                            if f.get("type") == "number":
                                                try:
                                                    metadata[f['key']] = float(row[colname])
                                                except Exception:
                                                    metadata[f['key']] = row[colname]
                                            elif f.get("type") == "date":
                                                metadata[f['key']] = str(pd.to_datetime(row[colname]).date())
                                            else:
                                                metadata[f['key']] = row[colname]
                                    # try to get amount/date heuristically
                                    for a in ["amount", "amount_paid", "toll_amount", "cost", "price"]:
                                        if a in metadata and metadata[a] is not None:
                                            try:
                                                amount_val = float(metadata[a])
                                            except Exception:
                                                pass
                                    date_val = metadata.get("date") or metadata.get("transaction_date") or metadata.get("date_occurred")

                                # Insert into DB
                                cur.execute("""
                                    INSERT INTO expenses (date, category, amount, truck_id, description, metadata, apply_mode, gallons)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (date_val, selected_category, float(amount_val or 0.0), truck_id, None, json.dumps(metadata), cat.get("default_apply_mode", "individual"), gallons_val))
                                success_count += 1

                            except Exception as e:
                                errors.append(f"Row {idx+1}: {e}")

                        conn.commit()
                        conn.close()
                        st.success(f"Uploaded {success_count} expense records for category {selected_category}.")
                        if created_trucks:
                            st.info(f"Automatically created {created_trucks} trucks.")
                        if errors:
                            st.error("Some rows failed:")
                            for err in errors:
                                st.write(err)

    # Income upload (improved + Load number mapping + RPM/Mileage)
    with tab_income:
        st.subheader("📤 Bulk Upload Income")
    
        uploaded_file = st.file_uploader("Choose Income file (CSV/Excel)", type=['csv', 'xlsx', 'xls'], key="income_uploader")
    
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            
                st.success(f"Loaded {len(df)} rows")
                st.dataframe(df.head(10))
            
                columns = [None] + df.columns.tolist()
            
                st.subheader("Map Columns")
                st.caption("Map your file columns to the database fields")
            
                col1, col2, col3 = st.columns(3)
            
                with col1:
                    st.markdown("**Required Fields**")
                    date_col = st.selectbox("Date Column*", columns[1:], key="income_date_col")
                    truck_col = st.selectbox("Truck Column*", columns[1:], key="income_truck_col")
                    amount_col = st.selectbox("Invoice Amount Column*", columns[1:], key="income_amount_col")
                
                    st.markdown("**Load Info**")
                    broker_col = st.selectbox("Broker # Column", columns, key="income_broker_col")
                    load_number_col = st.selectbox("Load Number Column", columns, key="income_load_number_col")
                    driver_col = st.selectbox("Driver Column", columns, key="income_driver_col")
                    tonu_col = st.selectbox("TONU Column (Y/N)", columns, key="income_tonu_col")
                    stops_col = st.selectbox("Stops Column", columns, key="income_stops_col")
            
                with col2:
                    st.markdown("**Pickup Information**")
                    pickup_date_col = st.selectbox("Pickup Date Column", columns, key="income_pickup_date_col")
                    pickup_time_col = st.selectbox("Pickup Time Column", columns, key="income_pickup_time_col")
    
                    # Option 1: Full address
                    pickup_full_address_col = st.selectbox("Pickup Full Address Column (recommended)", columns, key="income_pickup_full_address_col")
    
                    st.caption("OR use separate components:")
                    pickup_city_col = st.selectbox("Pickup City Column", columns, key="income_pickup_city_col")
                    pickup_state_col = st.selectbox("Pickup State Column", columns, key="income_pickup_state_col")
                    pickup_zip_col = st.selectbox("Pickup Zip Column", columns, key="income_pickup_zip_col")

                with col3:
                    st.markdown("**Delivery Information**")
                    delivery_date_col = st.selectbox("Delivery Date Column", columns, key="income_delivery_date_col")
                    delivery_time_col = st.selectbox("Delivery Time Column", columns, key="income_delivery_time_col")
    
                    # Option 1: Full address
                    delivery_full_address_col = st.selectbox("Delivery Full Address Column (recommended)", columns, key="income_delivery_full_address_col")
    
                    st.caption("OR use separate components:")
                    delivery_city_col = st.selectbox("Delivery City Column", columns, key="income_delivery_city_col")
                    delivery_state_col = st.selectbox("Delivery State Column", columns, key="income_delivery_state_col")
                    delivery_zip_col = st.selectbox("Delivery Zip Column", columns, key="income_delivery_zip_col")
                
                    st.markdown("**Mileage & Rate**")
                    rpm_col = st.selectbox("RPM Column", columns, key="income_rpm_col")
                    distance_col = st.selectbox("Distance Column (Loaded Miles)", columns, key="income_distance_col")
            
                if st.button("Upload Income Records", key="upload_income_btn"):
                    if not date_col or not truck_col or not amount_col:
                        st.error("Date, Truck, and Amount columns are required!")
                    else:
                        conn = get_db_connection()
                        cur = conn.cursor()
                    
                        # Get truck mapping
                        cur.execute("SELECT truck_id, number FROM trucks")
                        truck_map = {str(row[1]).strip().upper(): row[0] for row in cur.fetchall()}
                    
                        success_count = 0
                        error_count = 0
                        errors = []
                    
                        for idx, row in df.iterrows():
                            try:
                                # Required fields
                                date_val = to_date(row[date_col])
                                if not date_val:
                                    errors.append(f"Row {idx+2}: Invalid date")
                                    error_count += 1
                                    continue
                            
                                truck_num = str(row[truck_col]).strip().upper()
                                truck_id = truck_map.get(truck_num)
                                if not truck_id:
                                    errors.append(f"Row {idx+2}: Truck '{truck_num}' not found")
                                    error_count += 1
                                    continue
                            
                                amount_val = float(row[amount_col]) if pd.notna(row[amount_col]) else 0.0
                            
                                # Optional fields
                                broker_val = str(row[broker_col]).strip() if broker_col and pd.notna(row[broker_col]) else None
                                load_number_val = str(row[load_number_col]).strip() if load_number_col and pd.notna(row[load_number_col]) else None
                                driver_val = str(row[driver_col]).strip() if driver_col and pd.notna(row[driver_col]) else None
                            
                                # TONU - convert Y/N to Y/N
                                tonu_val = 'N'
                                if tonu_col and pd.notna(row[tonu_col]):
                                    tonu_str = str(row[tonu_col]).strip().upper()
                                    if tonu_str in ['Y', 'YES', '1', 'TRUE']:
                                        tonu_val = 'Y'
                            
                                stops_val = int(row[stops_col]) if stops_col and pd.notna(row[stops_col]) else None
                            
                                # Pickup info
                                pickup_date_val = to_date(row[pickup_date_col]) if pickup_date_col and pd.notna(row[pickup_date_col]) else None
                                pickup_time_val = str(row[pickup_time_col]).strip() if pickup_time_col and pd.notna(row[pickup_time_col]) else None

                                # Full address takes priority
                                pickup_full_address_val = None
                                pickup_city_val = None
                                pickup_state_val = None
                                pickup_zip_val = None

                                if pickup_full_address_col and pd.notna(row[pickup_full_address_col]):
                                    pickup_full_address_val = str(row[pickup_full_address_col]).strip()
                                else:
                                    # Use components
                                    pickup_city_val = str(row[pickup_city_col]).strip() if pickup_city_col and pd.notna(row[pickup_city_col]) else None
                                    pickup_state_val = str(row[pickup_state_col]).strip() if pickup_state_col and pd.notna(row[pickup_state_col]) else None
    
                                    # Handle ZIP as integer to avoid .0
                                    if pickup_zip_col and pd.notna(row[pickup_zip_col]):
                                        try:
                                            pickup_zip_val = str(int(float(row[pickup_zip_col])))
                                        except:
                                            pickup_zip_val = str(row[pickup_zip_col]).strip()
    
                                    # Build address from components
                                    if pickup_city_val or pickup_state_val or pickup_zip_val:
                                        parts = [p for p in [pickup_city_val, pickup_state_val, pickup_zip_val] if p]
                                        pickup_full_address_val = ", ".join(parts)

                                # Delivery info
                                delivery_date_val = to_date(row[delivery_date_col]) if delivery_date_col and pd.notna(row[delivery_date_col]) else None
                                delivery_time_val = str(row[delivery_time_col]).strip() if delivery_time_col and pd.notna(row[delivery_time_col]) else None

                                # Full address takes priority
                                delivery_full_address_val = None
                                delivery_city_val = None
                                delivery_state_val = None
                                delivery_zip_val = None

                                if delivery_full_address_col and pd.notna(row[delivery_full_address_col]):
                                    delivery_full_address_val = str(row[delivery_full_address_col]).strip()
                                else:
                                    # Use components
                                    delivery_city_val = str(row[delivery_city_col]).strip() if delivery_city_col and pd.notna(row[delivery_city_col]) else None
                                    delivery_state_val = str(row[delivery_state_col]).strip() if delivery_state_col and pd.notna(row[delivery_state_col]) else None
    
                                    # Handle ZIP as integer to avoid .0
                                    if delivery_zip_col and pd.notna(row[delivery_zip_col]):
                                        try:
                                            delivery_zip_val = str(int(float(row[delivery_zip_col])))
                                        except:
                                            delivery_zip_val = str(row[delivery_zip_col]).strip()
    
                                    # Build address from components
                                    if delivery_city_val or delivery_state_val or delivery_zip_val:
                                        parts = [p for p in [delivery_city_val, delivery_state_val, delivery_zip_val] if p]
                                        delivery_full_address_val = ", ".join(parts)

                                # Mileage & Rate
                                rpm_val = float(row[rpm_col]) if rpm_col and pd.notna(row[rpm_col]) else None

                                # Distance column = loaded miles (only for non-TONU loads)
                                loaded_miles_val = None
                                if tonu_val == 'N' and distance_col and pd.notna(row[distance_col]):
                                    loaded_miles_val = float(row[distance_col])

                                # Empty miles will be calculated later, set to NULL for now
                                empty_miles_val = None

                                # Source = "Imported" by default
                                source_val = "Imported"

                                # Insert record
                                cur.execute("""
                                    INSERT INTO income (
                                        date, source, amount, truck_id, description,
                                        driver_name, broker_number, tonu, stops,
                                        pickup_date, pickup_time, pickup_city, pickup_state, pickup_zip, pickup_full_address,
                                        delivery_date, delivery_time, delivery_city, delivery_state, delivery_zip, delivery_full_address,
                                        rpm, empty_miles, loaded_miles
                                    )
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    date_val, source_val, amount_val, truck_id, load_number_val,
                                    driver_val, broker_val, tonu_val, stops_val,
                                    pickup_date_val, pickup_time_val, pickup_city_val, pickup_state_val, pickup_zip_val, pickup_full_address_val,
                                    delivery_date_val, delivery_time_val, delivery_city_val, delivery_state_val, delivery_zip_val, delivery_full_address_val,
                                    rpm_val, empty_miles_val, loaded_miles_val
                                ))
                            
                                success_count += 1
                            
                            except Exception as e:
                                errors.append(f"Row {idx+2}: {str(e)}")
                                error_count += 1
                    
                        conn.commit()
                        conn.close()
                    
                        st.success(f"✅ Successfully uploaded {success_count} income records!")
                        if error_count > 0:
                            st.warning(f"⚠️ {error_count} rows failed")
                            with st.expander("View Errors"):
                                for err in errors[:50]:  # Show first 50 errors
                                    st.text(err)
                    
                        safe_rerun()
        
            except Exception as e:
                st.error(f"Failed to read file: {e}")

# -------------------------
# Histories viewer (final fixed version)
# -------------------------
if page == "Histories":
    st.header("📜 History Viewer")

    sub = st.selectbox("Which history to view?", ["Loan history", "Trailer assignments", "Driver assignments"])

    # -------------------------
    # LOAN HISTORY
    # -------------------------
    if sub == "Loan history":
        entity_type = st.selectbox("Entity type", ["truck", "trailer"])

        # Select truck or trailer
        if entity_type == "truck":
            trucks_df = get_trucks()
            sel = st.selectbox(
                "Select truck",
                trucks_df['truck_id'].tolist(),
                format_func=lambda x: trucks_df.loc[trucks_df['truck_id'] == x, 'number'].iloc[0]
            )
        else:
            trailers_df = get_trailers()
            sel = st.selectbox(
                "Select trailer",
                trailers_df['trailer_id'].tolist(),
                format_func=lambda x: trailers_df.loc[trailers_df['trailer_id'] == x, 'number'].iloc[0]
            )

        # ---- Date range picker ----
        today = date.today()
        if "loan_hist_start" not in st.session_state:
            st.session_state["loan_hist_start"] = COMPANY_START
        if "loan_hist_end" not in st.session_state:
            st.session_state["loan_hist_end"] = today

        with st.expander("Select Date Range", expanded=True):
            cols = st.columns(2)
            st.session_state["loan_hist_start"] = cols[0].date_input(
                "Start Date", value=st.session_state["loan_hist_start"]
            )
            st.session_state["loan_hist_end"] = cols[1].date_input(
                "End Date", value=st.session_state["loan_hist_end"]
            )

        if st.session_state["loan_hist_start"] < COMPANY_START:
            st.session_state["loan_hist_start"] = COMPANY_START
        if st.session_state["loan_hist_end"] < st.session_state["loan_hist_start"]:
            st.session_state["loan_hist_end"] = st.session_state["loan_hist_start"]

        start_date = st.session_state["loan_hist_start"]
        end_date = st.session_state["loan_hist_end"]

        # Load loan history (via helper or direct)
        rows = get_loan_history(entity_type, sel)
        df = pd.DataFrame(rows)

        # Filter by chosen range
        if not df.empty and 'date' in df.columns:
            df = df[df['date'].between(start_date.isoformat(), end_date.isoformat())]

        if df.empty:
            st.info("No loan records found in this range.")
        else:
            # Dynamically detect amount column
            amt_col = next((c for c in df.columns if "amount" in c.lower() or "loan" in c.lower()), None)
            if amt_col:
                total_amt = pd.to_numeric(df[amt_col], errors="coerce").fillna(0).sum()
                st.metric("Total Loan Amount", f"${total_amt:,.2f}")
            else:
                st.info("No amount column found in loan history.")

            st.dataframe(df, use_container_width=True)
            export_buttons(df, f"{entity_type}_loan_history", f"{entity_type.capitalize()} Loan History")

    # -------------------------
    # TRAILER ASSIGNMENTS (with date range + robust truck join)
    # -------------------------
    elif sub == "Trailer assignments":
        trailers_df = get_trailers()
        sel = st.selectbox(
            "Select trailer",
            trailers_df['trailer_id'].tolist(),
            format_func=lambda x: trailers_df.loc[trailers_df['trailer_id'] == x, 'number'].iloc[0]
        )

        # Date range defaults
        today = date.today()
        if "trailer_hist_start" not in st.session_state:
            st.session_state["trailer_hist_start"] = COMPANY_START
        if "trailer_hist_end" not in st.session_state:
            st.session_state["trailer_hist_end"] = today

        with st.expander("Select Date Range", expanded=True):
            c1, c2 = st.columns(2)
            st.session_state["trailer_hist_start"] = c1.date_input("Start Date", value=st.session_state["trailer_hist_start"])
            st.session_state["trailer_hist_end"] = c2.date_input("End Date", value=st.session_state["trailer_hist_end"])

        if st.session_state["trailer_hist_start"] < COMPANY_START:
            st.session_state["trailer_hist_start"] = COMPANY_START
        if st.session_state["trailer_hist_end"] < st.session_state["loan_hist_start"]:
            st.session_state["trailer_hist_end"] = st.session_state["loan_hist_start"]

        start_date = st.session_state["trailer_hist_start"]
        end_date = st.session_state["trailer_hist_end"]

        conn_ta = get_db_connection()
        try:
            df = pd.read_sql_query(
                """
                SELECT 
                    h.rowid AS ID,
                    t.number AS Trailer,
                    tr.number AS Truck,
                    h.start_date,
                    h.end_date
                FROM trailer_truck_history h
                LEFT JOIN trailers t ON t.trailer_id = h.trailer_id
                LEFT JOIN trucks tr ON tr.truck_id = h.truck_id
                WHERE h.trailer_id = ?
                ORDER BY h.start_date DESC
                """,
                conn_ta,
                params=(sel,),
            )
        finally:
            conn_ta.close()

        if not df.empty:
            for col in ["start_date", "end_date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

            s, e = start_date, end_date
            def overlaps(row):
                sd = row.get("start_date")
                ed = row.get("end_date")
                if pd.isna(sd):
                    return False
                if pd.isna(ed):
                    ed = date.max
                return (sd <= e and ed >= s)

            df = df[df.apply(overlaps, axis=1)]

        if df.empty:
            st.info("No trailer assignments found in this range.")
        else:
            st.dataframe(df, use_container_width=True)
            export_buttons(df, "trailer_assignment_history", "Trailer Assignments History")

    # -------------------------
    # DRIVER ASSIGNMENTS (with date range)
    # -------------------------
    elif sub == "Driver assignments":
        drivers_df = get_drivers()
        sel = st.selectbox(
            "Select driver",
            drivers_df['driver_id'].tolist(),
            format_func=lambda x: drivers_df.loc[drivers_df['driver_id'] == x, 'name'].iloc[0]
        )

        # Date range defaults
        today = date.today()
        if "driver_hist_start" not in st.session_state:
            st.session_state["driver_hist_start"] = COMPANY_START
        if "driver_hist_end" not in st.session_state:
            st.session_state["driver_hist_end"] = today

        with st.expander("Select Date Range", expanded=True):
            cols = st.columns(2)
            st.session_state["driver_hist_start"] = cols[0].date_input(
                "Start Date", value=st.session_state["driver_hist_start"]
            )
            st.session_state["driver_hist_end"] = cols[1].date_input(
                "End Date", value=st.session_state["driver_hist_end"]
            )

        if st.session_state["driver_hist_start"] < COMPANY_START:
            st.session_state["driver_hist_start"] = COMPANY_START
        if st.session_state["driver_hist_end"] < st.session_state["loan_hist_start"]:
            st.session_state["driver_hist_end"] = st.session_state["loan_hist_start"]


        start_date = st.session_state["driver_hist_start"]
        end_date = st.session_state["driver_hist_end"]

        # Query: join to get readable Driver name and Truck number
        conn_da = get_db_connection()
        try:
            df = pd.read_sql_query(
                """
                SELECT 
                    h.rowid AS ID,
                    d.name AS Driver,
                    tr.number AS Truck,
                    h.start_date,
                    h.end_date
                FROM driver_assignment_history h
                LEFT JOIN drivers d ON d.driver_id = h.driver_id
                LEFT JOIN trucks tr ON tr.truck_id = h.truck_id
                WHERE h.driver_id = ?
                ORDER BY h.start_date DESC
                """,
                conn_da,
                params=(sel,),
            )
        finally:
            conn_da.close()

        # Apply overlap date filter (same logic as trailer assignments)
        if not df.empty:
            # Normalize to date objects
            for col in ["start_date", "end_date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

            s = start_date
            e = end_date

            def overlaps(row):
                sd = row.get("start_date")
                ed = row.get("end_date")
                if pd.isna(sd):
                    return False
                if pd.isna(ed):
                    ed = date.max  # treat NULL end as ongoing
                return (sd <= e and ed >= s)

            df = df[df.apply(overlaps, axis=1)]

        if df.empty:
            st.info("No driver assignments found in this range.")
        else:
            st.dataframe(df, use_container_width=True)
            export_buttons(df, "driver_assignment_history", "Driver Assignments History")

# -------------------------
# Dispatchers Management (new page)
# -------------------------
elif page == "Dispatchers":
    st.title("Dispatchers")
    def render_dispatcher_management_ui():
        st.subheader("🚛 Dispatchers Management")

        # Add new dispatcher form
        with st.form("add_dispatcher_form", clear_on_submit=True):
            new_name = st.text_input("Dispatcher name", key="new_dispatcher_name")
            new_phone = st.text_input("Phone", key="new_dispatcher_phone")
            new_email = st.text_input("Email", key="new_dispatcher_email")
            new_notes = st.text_area("Notes", key="new_dispatcher_notes")
            add_submitted = st.form_submit_button("Add dispatcher")
        if add_submitted:
            if not new_name.strip():
                st.error("Dispatcher name is required.")
            else:
                try:
                    add_dispatcher(new_name, phone=new_phone, email=new_email, notes=new_notes)
                    st.success(f"Dispatcher '{new_name}' added.")
                    safe_rerun()
                except Exception as e:
                    st.error(f"Error adding dispatcher: {e}")

        st.markdown("----")
        dispatchers = get_all_dispatchers()
        trucks_df = get_trucks()

        if not dispatchers:
            st.info("No dispatchers yet. Use the form above to add one.")
        else:
            for d in dispatchers:
                exp = st.expander(f"{d['name']} (ID {d['dispatcher_id']})", expanded=False)
                with exp:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        edit_name = st.text_input("Name", value=d.get('name', ''), key=f"name_{d['dispatcher_id']}")
                        edit_phone = st.text_input("Phone", value=d.get('phone', ''), key=f"phone_{d['dispatcher_id']}")
                        edit_email = st.text_input("Email", value=d.get('email', ''), key=f"email_{d['dispatcher_id']}")
                        edit_notes = st.text_area("Notes", value=d.get('notes', ''), key=f"notes_{d['dispatcher_id']}")
                    with col2:
                        # show and edit assignments
                        assigned = get_trucks_for_dispatcher(d['dispatcher_id'])
                        truck_options = [f"{int(r['truck_id'])} - {r['number']}" for _, r in trucks_df.iterrows()]
                        default_sel = [f"{tid} - {trucks_df[trucks_df['truck_id']==tid]['number'].iloc[0]}" for tid in assigned if tid in trucks_df['truck_id'].tolist()]
                        sel = st.multiselect("Assigned trucks", options=truck_options, default=default_sel, key=f"assign_{d['dispatcher_id']}")
                        sel_ids = [int(s.split(" - ", 1)[0]) for s in sel]

                    cols = st.columns([1,1,1])
                    if cols[0].button("Save changes", key=f"save_{d['dispatcher_id']}"):
                        try:
                            update_dispatcher(d['dispatcher_id'], edit_name, phone=edit_phone, email=edit_email, notes=edit_notes)
                            assign_trucks_to_dispatcher(d['dispatcher_id'], sel_ids)
                            st.success("Saved.")
                            safe_rerun()
                        except Exception as e:
                            st.error(f"Error saving dispatcher: {e}")

                    if cols[1].button("Delete dispatcher", key=f"delete_{d['dispatcher_id']}"):
                        # a simple confirm flow:
                        if st.checkbox(f"Confirm delete dispatcher {d['name']} (ID {d['dispatcher_id']})", key=f"confirm_del_{d['dispatcher_id']}"):
                            try:
                                delete_dispatcher(d['dispatcher_id'])
                                st.success("Dispatcher deleted.")
                                safe_rerun()
                            except Exception as e:
                                st.error(f"Error deleting dispatcher: {e}")

                    if cols[2].button("Save assignments only", key=f"assign_only_{d['dispatcher_id']}"):
                        try:
                            assign_trucks_to_dispatcher(d['dispatcher_id'], sel_ids)
                            st.success("Assignments saved.")
                            safe_rerun()
                        except Exception as e:
                            st.error(f"Error saving assignments: {e}")

    render_dispatcher_management_ui()

# -------------------------
# Settings: reset / backup / restore
# -------------------------
elif page == "Settings":
    st.header("⚙️ Settings")

    # --- Reset DB ---
    st.subheader("Reset Database (Deletes all data)")
    if st.button("Reset Entire Database"):
        confirm = st.checkbox("⚠️ Confirm Reset Entire Database (IRREVERSIBLE)", key="confirm_reset_all")
        if confirm:
            try:
                close_all_db_connections_if_any()
                if os.path.exists(DB_FILE):
                    os.remove(DB_FILE)
                # Recreate schema via your init function
                try:
                    init_database()
                    init_history_tables()
                    ensure_dispatcher_tables()
                    ensure_truck_dispatcher_link()
                except Exception:
                    st.warning("init_database()/init_history_tables() failed; DB file removed.")
                st.success("Database reset successfully!")
                safe_rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")
                st.text(traceback.format_exc())

    # --- Reset individual tables ---
    st.subheader("Reset Individual Tables")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset Trucks"):
            reset_trucks(); st.success("Trucks table reset!")
        if st.button("Reset Trailers"):
            reset_trailers(); st.success("Trailers table reset!")
        if st.button("Reset Drivers"):
            reset_drivers(); st.success("Drivers table reset!")
    with col2:
        if st.button("Reset Expenses"):
            reset_expenses(); st.success("Expenses table reset!")
        if st.button("Reset Income"):
            reset_income(); st.success("Income table reset!")

    # --- Backup / Restore ---
    st.markdown("---")
    st.subheader("Backup / Restore Database")

    # Export full .db file
    if st.button("Export full .db file"):
        try:
            db_bytes = backup_db_file_bytes()
            st.download_button("Download fleet_management.db", data=db_bytes, file_name="fleet_management.db", mime="application/x-sqlite3")
        except Exception as e:
            st.error(f"Export failed: {e}")
            st.text(traceback.format_exc())

    # Export SQL dump (text)
    if st.button("Export SQL dump (text)"):
        try:
            dump_bytes = export_sql_dump_bytes()
            st.download_button("Download SQL dump", data=dump_bytes, file_name="fleet_management_dump.sql", mime="text/sql")
        except Exception as e:
            st.error(f"SQL dump failed: {e}")
            st.text(traceback.format_exc())

    st.write("Upload a backup .db file to restore / replace the current database (the app will create a safety backup first).")

    # Use a form to avoid checkbox/button ordering issues
    with st.form("restore_form"):
        restore_file = st.file_uploader("Upload fleet_management.db", type=['db', 'sqlite', 'sqlite3'], key="restore_db_file_form")
        confirm_restore = st.checkbox("I confirm: overwrite current database with uploaded file (I have a backup)", key="confirm_restore_form")
        submit_restore = st.form_submit_button("Restore Now")

    def _restore_from_bytes(data_bytes):
        try:
            st.write({"debug": "entered_restore_func", "bytes": len(data_bytes) if data_bytes else 0})
            if not data_bytes:
                st.error("Uploaded file is empty.")
                return False

            st.info(f"Uploaded bytes: {len(data_bytes):,}")

            # 0) Paths (absolute)
            app_dir = os.path.dirname(os.path.abspath(__file__))
            live_db = os.path.abspath(DB_FILE) if os.path.isabs(DB_FILE) else os.path.join(app_dir, DB_FILE)
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            pre_restore_backup = f"{live_db}.pre_restore.{timestamp}.backup"

            # 1) Best-effort close of long-lived DB handles
            try:
                if "close_all_db_connections" in globals():
                    close_all_db_connections()
                elif "close_all_db_connections_if_any" in globals():
                    close_all_db_connections_if_any()
                st.write({"debug": "connections_closed"})
            except Exception as e:
                st.warning(f"Could not close connections cleanly: {e}")

            # 2) Write uploaded bytes to a temp file in same folder (same drive)
            tmp = None
            try:
                db_dir = os.path.dirname(live_db) or "."
                tmp = tempfile.NamedTemporaryFile(delete=False, dir=db_dir, prefix="restore_tmp_", suffix=".db")
                tmp.write(data_bytes)
                tmp.flush()
                tmp.close()
                tmp_path = tmp.name
                st.write({"debug": "temp_written", "tmp_path": tmp_path})
            except Exception as e:
                st.error(f"Failed to write temp file for upload: {e}")
                st.text(traceback.format_exc())
                if tmp and os.path.exists(tmp.name):
                    try:
                        os.remove(tmp.name)
                    except Exception:
                        pass
                return False

            # 3) Create a timestamped pre-restore backup of the CURRENT live DB
            if os.path.exists(live_db):
                try:
                    shutil.copy2(live_db, pre_restore_backup)
                    st.write(f"Created pre-restore backup: {pre_restore_backup}")
                except Exception as e:
                    st.warning(f"Could not create pre-restore backup: {e}")

            # 4) Try to move the live DB out of the way first (reduces Windows lock issues)
            moved_old = None
            if os.path.exists(live_db):
                backup_old = f"{os.path.splitext(live_db)[0]}_old_{timestamp}.db"
                for attempt in range(6):  # retry a few times
                    try:
                        os.replace(live_db, backup_old)
                        moved_old = backup_old
                        st.write({"debug": "live_moved_aside", "to": backup_old})
                        break
                    except PermissionError:
                        time.sleep(0.3)
                    except Exception as e:
                        st.warning(f"Could not move live DB aside: {e}")
                        break

            # 5) Try atomic replace; on Windows, if still locked, retry a few times; then fallback to copy
            replaced = False
            last_err = None
            for attempt in range(6):
                try:
                    os.replace(tmp_path, live_db)
                    replaced = True
                    st.write({"debug": "replace_success"})
                    break
                except PermissionError as e:
                    last_err = e
                    try:
                        if "close_all_db_connections" in globals():
                            close_all_db_connections()
                        elif "close_all_db_connections_if_any" in globals():
                            close_all_db_connections_if_any()
                    except Exception:
                        pass
                    time.sleep(0.35)
                except OSError as e:
                    last_err = e
                    break

            if not replaced:
                # Fallback: copy then remove temp
                try:
                    shutil.copy2(tmp_path, live_db)
                    os.remove(tmp_path)
                    replaced = True
                    st.success("Database file copied into place (fallback).")
                    st.write({"debug": "copy_fallback_success"})
                except Exception as e2:
                    # Attempt rollback: put the previous live back if we moved it aside
                    if moved_old and not os.path.exists(live_db):
                        try:
                            os.replace(moved_old, live_db)
                        except Exception:
                            pass
                    st.error(f"Failed to move uploaded DB into place: {last_err or e2}")
                    st.text(traceback.format_exc())
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
                    return False

            # Temp cleaned by replace above; if still present, remove
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

            st.success("Database file replaced on disk.")

            # 6) Integrity check and reopen
            try:
                test_conn = sqlite3.connect(live_db)
                cur = test_conn.cursor()
                cur.execute("PRAGMA integrity_check;")
                result = cur.fetchone()
                cur.close()
                test_conn.close()
                st.write({"debug": "integrity_result", "result": result[0] if result else None})
                if result and str(result[0]).lower() == "ok":
                    st.success("Restored DB integrity check passed.")
                else:
                    st.error(f"DB integrity check failed: {result}")
                    if os.path.exists(pre_restore_backup):
                        try:
                            try:
                                if "close_all_db_connections" in globals():
                                    close_all_db_connections()
                                elif "close_all_db_connections_if_any" in globals():
                                    close_all_db_connections_if_any()
                            except Exception:
                                pass
                            os.replace(pre_restore_backup, live_db)
                            st.warning("Original DB restored from pre-restore backup (integrity check failed).")
                        except Exception:
                            st.error("Could not restore the original DB from backup automatically. Manual restore needed.")
                    return False
            except Exception as e:
                st.error(f"Could not open/verify restored DB: {e}")
                st.text(traceback.format_exc())
                if os.path.exists(pre_restore_backup):
                    try:
                        if "close_all_db_connections" in globals():
                            close_all_db_connections()
                        elif "close_all_db_connections_if_any" in globals():
                            close_all_db_connections_if_any()
                    except Exception:
                        pass
                    try:
                        os.replace(pre_restore_backup, live_db)
                        st.warning("Original DB restored from pre-restore backup (open/verify error).")
                    except Exception:
                        st.error("Automatic restoration from backup failed; manual intervention required.")
                return False

            # 7) Reinitialize app connection
            try:
                if "close_all_db_connections" in globals():
                    close_all_db_connections()
                elif "close_all_db_connections_if_any" in globals():
                    close_all_db_connections_if_any()
                _ = get_db_connection()  # force-open fresh handle
                st.write({"debug": "reopened_connection"})
            except Exception as e:
                st.warning(f"Reopen warning (non-fatal): {e}")

            st.success("Restore completed. Reloading app to pick up the new database...")
            safe_rerun()
            return True

        except Exception as e:
            st.error(f"Unexpected error during restore: {e}")
            st.text(traceback.format_exc())
            return False

    # -------------------------
    # Database Backup / Restore
    # -------------------------
    st.subheader("📦 Database Backup / Restore")

    # Optional: show current DB path to avoid confusion
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        live_db = os.path.abspath(DB_FILE) if os.path.isabs(DB_FILE) else os.path.join(app_dir, DB_FILE)
        st.caption(f"DB file: {live_db}")
    except Exception:
        pass

    # Backup download button (optional, keep if you already have one)
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("Download Current DB"):
            try:
                with open(live_db, "rb") as f:
                    db_bytes = f.read()
                st.download_button(
                    label="Click to download fleet_management.db",
                    data=db_bytes,
                    file_name="fleet_management.db",
                    mime="application/octet-stream",
                )
            except Exception as e:
                st.error(f"Could not read DB for download: {e}")

    # Restore form
    with st.form("db_restore_form"):
        restore_file = st.file_uploader("Upload DB backup (.db)", type=["db", "sqlite"])
        confirm_restore = st.checkbox("I understand this will overwrite current data.")
        submit_restore = st.form_submit_button("Restore Database")

    # Trigger restore when form submitted
    if submit_restore:
        st.write({"debug": "restore_submit_clicked"})
        if restore_file is None:
            st.error("Please choose a .db file to upload before clicking Restore.")
        elif not confirm_restore:
            st.warning("Please check the confirm checkbox to proceed.")
        else:
            try:
                file_bytes = restore_file.getvalue()
                st.write({"debug": "restore_file_read", "size": len(file_bytes) if file_bytes else 0})
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
                st.text(traceback.format_exc())
                file_bytes = None

            if file_bytes:
                with st.spinner("Performing restore..."):
                    ok = _restore_from_bytes(file_bytes)
                    st.write({"debug": "restore_returned", "ok": ok})
                    if not ok:
                        st.error("Restore did not complete successfully. Check messages above.")

# ============================================================================
# USER MANAGEMENT PAGE (Admin Only)
# ============================================================================
elif page == "👥 User Management":
    # Access already checked by sidebar navigation
    
    st.title("👥 User Management")
    
    tab1, tab2, tab3 = st.tabs(["View Users", "Add User", "Activity Logs"])
    
    with tab1:
        st.subheader("Current Users")
        conn = get_db_connection()
        users_df = pd.read_sql_query("""
            SELECT user_id, username, full_name, email, role, allowed_pages, is_active, created_at, last_login
            FROM users
            ORDER BY created_at DESC
        """, conn)
        conn.close()
    
        if not users_df.empty:
            for _, user in users_df.iterrows():
                # Parse allowed pages
                try:
                    user_pages = json.loads(user['allowed_pages']) if user['allowed_pages'] else []
                except:
                    user_pages = []
            
                pages_preview = ", ".join(user_pages[:3]) + ("..." if len(user_pages) > 3 else "")
            
                with st.expander(f"{'✅' if user['is_active'] else '❌'} {user['username']} - {user['full_name'] or 'No name'} ({user['role']})"):
                    col1, col2 = st.columns(2)
                
                    with col1:
                        st.write(f"**User ID:** {user['user_id']}")
                        st.write(f"**Username:** {user['username']}")
                        st.write(f"**Full Name:** {user['full_name'] or 'Not set'}")
                        st.write(f"**Email:** {user['email'] or 'Not set'}")
                
                    with col2:
                        st.write(f"**Role:** {user['role']}")
                        st.write(f"**Status:** {'Active' if user['is_active'] else 'Inactive'}")
                        st.write(f"**Created:** {user['created_at']}")
                        st.write(f"**Last Login:** {user['last_login'] or 'Never'}")
                
                    st.markdown("**Allowed Pages:**")
                    st.write(", ".join(user_pages) if user_pages else "No pages assigned")
                
                    # Edit Page Permissions
                    st.markdown("---")
                    st.markdown("**Edit Page Permissions:**")
                
                    if user['role'] == 'admin':
                        st.info("👑 Admin users have access to all pages automatically")
                        new_pages = ALL_PAGES
                    else:
                        new_pages = st.multiselect(
                            "Update Allowed Pages",
                            ALL_PAGES,
                            default=user_pages,
                            key=f"pages_{user['user_id']}"
                        )
                
                    if st.button("💾 Update Pages", key=f"update_pages_{user['user_id']}"):
                        conn = get_db_connection()
                        conn.execute("UPDATE users SET allowed_pages = ? WHERE user_id = ?", 
                               (json.dumps(new_pages), user['user_id']))
                        conn.commit()
                        conn.close()
                        log_session_action(st.session_state.user["user_id"], st.session_state.user["username"], 
                                         f"updated_pages_for_{user['username']}")
                        st.success("Page permissions updated!")
                        st.rerun()
                
                    # Other Actions
                    st.markdown("---")
                    st.markdown("**Other Actions:**")
                    action_col1, action_col2, action_col3 = st.columns(3)
                
                    with action_col1:
                        new_role = st.selectbox("Change Role", ["admin", "dispatcher", "accountant", "driver", "viewer"], 
                                               index=["admin", "dispatcher", "accountant", "driver", "viewer"].index(user['role']),
                                               key=f"role_{user['user_id']}")
                        if st.button("Update Role", key=f"update_role_{user['user_id']}"):
                            conn = get_db_connection()
                            # If changing to admin, give all pages
                            if new_role == "admin":
                                conn.execute("UPDATE users SET role = ?, allowed_pages = ? WHERE user_id = ?", 
                                           (new_role, json.dumps(ALL_PAGES), user['user_id']))
                            else:
                                conn.execute("UPDATE users SET role = ? WHERE user_id = ?", (new_role, user['user_id']))
                            conn.commit()
                            conn.close()
                            log_session_action(st.session_state.user["user_id"], st.session_state.user["username"], 
                                             f"changed_role_for_{user['username']}_to_{new_role}")
                            st.success(f"Role updated to {new_role}")
                            st.rerun()
                
                    with action_col2:
                        if user['is_active']:
                            if st.button("🚫 Deactivate", key=f"deactivate_{user['user_id']}"):
                                if user['username'] == 'admin':
                                    st.error("Cannot deactivate admin user")
                                else:
                                    conn = get_db_connection()
                                    conn.execute("UPDATE users SET is_active = 0 WHERE user_id = ?", (user['user_id'],))
                                    conn.commit()
                                    conn.close()
                                    log_session_action(st.session_state.user["user_id"], st.session_state.user["username"], 
                                                     f"deactivated_user_{user['username']}")
                                    st.success("User deactivated")
                                    st.rerun()
                        else:
                            if st.button("✅ Activate", key=f"activate_{user['user_id']}"):
                                conn = get_db_connection()
                                conn.execute("UPDATE users SET is_active = 1 WHERE user_id = ?", (user['user_id'],))
                                conn.commit()
                                conn.close()
                                log_session_action(st.session_state.user["user_id"], st.session_state.user["username"], 
                                                 f"activated_user_{user['username']}")
                                st.success("User activated")
                                st.rerun()
                
                    with action_col3:
                        if st.button("🔑 Reset Password", key=f"reset_{user['user_id']}"):
                            new_password = "reset123"
                            password_hash = hash_password(new_password)
                            conn = get_db_connection()
                            conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (password_hash, user['user_id']))
                            conn.commit()
                            conn.close()
                            log_session_action(st.session_state.user["user_id"], st.session_state.user["username"], 
                                             f"reset_password_for_{user['username']}")
                            st.success(f"Password reset to: `{new_password}`")
        else:
            st.info("No users found")
    
    with tab2:
        st.subheader("Add New User")
    
        with st.form("add_user_form"):
            new_username = st.text_input("Username*", help="Unique username for login")
            new_password = st.text_input("Password*", type="password", help="Initial password")
            new_full_name = st.text_input("Full Name")
            new_email = st.text_input("Email")
            new_role = st.selectbox("Role*", ["admin", "dispatcher", "accountant", "driver", "viewer"])
        
            st.markdown("**Select Pages This User Can Access:**")
        
            # If admin, give access to all pages automatically
            if new_role == "admin":
                st.info("👑 Admin users have access to all pages automatically")
                selected_pages = ALL_PAGES
            else:
                # Multi-select for pages
                selected_pages = st.multiselect(
                    "Allowed Pages",
                    ALL_PAGES,
                    default=["Dashboard"],
                    help="Select which pages this user can access"
                )
        
            submit_user = st.form_submit_button("Create User")
        
            if submit_user:
                if not new_username or not new_password:
                    st.error("Username and password are required")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif not selected_pages and new_role != "admin":
                    st.error("Please select at least one page for this user")
                else:
                    try:
                        conn = get_db_connection()
                        password_hash = hash_password(new_password)
                        allowed_pages_json = json.dumps(selected_pages)
                        conn.execute("""
                            INSERT INTO users (username, password_hash, full_name, email, role, allowed_pages, is_active)
                            VALUES (?, ?, ?, ?, ?, ?, 1)
                        """, (new_username, password_hash, new_full_name, new_email, new_role, allowed_pages_json))
                        conn.commit()
                        conn.close()
                        log_session_action(st.session_state.user["user_id"], st.session_state.user["username"], 
                                         f"created_user_{new_username}")
                        st.success(f"User '{new_username}' created successfully!")
                        st.info(f"**Login credentials:**\nUsername: `{new_username}`\nPassword: `{new_password}`\n\n**Allowed Pages:** {', '.join(selected_pages)}")
                    except Exception as e:
                        st.error(f"Failed to create user: {e}")
    
    with tab3:
        st.subheader("Activity Logs")
        
        log_filter = st.selectbox("Filter by Action", ["All", "login", "logout", "failed_login_attempt", "created_user", "changed_role", "deactivated_user", "activated_user"])
        
        conn = get_db_connection()
        if log_filter == "All":
            logs_df = pd.read_sql_query("""
                SELECT * FROM session_logs
                ORDER BY timestamp DESC
                LIMIT 100
            """, conn)
        else:
            logs_df = pd.read_sql_query("""
                SELECT * FROM session_logs
                WHERE action LIKE ?
                ORDER BY timestamp DESC
                LIMIT 100
            """, conn, params=(f"%{log_filter}%",))
        conn.close()
        
        if not logs_df.empty:
            st.dataframe(logs_df, use_container_width=True)
        else:
            st.info("No activity logs found")

elif page == "Reports":
    st.header("📊 Reports")

    # -----------------------------
    # Date Preset Toolbar (KEEPING)
    # -----------------------------
    with st.expander("Date Range", expanded=True):
        today = date.today()
        presets = {
            "This Month": (today.replace(day=1), today),
            "Last 30 Days": (today - timedelta(days=30), today),
            "Last Month": ((today.replace(day=1) - timedelta(days=1)).replace(day=1),
                           today.replace(day=1) - timedelta(days=1)),
            "Year to Date": (date(today.year, 1, 1), today),
            "All Time": (COMPANY_START, today),
        }
        cols = st.columns(len(presets) + 2)
        for i, (label, (p_start, p_end)) in enumerate(presets.items()):
            if cols[i].button(label):
                st.session_state["report_start_date"] = p_start
                st.session_state["report_end_date"] = p_end

        if "report_start_date" not in st.session_state:
            st.session_state["report_start_date"] = max(today.replace(day=1), COMPANY_START)
        if "report_end_date" not in st.session_state:
            st.session_state["report_end_date"] = today

        st.session_state["report_start_date"] = cols[-2].date_input(
            "Start", value=st.session_state["report_start_date"]
        )
        st.session_state["report_end_date"] = cols[-1].date_input(
            "End", value=st.session_state["report_end_date"]
        )
        if st.session_state["report_start_date"] < COMPANY_START:
            st.session_state["report_start_date"] = COMPANY_START
        if st.session_state["report_end_date"] < st.session_state["report_start_date"]:
            st.session_state["report_end_date"] = st.session_state["report_start_date"]

    start_date = st.session_state["report_start_date"]
    end_date = st.session_state["report_end_date"]

    # -----------------------------
    # Local helpers + guards
    # -----------------------------
    def _fmt_money(v):
        try:
            return f"${float(v):,.2f}"
        except Exception:
            return v

    def _days_between(a, b):
        return (b - a).days + 1

    # Use your existing helper; if missing, add a fallback
    if "safe_read_sql" not in globals():
        def safe_read_sql(query, conn, params=None):
            try:
                return pd.read_sql_query(query, conn, params=params)
            except Exception as e:
                st.warning(f"Query failed: {e}")
                return pd.DataFrame()

    def table_exists(name: str) -> bool:
        try:
            c = get_db_connection().cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,))
            return c.fetchone() is not None
        except Exception:
            return False

    def column_exists(table: str, column: str) -> bool:
        try:
            c = get_db_connection().cursor()
            c.execute(f"PRAGMA table_info({table});")
            cols = [r[1] for r in c.fetchall()]
            return column in cols
        except Exception:
            return False

    def _to_date(x, default=None):
        # Robust missing detection without pd.NA truthiness
        try:
            import pandas as pd
        except Exception:
            pass

        # Treat clearly missing values
        if x is None or x == "":
            return default

        # Handle pandas NA/NaT safely
        try:
            # pd.isna covers pd.NA, NaT, np.nan
            if 'pd' in globals():
                if pd.isna(x):
                    return default
        except Exception:
            pass

        # Try parsing YYYY-MM-DD fast path
        try:
            return datetime.strptime(str(x), "%Y-%m-%d").date()
        except Exception:
            pass

        # Fallback to pandas parser (handles many formats)
        try:
            if 'pd' in globals():
                dt = pd.to_datetime(x, errors="coerce")
                if pd.isna(dt):
                    return default
                return dt.date()
        except Exception:
            pass

        return default

    # Your existing dispatcher map (KEEP)
    def get_dispatcher_map(conn):
        try:
            df = pd.read_sql_query(
                """
                SELECT t.truck_id, COALESCE(d.name, '') AS dispatcher_name
                FROM trucks t
                LEFT JOIN dispatchers d ON d.dispatcher_id = t.dispatcher_id
                """,
                conn
            )
            base_map = dict(zip(df["truck_id"], df["dispatcher_name"]))

            try:
                df2 = pd.read_sql_query(
                    """
                    SELECT dt.truck_id, d.name AS dispatcher_name
                    FROM dispatcher_trucks dt
                    JOIN dispatchers d ON d.dispatcher_id = dt.dispatcher_id
                    """,
                    conn
                )
                fallback_map = dict(zip(df2["truck_id"], df2["dispatcher_name"]))
            except Exception:
                fallback_map = {}

            out = {}
            for tid in base_map.keys():
                name = base_map.get(tid) or ""
                if not name:
                    name = fallback_map.get(tid, "")
                out[tid] = name
            return out
        except Exception:
            return {}

    conn = get_db_connection()

    # Quick schema flags
    has_trucks = table_exists("trucks")
    has_expenses = table_exists("expenses")
    has_income = table_exists("income")
    has_assignments = table_exists("assignments")
    has_dispatchers = table_exists("dispatchers")
    has_loans_hist = table_exists("loans_history")
    has_tr_hist = table_exists("trailer_truck_history")
    has_trailers = table_exists("trailers")

    # Truck list (KEEP)
    conn_tmp = get_db_connection()
    try:
        trucks_df = safe_read_sql("SELECT truck_id, number FROM trucks", conn) if has_trucks else pd.DataFrame(columns=["truck_id","number"])
    # Trailer number by truck (via trailers.truck_id)
        trailer_map = {}
        if has_trailers and column_exists("trailers", "truck_id"):
            tr_df = safe_read_sql(
                """
                SELECT tr.trailer_id, tr.number AS trailer_number, tr.truck_id
                FROM trailers tr
                WHERE tr.truck_id IS NOT NULL
                """,
                conn,
            )
            if tr_df is not None and not tr_df.empty:
                trailer_map = dict(zip(tr_df["truck_id"], tr_df["trailer_number"]))
    finally:
        conn_tmp.close()

    # -------------------------------------------
    # Dispatcher-aware Income and Expenses
    # -------------------------------------------
    # Income per truck and dispatcher by date overlap
    conn_tmp = get_db_connection()
    try:
        if has_income and column_exists("income", "truck_id"):
            if has_assignments and has_dispatchers:
                q_inc = """
                    SELECT 
                        i.truck_id,
                        tr.number AS truck_number,
                        disp.dispatcher_id,
                        disp.name AS dispatcher_name,
                        SUM(i.amount) AS total_income
                    FROM income i
                    LEFT JOIN trucks tr ON tr.truck_id = i.truck_id
                    LEFT JOIN assignments a ON a.truck_id = i.truck_id 
                        AND DATE(a.start_date) <= DATE(i.date)
                        AND (a.end_date IS NULL OR a.end_date = '' OR DATE(a.end_date) >= DATE(i.date))
                    LEFT JOIN dispatchers disp ON disp.dispatcher_id = a.dispatcher_id
                    WHERE DATE(i.date) BETWEEN DATE(?) AND DATE(?)
                    GROUP BY i.truck_id, tr.number, disp.dispatcher_id, disp.name
                """
                income_df = safe_read_sql(q_inc, conn, [start_date.isoformat(), end_date.isoformat()])
            else:
                income_df = safe_read_sql(
                    """
                    SELECT truck_id, COALESCE(SUM(amount), 0) AS total_income
                    FROM income
                    WHERE DATE(date) BETWEEN DATE(?) AND DATE(?)
                    GROUP BY truck_id
                    """,
                    conn_tmp,
                    params=[start_date.isoformat(), end_date.isoformat()],
                )
                income_df["truck_number"] = income_df["truck_id"].map(dict(zip(trucks_df.truck_id, trucks_df.number)))
                income_df["dispatcher_id"] = None
                income_df["dispatcher_name"] = None
        else:
            income_df = pd.DataFrame(columns=["truck_id","truck_number","dispatcher_id","dispatcher_name","total_income"])
    finally:
        conn_tmp.close()

    # Expenses per truck and dispatcher by date overlap
    conn_tmp = get_db_connection()
    try:
        if has_expenses and column_exists("expenses", "truck_id"):
            if has_assignments and has_dispatchers and column_exists("expenses","date"):
                q_exp = """
                    SELECT 
                        e.truck_id,
                        tr.number AS truck_number,
                        disp.dispatcher_id,
                        disp.name AS dispatcher_name,
                        SUM(e.amount) AS total_expenses
                    FROM expenses e
                    LEFT JOIN trucks tr ON tr.truck_id = e.truck_id
                    LEFT JOIN assignments a ON a.truck_id = e.truck_id
                        AND DATE(a.start_date) <= DATE(e.date)
                        AND (a.end_date IS NULL OR a.end_date = '' OR DATE(a.end_date) >= DATE(e.date))
                    LEFT JOIN dispatchers disp ON disp.dispatcher_id = a.dispatcher_id
                    WHERE DATE(e.date) BETWEEN DATE(?) AND DATE(?)
                    GROUP BY e.truck_id, tr.number, disp.dispatcher_id, disp.name
                """
                expense_df = safe_read_sql(q_exp, conn, [start_date.isoformat(), end_date.isoformat()])
            else:
                expense_df = safe_read_sql(
                    """
                    SELECT truck_id, COALESCE(SUM(amount), 0) AS total_expenses
                    FROM expenses
                    WHERE DATE(date) BETWEEN DATE(?) AND DATE(?)
                    GROUP BY truck_id
                    """,
                    conn_tmp,
                    params=[start_date.isoformat(), end_date.isoformat()],
                )
                expense_df["truck_number"] = expense_df["truck_id"].map(dict(zip(trucks_df.truck_id, trucks_df.number)))
                expense_df["dispatcher_id"] = None
                expense_df["dispatcher_name"] = None
        else:
            expense_df = pd.DataFrame(columns=["truck_id","truck_number","dispatcher_id","dispatcher_name","total_expenses"])
    finally:
        conn_tmp.close()

    # --------------------------------------------------
    # Accurate (pro-rated) Loans per Truck using History
    # --------------------------------------------------
    MONTHLY_TO_DAILY_DIVISOR = 30.0  # change to 30.4375 if you prefer

    # Fetch loans overlapping window
    if has_loans_hist:
        df_loans = safe_read_sql(
            """
            SELECT
                lh.entity_type, -- 'truck' | 'trailer'
                lh.entity_id,
                lh.monthly_amount,
                DATE(lh.start_date) AS s,
                DATE(COALESCE(NULLIF(lh.end_date,''), '9999-12-31')) AS e
            FROM loans_history lh
            WHERE (lh.end_date IS NULL OR lh.end_date = '' OR DATE(lh.end_date) >= DATE(?))
              AND DATE(lh.start_date) <= DATE(?)
            """,
            conn,
            [start_date.isoformat(), end_date.isoformat()],
        )
        if df_loans is None:
            df_loans = pd.DataFrame(columns=["entity_type","entity_id","monthly_amount","s","e"])
    else:
        df_loans = pd.DataFrame(columns=["entity_type","entity_id","monthly_amount","s","e"])

    # Trailer-truck history intervals (if available)
    hist_rows = []
    if has_tr_hist and column_exists("trailer_truck_history","truck_id"):
        df_hist = safe_read_sql(
            """
            SELECT trailer_id, truck_id, DATE(start_date) AS s, DATE(COALESCE(NULLIF(end_date,''), '9999-12-31')) AS e
            FROM trailer_truck_history
            WHERE truck_id IS NOT NULL
              AND ( (end_date IS NULL OR end_date = '') 
                    OR NOT (DATE(end_date) < DATE(?) OR DATE(start_date) > DATE(?)) )
            """,
            conn,
            [start_date.isoformat(), end_date.isoformat()],
        )
        if df_hist is not None and not df_hist.empty:
            for _, r in df_hist.iterrows():
                hist_rows.append({
                    "trailer_id": int(r["trailer_id"]),
                    "truck_id": int(r["truck_id"]),
                    "s": _to_date(r["s"], date(1900,1,1)),
                    "e": _to_date(r["e"], date(9999,12,31)),
                })

    # Fallback: current trailer->truck mapping
    current_trailer_to_truck = {}
    if has_trailers and column_exists("trailers","truck_id"):
        df_tr_now = safe_read_sql("SELECT trailer_id, truck_id FROM trailers;", conn)
        if df_tr_now is not None:
            for _, r in df_tr_now.iterrows():
                if pd.notna(r["truck_id"]):
                    current_trailer_to_truck[int(r["trailer_id"])] = int(r["truck_id"])

    # Build truck number map
    truck_num_map = dict(zip(trucks_df.truck_id, trucks_df.number)) if not trucks_df.empty else {}

    # Pro-rate loans by day within [start_date, end_date]
    prorated_rows = []
    if df_loans is not None and not df_loans.empty:
        for _, r in df_loans.iterrows():
            et = str(r.get("entity_type") or "").lower()
            eid = r.get("entity_id")
            monthly = float(r.get("monthly_amount", 0) or 0.0)
            ls = _to_date(r.get("s"), date(1900,1,1))
            le = _to_date(r.get("e"), date(9999,12,31))
            if eid is None:
                continue

            os = max(start_date, ls)
            oe = min(end_date, le)
            if os > oe:
                continue

            daily = monthly / MONTHLY_TO_DAILY_DIVISOR
            amount_for_window = daily * _days_between(os, oe)

            if et == "truck":
                prorated_rows.append({"truck_id": int(eid), "loan_amount": amount_for_window})

            elif et == "trailer":
                tr_id = int(eid)
                if hist_rows:
                    trailer_assigns = [h for h in hist_rows if h["trailer_id"] == tr_id]
                    if not trailer_assigns:
                        t_truck = current_trailer_to_truck.get(tr_id)
                        if t_truck is not None:
                            prorated_rows.append({"truck_id": int(t_truck), "loan_amount": amount_for_window})
                        continue

                    total_assign_days = 0
                    portions = []
                    for a in trailer_assigns:
                        as_ = max(os, a["s"])
                        ae_ = min(oe, a["e"])
                        if as_ <= ae_:
                            d = _days_between(as_, ae_)
                            portions.append((a["truck_id"], d))
                            total_assign_days += d

                    if total_assign_days == 0:
                        t_truck = current_trailer_to_truck.get(tr_id)
                        if t_truck is not None:
                            prorated_rows.append({"truck_id": int(t_truck), "loan_amount": amount_for_window})
                    else:
                        for t_id, d in portions:
                            share = amount_for_window * (d / total_assign_days)
                            prorated_rows.append({"truck_id": int(t_id), "loan_amount": share})
                else:
                    t_truck = current_trailer_to_truck.get(tr_id)
                    if t_truck is not None:
                        prorated_rows.append({"truck_id": int(t_truck), "loan_amount": amount_for_window})

    if prorated_rows:
        loans_df = pd.DataFrame(prorated_rows).groupby("truck_id", as_index=False)["loan_amount"].sum()
        loans_df["truck_number"] = loans_df["truck_id"].map(truck_num_map)
    else:
        loans_df = pd.DataFrame(columns=["truck_id","loan_amount","truck_number"])

    # --------------------------------------------------
    # Merge Summary (truck + dispatcher), then display
    # --------------------------------------------------
    # Ensure dispatcher cols exist for merging
    for df_ in [income_df, expense_df]:
        if "dispatcher_id" not in df_.columns:
            df_["dispatcher_id"] = None
        if "dispatcher_name" not in df_.columns:
            df_["dispatcher_name"] = None
        if "truck_number" not in df_.columns:
            df_["truck_number"] = df_["truck_id"].map(truck_num_map)

    def _merge(a, b, on, how="outer"):
        if a is None or a.empty:
            return b.copy() if b is not None else pd.DataFrame()
        if b is None or b.empty:
            return a.copy()
        return a.merge(b, on=on, how=how)

    common_keys = ["truck_id","truck_number","dispatcher_id","dispatcher_name"]
    summary = _merge(income_df, expense_df, on=common_keys)

    # Loans are truck-level; attach to each dispatcher row for that truck
    if summary is not None and not summary.empty:
        loans_for_merge = summary[["truck_id","truck_number"]].drop_duplicates().merge(
            loans_df[["truck_id","truck_number","loan_amount"]], on=["truck_id","truck_number"], how="left"
        )
        loans_for_merge = summary[common_keys].drop_duplicates().merge(
            loans_df[["truck_id","truck_number","loan_amount"]], on=["truck_id","truck_number"], how="left"
        )
        summary = _merge(summary, loans_for_merge, on=common_keys)
    else:
        # If no summary rows, still make a base from trucks + loans
        summary = trucks_df.rename(columns={"number":"truck_number"})[["truck_id","truck_number"]].copy()
        summary["dispatcher_id"] = None
        summary["dispatcher_name"] = None
        summary = summary.merge(loans_df[["truck_id","truck_number","loan_amount"]], on=["truck_id","truck_number"], how="left")

    # Fill numeric defaults
    for c in ["total_income","total_expenses","loan_amount"]:
        if c not in summary.columns:
            summary[c] = 0.0
        summary[c] = pd.to_numeric(summary[c], errors="coerce").fillna(0.0)

    summary["total_costs"] = summary["total_expenses"] + summary["loan_amount"]
    summary["profit_loss"] = summary["total_income"] - summary["total_costs"]
    # Attach trailer by current mapping (history already used for loans)
    summary["trailer_number"] = summary["truck_id"].map(trailer_map).fillna("Not Assigned")

    # Dispatcher map for display when missing dispatcher_name
    dispatcher_map = get_dispatcher_map(conn)
    summary["dispatcher_name"] = summary["dispatcher_name"].fillna(summary["truck_id"].map(dispatcher_map))

    st.subheader("Per-Truck Summary (Income, Expenses, Loans, Dispatcher-aware)")
    if summary.empty:
        st.info("No data for the selected range.")
    else:
        disp = summary[[
        "truck_number",
        "trailer_number",
        "dispatcher_name",
        "total_income",
        "total_expenses",
        "loan_amount",
        "total_costs",
        "profit_loss"
        ]].copy()
        for c in ["total_income","total_expenses","loan_amount","total_costs","profit_loss"]:
            disp[c] = disp[c].apply(_fmt_money)

        # Top totals
        st.metric("All Trucks - Total Income", _fmt_money(summary["total_income"].sum()))
        st.metric("All Trucks - Total Expenses", _fmt_money(summary["total_expenses"].sum()))
        st.metric("All Trucks - Total Loans (Prorated)", _fmt_money(summary["loan_amount"].sum()))
        st.metric("All Trucks - Profit/Loss", _fmt_money(summary["profit_loss"].sum()))
        st.dataframe(disp.sort_values("truck_number"), use_container_width=True)

    # ---------------------------------------
    # Profit / Loss per Truck (roll-up)
    # ---------------------------------------
    st.subheader("Profit / Loss per Truck (Includes Prorated Loans)")
    if summary.empty:
        st.info("No data to display.")
    else:
        perf = summary.groupby(["truck_number"], as_index=False).agg(
            total_income=("total_income","sum"),
            total_expenses=("total_expenses","sum"),
            loan_amount=("loan_amount","sum"),
            profit_after_loans=("profit_loss","sum"),
        )
        for c in ["total_income","total_expenses","loan_amount","profit_after_loans"]:
            perf[c] = perf[c].apply(_fmt_money)

        st.metric("All Trucks - Profit After Loans", _fmt_money(summary["profit_loss"].sum()))
        st.dataframe(perf.sort_values("profit_after_loans", ascending=False), use_container_width=True)
        export_buttons(perf, "profit_loss_per_truck", "Profit-Loss per Truck Report")

    # ---------------------------------------
    # Category Breakdown per Truck (Loans as expenses)
    # ---------------------------------------
    st.subheader("Category Breakdown per Truck (Loans treated as expenses)")
    cat_df = pd.DataFrame(columns=["truck_id","category","total"])
    if has_expenses:
        cat_df = safe_read_sql(
            """
            SELECT truck_id, category, COALESCE(SUM(amount), 0) AS total
            FROM expenses
            WHERE DATE(date) BETWEEN DATE(?) AND DATE(?)
            GROUP BY truck_id, category
            """,
            conn,
            params=[start_date.isoformat(), end_date.isoformat()],
        )

    # Attach pseudo-categories for loans
    if loans_df is not None and not loans_df.empty:
        add_rows = []
        for _, row in loans_df.iterrows():
            tid = row["truck_id"]
            total_loan = float(row.get("loan_amount", 0.0) or 0.0)
            if total_loan:
                add_rows.append({"truck_id": tid, "category": "Loans (Prorated)", "total": total_loan})
        if add_rows:
            loans_cat_df = pd.DataFrame(add_rows)
            cat_df = pd.concat([cat_df, loans_cat_df], ignore_index=True)

    if trucks_df.empty:
        st.info("No trucks found.")
    else:
        # Create dropdown for truck selection (no "All Trucks" option)
        truck_options = [f"Truck {row['number']}" for _, row in trucks_df.sort_values("number").iterrows()]
        selected_truck_cat = st.selectbox("Select Truck for Category Breakdown", truck_options, key="cat_breakdown_truck_select")
        
        by_truck = cat_df.groupby("truck_id")
        
        # Show only selected truck
        truck_num = selected_truck_cat.replace("Truck ", "")
        trow = trucks_df[trucks_df["number"] == truck_num].iloc[0]
        tid = trow["truck_id"]
        tnum = trow["number"]
        
        sub = by_truck.get_group(tid) if tid in by_truck.indices else pd.DataFrame(columns=["category","total"])
        sub_display = sub[["category","total"]].rename(columns={"category":"Category","total":"Amount"})
        st.metric("Total Amount", _fmt_money(pd.to_numeric(sub_display["Amount"], errors="coerce").fillna(0.0).sum()))
        if not sub_display.empty:
            st.dataframe(sub_display.sort_values("Amount", ascending=False), use_container_width=True)
            export_buttons(sub_display, f"truck_{tnum}_category_breakdown", f"Truck {tnum} Category Breakdown")
        else:
            st.write("No expenses for this truck in the selected range.")


    # ---------------------------------------
    # Per-Truck Expense Breakdown with Deletes
    # ---------------------------------------
    st.subheader("Per-Truck Expense Breakdown with Delete Actions")
    exp_df = safe_read_sql(
        """
        SELECT e.expense_id, e.truck_id, t.number AS truck_number,
               e.category, e.amount, e.date, e.description
        FROM expenses e
        LEFT JOIN trucks t ON e.truck_id = t.truck_id
        WHERE DATE(e.date) BETWEEN DATE(?) AND DATE(?)
        ORDER BY e.date DESC, e.expense_id DESC
        """,
        conn,
        params=[start_date.isoformat(), end_date.isoformat()],
    )

    if not exp_df.empty:
        total_expenses_all = pd.to_numeric(exp_df["amount"], errors="coerce").fillna(0.0).sum()
        st.metric("All Trucks - Total Expenses (Selected Range)", _fmt_money(total_expenses_all))

    if exp_df.empty:
        st.info("No expenses in selected range.")
    else:
        # Create dropdown for truck selection (no "All Trucks" option)
        truck_list = sorted(exp_df["truck_number"].unique())
        truck_options_exp = [f"Truck {tnum}" for tnum in truck_list]
        selected_truck_exp = st.selectbox("Select Truck for Expense Breakdown", truck_options_exp, key="exp_breakdown_truck_select")
        
        # Show only selected truck
        truck_num = selected_truck_exp.replace("Truck ", "")
        group = exp_df[exp_df["truck_number"] == truck_num]
        
        truck_total = pd.to_numeric(group["amount"], errors="coerce").fillna(0.0).sum()
        st.metric("Truck Total", _fmt_money(truck_total))

        def render_row(r):
            cols = st.columns([2, 2, 2, 2, 4, 2])
            cols[0].write(r["date"])
            cols[1].write(r["category"])
            cols[2].write(_fmt_money(r["amount"]))
            cols[3].write(r.get("description", "") or "")
            cols[4].write(f"Expense ID: {int(r['expense_id'])}")
            if cols[5].button("Delete", key=f"del_exp_{int(r['expense_id'])}"):
                try:
                    conn_del = get_db_connection()
                    conn_del.execute("DELETE FROM expenses WHERE expense_id = ?", (int(r["expense_id"]),))
                    conn_del.commit()
                    conn_del.close()
                    st.success(f"Deleted expense {int(r['expense_id'])}")
                    safe_rerun()
                except Exception as e:
                    st.error(f"Failed to delete expense {int(r['expense_id'])}: {e}")

        header_cols = st.columns([2, 2, 2, 2, 4, 2])
        header_cols[0].markdown("**Date**")
        header_cols[1].markdown("**Category**")
        header_cols[2].markdown("**Amount**")
        header_cols[3].markdown("**Description**")
        header_cols[4].markdown("**Info**")
        header_cols[5].markdown("**Action**")

        for _, r in group.iterrows():
            render_row(r)
        export_buttons(group, f"truck_{truck_num}_expenses", f"Truck {truck_num} Expenses Report")

    # ---------------------------------------
    # Fuel Discounts by Truck (KEEP structure)
    # ---------------------------------------
    st.subheader("Fuel Discounts by Truck")
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    fuel_df = safe_read_sql(
        """
        SELECT 
            t.number AS truck_label,
            SUM(
                COALESCE(
                    CAST(json_extract(e.metadata, '$.discount_amount') AS REAL),
                    0
                )
            ) AS total_discount
        FROM trucks t
        LEFT JOIN expenses e 
            ON t.truck_id = e.truck_id
            AND DATE(e.date) BETWEEN DATE(?) AND DATE(?)
            AND LOWER(e.category) LIKE '%fuel%'
        GROUP BY t.number
        ORDER BY total_discount DESC;
        """,
        conn,
        params=[start_str, end_str],
    )

    if fuel_df is None or fuel_df.empty:
        st.info("No fuel discounts found in the selected range.")
    else:
        total_fuel_disc = pd.to_numeric(fuel_df["total_discount"], errors="coerce").fillna(0.0).sum()
        st.metric("All Trucks - Total Fuel Discounts", _fmt_money(total_fuel_disc))
        out = fuel_df.rename(columns={"truck_label": "Truck", "total_discount": "Fuel Discount Total"})
        st.dataframe(out.sort_values("Fuel Discount Total", ascending=False), use_container_width=True)
        export_buttons(out, "fuel_discounts", "Fuel Discounts Report")

    st.divider()
    st.header("📊 Performance Charts")
    st.caption("Visual analytics for income, expenses, profit, and performance metrics")

    # Date selector for charts
    col_chart1, col_chart2 = st.columns([2, 1])
    with col_chart1:
        chart_end_date = st.date_input(
            "Select end date for 12-week analysis",
            value=datetime.now().date(),
            key="chart_end_date"
        )
    with col_chart2:
        st.metric("Weeks Back", "12", help="Charts show 12 weeks of data ending on selected date")

    # Calculate start date (12 weeks back from selected date)
    chart_start_date = chart_end_date - timedelta(weeks=12)

    st.caption(f"Showing data from {chart_start_date} to {chart_end_date}")

    # Fetch data for all charts
    conn_charts = get_db_connection()

    # Get weekly income data
    df_income_weekly = pd.read_sql_query("""
        SELECT 
            strftime('%Y-%W', date) as week,
            date(date, 'weekday 0', '-6 days') as week_start,
            SUM(amount) as total_income
        FROM income
        WHERE DATE(date) BETWEEN DATE(?) AND DATE(?)
        GROUP BY week
        ORDER BY week
    """, conn_charts, params=(chart_start_date, chart_end_date))

    # Get weekly expense data
    df_expense_weekly = pd.read_sql_query("""
        SELECT 
            strftime('%Y-%W', date) as week,
            date(date, 'weekday 0', '-6 days') as week_start,
            SUM(amount) as total_expense
        FROM expenses
        WHERE DATE(date) BETWEEN DATE(?) AND DATE(?)
        GROUP BY week
        ORDER BY week
    """, conn_charts, params=(chart_start_date, chart_end_date))

    # Get weekly data by dispatcher
    df_dispatcher_weekly = pd.read_sql_query("""
        SELECT
            strftime('%Y-%W', i.date) as week,
            date(i.date, 'weekday 0', '-6 days') as week_start,
            COALESCE(d.name, 'Unassigned') as dispatcher,
            SUM(i.amount) as total_income
        FROM income i
        LEFT JOIN trucks t ON i.truck_id = t.truck_id
        LEFT JOIN dispatchers d ON t.dispatcher_id = d.dispatcher_id
        WHERE i.date BETWEEN ? AND ?
        GROUP BY week, dispatcher
        ORDER BY week, dispatcher
    """, conn_charts, params=(chart_start_date, chart_end_date))

    # Get weekly RPM data (Rate Per Mile)
    df_rpm_weekly = pd.read_sql_query("""
        SELECT 
            strftime('%Y-%W', date) as week,
            date(date, 'weekday 0', '-6 days') as week_start,
            SUM(amount) as total_income,
            SUM(COALESCE(loaded_miles, 0) + COALESCE(empty_miles, 0)) as total_miles
        FROM income
        WHERE DATE(date) BETWEEN DATE(?) AND DATE(?)
        GROUP BY week
        ORDER BY week
    """, conn_charts, params=(chart_start_date, chart_end_date))

    # Calculate RPM
    df_rpm_weekly['rpm'] = df_rpm_weekly.apply(
        lambda row: row['total_income'] / row['total_miles'] if row['total_miles'] > 0 else 0,
        axis=1
    )

    # Get weekly fuel data (total and by truck) – robust to missing columns/values
    df_fuel_weekly = pd.read_sql_query("""
        SELECT
            strftime('%Y-%W', e.date) AS week,
            DATE(e.date, 'weekday 0', '-6 days') AS week_start,
            e.truck_id,
            t.number AS truck_number,
            SUM(
                CASE
                    WHEN LOWER(COALESCE(e.category, '')) LIKE '%fuel%'
                    THEN COALESCE(e.gallons, CAST(json_extract(e.metadata, '$.gallons') AS REAL))
                    ELSE 0
                END
            ) AS total_gallons
        FROM expenses e
        LEFT JOIN trucks t ON t.truck_id = e.truck_id
        WHERE e.date BETWEEN ? AND ?
        GROUP BY week, week_start, e.truck_id, t.number
        ORDER BY week_start, e.truck_id
    """, conn_charts, params=(chart_start_date, chart_end_date))

    # Get weekly miles by truck
    df_miles_weekly = pd.read_sql_query("""
        SELECT 
            strftime('%Y-%W', date) as week,
            date(date, 'weekday 0', '-6 days') as week_start,
            truck_id,
            SUM(COALESCE(loaded_miles, 0) + COALESCE(empty_miles, 0)) as total_miles
        FROM income
        WHERE DATE(date) BETWEEN DATE(?) AND DATE(?)
        GROUP BY week, truck_id
        ORDER BY week, truck_id
    """, conn_charts, params=(chart_start_date, chart_end_date))

    conn_charts.close()

    # Chart 1: Income vs Expenses
    st.subheader("💰 Income vs Expenses (12 Weeks)")

    if not df_income_weekly.empty or not df_expense_weekly.empty:
        # Merge income and expense data
        df_income_expense = pd.merge(
            df_income_weekly[['week', 'week_start', 'total_income']], 
            df_expense_weekly[['week', 'total_expense']], 
            on='week', 
            how='outer'
        ).fillna(0)
    
        # Format week labels
        df_income_expense['week_label'] = pd.to_datetime(df_income_expense['week_start']).dt.strftime('%b %d')
    
        fig1 = go.Figure()
    
        fig1.add_trace(go.Scatter(
            x=df_income_expense['week_label'],
            y=df_income_expense['total_income'],
            mode='lines+markers',
            name='Income',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
    
        fig1.add_trace(go.Scatter(
            x=df_income_expense['week_label'],
            y=df_income_expense['total_expense'],
            mode='lines+markers',
            name='Expenses',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
    
        fig1.update_layout(
            title="Weekly Income vs Expenses",
            xaxis_title="Week Starting",
            yaxis_title="Amount ($)",
            hovermode='x unified',
            height=400
        )
    
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No income or expense data available for the selected period")

    # Chart 2: Profit per Week
    st.subheader("📈 Profit per Week (12 Weeks)")

    if not df_income_weekly.empty or not df_expense_weekly.empty:
        # Calculate profit
        df_profit = pd.merge(
            df_income_weekly[['week', 'week_start', 'total_income']], 
            df_expense_weekly[['week', 'total_expense']], 
            on='week', 
            how='outer'
        ).fillna(0)
    
        df_profit['profit'] = df_profit['total_income'] - df_profit['total_expense']
        df_profit['week_label'] = pd.to_datetime(df_profit['week_start']).dt.strftime('%b %d')
    
        fig2 = go.Figure()
    
        fig2.add_trace(go.Scatter(
            x=df_profit['week_label'],
            y=df_profit['profit'],
            mode='lines+markers',
            name='Profit',
            line=dict(color='blue', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(0, 100, 255, 0.1)'
        ))
    
        # Add zero line
        fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
        fig2.update_layout(
            title="Weekly Profit",
            xaxis_title="Week Starting",
            yaxis_title="Profit ($)",
            hovermode='x unified',
            height=400
        )
    
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No profit data available for the selected period")

    # Chart 3: Income by Dispatcher
    st.subheader("👥 Income by Dispatcher (12 Weeks)")

    if not df_dispatcher_weekly.empty:
        # Pivot data for multiple lines
        df_dispatcher_pivot = df_dispatcher_weekly.pivot(
            index='week_start', 
            columns='dispatcher', 
            values='total_income'
        ).fillna(0)
    
        # Format week labels
        df_dispatcher_pivot.index = pd.to_datetime(df_dispatcher_pivot.index).strftime('%b %d')
    
        fig3 = go.Figure()
    
        # Add a line for each dispatcher
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        for idx, dispatcher in enumerate(df_dispatcher_pivot.columns):
            fig3.add_trace(go.Scatter(
                x=df_dispatcher_pivot.index,
                y=df_dispatcher_pivot[dispatcher],
                mode='lines+markers',
                name=dispatcher,
                line=dict(width=3, color=colors[idx % len(colors)]),
                marker=dict(size=8)
            ))
    
        fig3.update_layout(
            title="Weekly Income by Dispatcher",
            xaxis_title="Week Starting",
            yaxis_title="Income ($)",
            hovermode='x unified',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No dispatcher data available for the selected period")

    # Chart 4: Rate Per Mile (RPM)
    st.subheader("💵 Rate Per Mile (12 Weeks)")

    if not df_rpm_weekly.empty:
        df_rpm_weekly['week_label'] = pd.to_datetime(df_rpm_weekly['week_start']).dt.strftime('%b %d')
    
        fig4 = go.Figure()
    
        fig4.add_trace(go.Scatter(
            x=df_rpm_weekly['week_label'],
            y=df_rpm_weekly['rpm'],
            mode='lines+markers',
            name='RPM',
            line=dict(color='purple', width=3),
            marker=dict(size=8)
        ))
    
        fig4.update_layout(
            title="Weekly Rate Per Mile (Total Income ÷ Total Miles)",
            xaxis_title="Week Starting",
            yaxis_title="Rate Per Mile ($)",
            hovermode='x unified',
            height=400
        )
    
        st.plotly_chart(fig4, use_container_width=True)
    
        # Show summary stats
        avg_rpm = df_rpm_weekly['rpm'].mean()
        st.metric("Average RPM (12 weeks)", f"${avg_rpm:.2f}")
    else:
        st.info("No RPM data available for the selected period")

    # Chart 5: Fuel Gallons vs Total Miles
    st.subheader("⛽ Fuel Consumption vs Miles (12 Weeks)")

    # Truck selector
    truck_filter_fuel = st.selectbox(
        "Select Truck (or All)",
        options=['All Trucks'] + sorted(df_fuel_weekly['truck_number'].dropna().unique().tolist()),
        key="fuel_truck_filter"
    )

    if not df_fuel_weekly.empty and not df_miles_weekly.empty:
        # Filter by truck if selected
        if truck_filter_fuel != 'All Trucks':
            selected_truck_id = df_fuel_weekly[df_fuel_weekly['truck_number'] == truck_filter_fuel]['truck_id'].iloc[0]
            df_fuel_filtered = df_fuel_weekly[df_fuel_weekly['truck_id'] == selected_truck_id]
            df_miles_filtered = df_miles_weekly[df_miles_weekly['truck_id'] == selected_truck_id]
        else:
            # Aggregate all trucks
            df_fuel_filtered = df_fuel_weekly.groupby(['week', 'week_start']).agg({
                'total_gallons': 'sum'
            }).reset_index()
            df_miles_filtered = df_miles_weekly.groupby(['week', 'week_start']).agg({
                'total_miles': 'sum'
            }).reset_index()
    
        # Merge fuel and miles
        df_fuel_miles = pd.merge(
            df_fuel_filtered[['week', 'week_start', 'total_gallons']], 
            df_miles_filtered[['week', 'total_miles']], 
            on='week', 
            how='outer'
        ).fillna(0)
    
        try:
            # Filter out invalid dates first
            df_fuel_miles = df_fuel_miles[df_fuel_miles['week_start'].notna()]
            df_fuel_miles = df_fuel_miles[df_fuel_miles['week_start'] != '0']
            df_fuel_miles = df_fuel_miles[df_fuel_miles['week_start'] != 0]
    
            # Convert to datetime with error handling
            df_fuel_miles['week_label'] = pd.to_datetime(df_fuel_miles['week_start'], errors='coerce').dt.strftime('%b %d')
    
            # Remove rows where date conversion failed
            df_fuel_miles = df_fuel_miles[df_fuel_miles['week_label'].notna()]
        except Exception as e:
            st.warning(f"Some date values could not be parsed: {e}")
            df_fuel_miles['week_label'] = df_fuel_miles['week_start'].astype(str)
    
        # Create dual-axis chart
        fig5 = go.Figure()
    
        fig5.add_trace(go.Scatter(
            x=df_fuel_miles['week_label'],
            y=df_fuel_miles['total_gallons'],
            mode='lines+markers',
            name='Fuel (Gallons)',
            line=dict(color='orange', width=3),
            marker=dict(size=8),
            yaxis='y'
        ))
    
        fig5.add_trace(go.Scatter(
            x=df_fuel_miles['week_label'],
            y=df_fuel_miles['total_miles'],
            mode='lines+markers',
            name='Total Miles',
            line=dict(color='teal', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
    
        fig5.update_layout(
            title=f"Weekly Fuel vs Miles - {truck_filter_fuel}",
            xaxis_title="Week Starting",
            yaxis=dict(title="Fuel (Gallons)", side='left', showgrid=False),
            yaxis2=dict(title="Miles", side='right', overlaying='y', showgrid=False),
            hovermode='x unified',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    
        st.plotly_chart(fig5, use_container_width=True)
    
        # Calculate MPG
        total_gallons = df_fuel_miles['total_gallons'].sum()
        total_miles = df_fuel_miles['total_miles'].sum()
        if total_gallons > 0:
            mpg = total_miles / total_gallons
            st.metric(f"Average MPG - {truck_filter_fuel} (12 weeks)", f"{mpg:.2f}")
    else:
        st.info("No fuel or miles data available for the selected period")

    st.markdown("---")
    st.markdown("### ⛽ Fuel Efficiency by Truck")

    # Date range for fuel report
    fuel_col1, fuel_col2 = st.columns(2)
    with fuel_col1:
        fuel_start = st.date_input("Fuel Report Start Date", value=date.today().replace(month=1, day=1), key="fuel_start")
    with fuel_col2:
        fuel_end = st.date_input("Fuel Report End Date", value=date.today(), key="fuel_end")

    if fuel_start > fuel_end:
        st.error("Start date must be before end date.")
    else:
        conn_fuel = get_db_connection()
        try:
            # Query fuel expenses with gallons and loaded miles from income
            fuel_query = """
            SELECT 
                e.truck_id,
                t.number as truck_number,
                SUM(e.amount) as total_fuel_cost,
                SUM(e.gallons) as total_gallons,
                COALESCE(SUM(i.loaded_miles), 0) as total_loaded_miles,
                COALESCE(SUM(i.empty_miles), 0) as total_empty_miles
            FROM expenses e
            LEFT JOIN trucks t ON e.truck_id = t.truck_id
            LEFT JOIN income i ON e.truck_id = i.truck_id 
                AND i.date BETWEEN ? AND ?
            WHERE e.category = 'Fuel'
                AND e.date BETWEEN ? AND ?
                AND e.truck_id IS NOT NULL
            GROUP BY e.truck_id, t.number
            ORDER BY t.number
            """
        
            df_fuel = pd.read_sql_query(fuel_query, conn_fuel, 
                                        params=(fuel_start, fuel_end, fuel_start, fuel_end))
        finally:
            conn_fuel.close()

        if df_fuel.empty:
            st.info("No fuel data found for the selected date range.")
        else:
            # Calculate metrics
            df_fuel['total_miles'] = df_fuel['total_loaded_miles'] + df_fuel['total_empty_miles']
            df_fuel['mpg'] = df_fuel.apply(
                lambda row: round(row['total_miles'] / row['total_gallons'], 2) 
                if row['total_gallons'] > 0 else 0, axis=1
            )
            df_fuel['cost_per_mile'] = df_fuel.apply(
                lambda row: round(row['total_fuel_cost'] / row['total_miles'], 2) 
                if row['total_miles'] > 0 else 0, axis=1
            )
            df_fuel['cost_per_gallon'] = df_fuel.apply(
                lambda row: round(row['total_fuel_cost'] / row['total_gallons'], 2) 
                if row['total_gallons'] > 0 else 0, axis=1
            )

            # Display summary metrics
            st.markdown("#### Fleet Summary")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
            with metric_col1:
                total_fuel_cost = df_fuel['total_fuel_cost'].sum()
                st.metric("Total Fuel Cost", f"${total_fuel_cost:,.2f}")
        
            with metric_col2:
                total_gallons = df_fuel['total_gallons'].sum()
                st.metric("Total Gallons", f"{total_gallons:,.2f}")
        
            with metric_col3:
                total_miles = df_fuel['total_miles'].sum()
                fleet_mpg = round(total_miles / total_gallons, 2) if total_gallons > 0 else 0
                st.metric("Fleet Average MPG", f"{fleet_mpg}")
        
            with metric_col4:
                fleet_cpm = round(total_fuel_cost / total_miles, 2) if total_miles > 0 else 0
                st.metric("Fleet Fuel Cost/Mile", f"${fleet_cpm}")

            # Display detailed table
            st.markdown("#### Per-Truck Fuel Efficiency")
        
            display_fuel = df_fuel.copy()
            display_fuel['truck_number'] = display_fuel['truck_number'].fillna('Unknown')
            display_fuel['total_fuel_cost'] = display_fuel['total_fuel_cost'].apply(lambda x: f"${x:,.2f}")
            display_fuel['total_gallons'] = display_fuel['total_gallons'].apply(lambda x: f"{x:,.2f}")
            display_fuel['total_loaded_miles'] = display_fuel['total_loaded_miles'].apply(lambda x: f"{x:,.0f}")
            display_fuel['total_empty_miles'] = display_fuel['total_empty_miles'].apply(lambda x: f"{x:,.0f}")
            display_fuel['total_miles'] = display_fuel['total_miles'].apply(lambda x: f"{x:,.0f}")
            display_fuel['cost_per_mile'] = display_fuel['cost_per_mile'].apply(lambda x: f"${x:.2f}")
            display_fuel['cost_per_gallon'] = display_fuel['cost_per_gallon'].apply(lambda x: f"${x:.2f}")
        
            display_fuel = display_fuel[[
                'truck_number', 'total_fuel_cost', 'total_gallons', 
                'total_loaded_miles', 'total_empty_miles', 'total_miles',
                'mpg', 'cost_per_mile', 'cost_per_gallon'
            ]]
        
            display_fuel.columns = [
                'Truck', 'Total Fuel Cost', 'Total Gallons',
                'Loaded Miles', 'Empty Miles', 'Total Miles',
                'MPG', 'Cost/Mile', 'Cost/Gallon'
            ]
        
            st.dataframe(display_fuel, use_container_width=True, hide_index=True)

            # Export fuel report
            st.markdown("#### Export Fuel Report")
            export_col1, export_col2 = st.columns(2)
        
            with export_col1:
                if st.button("📥 Export to CSV", key="export_fuel_csv"):
                    csv = df_fuel.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"fuel_efficiency_{fuel_start}_{fuel_end}.csv",
                        "text/csv"
                    )
        
            with export_col2:
                if st.button("📥 Export to Excel", key="export_fuel_excel"):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_fuel.to_excel(writer, index=False, sheet_name='Fuel Efficiency')
                    st.download_button(
                        "Download Excel",
                        output.getvalue(),
                        f"fuel_efficiency_{fuel_start}_{fuel_end}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            # Visualization
            st.markdown("#### Fuel Efficiency Visualization")
        
            viz_option = st.selectbox(
                "Select Chart",
                ["MPG by Truck", "Cost per Mile by Truck", "Fuel Cost Distribution", "Miles Breakdown"],
                key="fuel_viz_option"
            )
        
            if viz_option == "MPG by Truck":
                fig = px.bar(
                    df_fuel,
                    x='truck_number',
                    y='mpg',
                    title='Miles Per Gallon by Truck',
                    labels={'truck_number': 'Truck', 'mpg': 'MPG'},
                    color='mpg',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
        
            elif viz_option == "Cost per Mile by Truck":
                fig = px.bar(
                    df_fuel,
                    x='truck_number',
                    y='cost_per_mile',
                    title='Fuel Cost per Mile by Truck',
                    labels={'truck_number': 'Truck', 'cost_per_mile': 'Cost per Mile ($)'},
                    color='cost_per_mile',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)
        
            elif viz_option == "Fuel Cost Distribution":
                fig = px.pie(
                    df_fuel,
                    values='total_fuel_cost',
                    names='truck_number',
                    title='Fuel Cost Distribution by Truck'
                )
                st.plotly_chart(fig, use_container_width=True)
        
            elif viz_option == "Miles Breakdown":
                # Prepare data for stacked bar chart
                miles_data = []
                for _, row in df_fuel.iterrows():
                    miles_data.append({
                        'Truck': row['truck_number'],
                        'Miles': row['total_loaded_miles'],
                        'Type': 'Loaded'
                    })
                    miles_data.append({
                        'Truck': row['truck_number'],
                        'Miles': row['total_empty_miles'],
                        'Type': 'Empty'
                    })
            
                df_miles = pd.DataFrame(miles_data)
                fig = px.bar(
                    df_miles,
                    x='Truck',
                    y='Miles',
                    color='Type',
                    title='Loaded vs Empty Miles by Truck',
                    barmode='stack',
                    color_discrete_map={'Loaded': '#2ecc71', 'Empty': '#e74c3c'}
                )
                st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 💰 Total Cost Per Mile by Truck
    # =========================
    st.markdown("### 💰 Total Cost Per Mile by Truck")
    st.caption("Comprehensive cost analysis including all expense categories")

    cpm_col1, cpm_col2 = st.columns(2)
    with cpm_col1:
        cpm_start = st.date_input("CPM Report Start Date", value=date.today().replace(month=1, day=1), key="cpm_start")
    with cpm_col2:
        cpm_end   = st.date_input("CPM Report End Date", value=date.today(), key="cpm_end")

    if cpm_start > cpm_end:
        st.error("Start date must be before end date.")
    else:
        # 1) Miles aggregation (income table) — CHECK COLUMN NAMES IF DIFFERENT
        conn_miles = get_db_connection()
        try:
            df_miles = safe_read_sql(
                """
                SELECT
                    i.truck_id,
                    COALESCE(SUM(i.loaded_miles), 0) AS total_loaded_miles,
                    COALESCE(SUM(i.empty_miles), 0)  AS total_empty_miles
                FROM income i
                WHERE i.date >= ? AND i.date <= ?
                GROUP BY i.truck_id
                """,
                conn_miles,
                params=(cpm_start, cpm_end)
            )
        finally:
            conn_miles.close()

        if df_miles is None or df_miles.empty:
            df_miles = pd.DataFrame(columns=['truck_id', 'total_loaded_miles', 'total_empty_miles'])
        for c in ['total_loaded_miles', 'total_empty_miles']:
            if c in df_miles.columns:
                df_miles[c] = pd.to_numeric(df_miles[c], errors='coerce').fillna(0.0)
            else:
                df_miles[c] = 0.0
        df_miles['total_miles'] = df_miles['total_loaded_miles'] + df_miles['total_empty_miles']

        # 2) Expenses by category in range -> df_expenses_by_cat (expenses table)
        conn_exp = get_db_connection()
        try:
            df_expenses_by_cat = safe_read_sql(
                """
                SELECT
                    e.truck_id,
                    t.number AS truck_number,
                    COALESCE(e.category, 'Uncategorized') AS category,
                    SUM(e.amount) AS category_total
                FROM expenses e
                LEFT JOIN trucks t ON t.truck_id = e.truck_id
                WHERE e.date >= ? AND e.date <= ?
                GROUP BY e.truck_id, t.number, e.category
                """,
                conn_exp,
                params=(cpm_start, cpm_end)
            )
        finally:
            conn_exp.close()

        if df_expenses_by_cat is None or df_expenses_by_cat.empty:
            df_expenses_by_cat = pd.DataFrame(columns=['truck_id', 'truck_number', 'category', 'category_total'])

        # 3) Pivot expenses -> df_pivot
        df_pivot = df_expenses_by_cat.pivot_table(
            index=['truck_id', 'truck_number'],
            columns='category',
            values='category_total',
            fill_value=0,
            aggfunc='sum'
        ).reset_index()

        # 4) All trucks -> ensure trucks with no activity appear
        conn_cpm_all = get_db_connection()
        try:
            df_trucks_all = safe_read_sql(
                "SELECT truck_id, number AS truck_number FROM trucks",
                conn_cpm_all
            )
        finally:
            conn_cpm_all.close()
        if df_trucks_all is None or df_trucks_all.empty:
            df_trucks_all = pd.DataFrame(columns=["truck_id", "truck_number"])

        # 5) Merge expenses into df_base
        df_base = df_trucks_all.merge(
            df_pivot.drop(columns=[c for c in ['truck_number'] if c in df_pivot.columns]),
            on='truck_id', how='left'
        )
        if 'truck_number_x' in df_base.columns and 'truck_number' not in df_base.columns:
            df_base = df_base.rename(columns={'truck_number_x': 'truck_number'})
        if 'truck_number_y' in df_base.columns:
            df_base = df_base.drop(columns=['truck_number_y'])
        df_base['truck_number'] = df_base['truck_number'].fillna('Unknown')

        # 6) Merge miles into df_base
        df_base = df_base.merge(
            df_miles[['truck_id', 'total_miles', 'total_loaded_miles', 'total_empty_miles']],
            on='truck_id', how='left'
        )
        for c in ['total_miles', 'total_loaded_miles', 'total_empty_miles']:
            if c in df_base.columns:
                df_base[c] = pd.to_numeric(df_base[c], errors='coerce').fillna(0.0)
            else:
                df_base[c] = 0.0

        # 7) Loans (truck + linked trailer loans, prorated for range)
        # If you already build `loans_df` elsewhere, keep that. Otherwise compute here:
        def compute_loans_df(start_d, end_d):
            # Build per-truck prorated loan totals including linked trailers
            loans_map = get_truck_loans_in_range(start_d, end_d)  # returns keyed by truck number
            if not loans_map:
                return pd.DataFrame(columns=['truck_id', 'loan_amount'])
            rows = []
            # Need truck_id -> truck_number map:
            conn_l = get_db_connection()
            try:
                df_trk = safe_read_sql("SELECT truck_id, number AS truck_number FROM trucks", conn_l)
            finally:
                conn_l.close()
            id_by_number = {}
            if df_trk is not None and not df_trk.empty:
                id_by_number = {str(r['truck_number']): int(r['truck_id']) for _, r in df_trk.iterrows()}
            for tr_number, val in loans_map.items():
                tid = id_by_number.get(str(tr_number))
                if tid is None:
                    continue
                rows.append({'truck_id': tid, 'loan_amount': float(val.get('total_loan', 0.0))})
            return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['truck_id', 'loan_amount'])

        if 'loans_df' in globals() and isinstance(loans_df, pd.DataFrame) and not loans_df.empty:
            loans_for_cpm = loans_df[['truck_id', 'loan_amount']].copy()
        else:
            loans_for_cpm = compute_loans_df(cpm_start, cpm_end)

        if loans_for_cpm is None or loans_for_cpm.empty:
            loans_for_cpm = pd.DataFrame(columns=['truck_id', 'loan_amount'])

        df_base = df_base.merge(loans_for_cpm, on='truck_id', how='left')
        df_base['loan_amount'] = pd.to_numeric(df_base['loan_amount'], errors='coerce').fillna(0.0)

        # 8) Totals and CPM
        protected_cols = {
            'truck_id', 'truck_number', 'total_miles',
            'total_loaded_miles', 'total_empty_miles', 'loan_amount'
        }
        category_columns = [col for col in df_base.columns if col not in protected_cols]
        for cat in category_columns:
            df_base[cat] = pd.to_numeric(df_base[cat], errors='coerce').fillna(0.0)

        df_base['total_expenses'] = df_base[category_columns].sum(axis=1) + df_base['loan_amount']
        df_base['cost_per_mile_raw'] = df_base.apply(
            lambda row: (row['total_expenses'] / row['total_miles']) if row['total_miles'] > 0 else float('nan'),
            axis=1
        )

        # 9) Display
        st.markdown("#### Fleet Summary")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Total Expenses", f"${df_base['total_expenses'].sum():,.2f}")
        with metric_col2:
            st.metric("Total Miles", f"{df_base['total_miles'].sum():,.0f}")
        with metric_col3:
            fleet_miles = df_base['total_miles'].sum()
            fleet_cpm = (df_base['total_expenses'].sum() / fleet_miles) if fleet_miles > 0 else 0
            st.metric("Fleet Avg Cost/Mile", f"${fleet_cpm:.2f}")
        with metric_col4:
            avg_truck_cpm = df_base[df_base['total_miles'] > 0]['cost_per_mile_raw'].mean()
            st.metric("Avg Truck Cost/Mile", f"${(0 if pd.isna(avg_truck_cpm) else avg_truck_cpm):.2f}")

        st.markdown("#### Cost Breakdown by Truck")
        display_cpm = df_base.copy()

        # Format numbers for display
        display_cpm['total_expenses'] = display_cpm['total_expenses'].apply(lambda x: f"${x:,.2f}")
        display_cpm['total_miles']    = display_cpm['total_miles'].apply(lambda x: f"{x:,.0f}")
        display_cpm['cost_per_mile']  = display_cpm['cost_per_mile_raw'].apply(lambda v: "N/A" if pd.isna(v) else f"${v:.2f}")

        # Normalize truck identifier in case it ended up as index or different name
        if display_cpm.index.name == 'truck_number' and 'truck_number' not in display_cpm.columns:
            display_cpm = display_cpm.reset_index()

        # If it exists under an alternate name, normalize it
        for alt in ['truck_number', 'truck', 'Truck', 'unit_number', 'unit', 'tractor_number']:
            if alt in display_cpm.columns:
                display_cpm = display_cpm.rename(columns={alt: 'Truck'}) if alt != 'Truck' else display_cpm
                break

        # Use 'Truck' in the display columns (since we renamed it for presentation)
        display_columns = ['Truck', 'total_expenses', 'total_miles', 'cost_per_mile'] + category_columns

        # Build safely to avoid KeyErrors if a category column is missing
        existing_cols = [c for c in display_columns if c in display_cpm.columns]
        display_cpm_table = display_cpm.reindex(columns=existing_cols)

        st.dataframe(display_cpm_table, use_container_width=True, hide_index=True)

        st.markdown("#### Cost Per Mile Visualizations")
        viz_option = st.selectbox(
            "Select Visualization",
            ["Total Cost/Mile by Truck", "Expense Category Distribution", "Miles vs Expenses"],
            key="cpm_viz_option"
        )

        if viz_option == "Total Cost/Mile by Truck":
            df_chart = df_base[df_base['total_miles'] > 0].copy()
            # if truck_number landed in the index, bring it out
            if df_chart.index.name == 'truck_number' and 'truck_number' not in df_chart.columns:
                df_chart = df_chart.reset_index()
            fig = px.bar(
                df_chart,
                x='truck_number',
                y='cost_per_mile_raw',
                title='Total Cost Per Mile by Truck',
                labels={'truck_number': 'Truck', 'cost_per_mile_raw': 'Cost per Mile ($)'},
                color='cost_per_mile_raw',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_option == "Expense Category Distribution":
            # Build category breakdown safely even if some category columns are missing
            category_data = []
            categories_present = [c for c in category_columns if c in df_base.columns]

            # If no categories present, show friendly message and skip chart
            if not categories_present:
                st.info("No expense categories available to chart for this date range.")
            else:
                for _, row in df_base.iterrows():
                    truck_label = row.get('truck_number', None)
                    # Normalize to string for Plotly
                    truck_label = "Unknown" if truck_label is None or (isinstance(truck_label, float) and pd.isna(truck_label)) else str(truck_label)
                    for cat in categories_present:
                        amt = row.get(cat, 0.0)
                        # Coerce to number and skip zeros/NaNs to avoid empty series
                        try:
                            amt = float(amt)
                        except Exception:
                            amt = 0.0
                        if amt and not pd.isna(amt):
                            category_data.append({'Truck': truck_label, 'Category': str(cat), 'Amount': amt})

                df_cat_chart = pd.DataFrame(category_data)

                if df_cat_chart.empty or df_cat_chart['Amount'].sum() == 0:
                    st.info("No non-zero category expenses to display for this date range.")
                else:
                    # Ensure columns exist and are of correct types
                    df_cat_chart['Truck'] = df_cat_chart['Truck'].astype(str)
                    df_cat_chart['Category'] = df_cat_chart['Category'].astype(str)
                    df_cat_chart['Amount'] = pd.to_numeric(df_cat_chart['Amount'], errors='coerce').fillna(0.0)

                    fig = px.bar(
                        df_cat_chart,
                        x='Truck',
                        y='Amount',
                        color='Category',
                        title='Expense Distribution by Category and Truck',
                        labels={'Amount': 'Total Expenses ($)'},
                        barmode='stack'
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Exports
        st.markdown("#### Export Cost Per Mile Report")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            if st.button("📥 Export to CSV", key="export_cpm_csv"):
                csv = df_base.to_csv(index=False)
                st.download_button("Download CSV", csv, f"cost_per_mile_{cpm_start}_{cpm_end}.csv", "text/csv")
        with export_col2:
            if st.button("📥 Export to Excel", key="export_cpm_excel"):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_base.to_excel(writer, index=False, sheet_name='Cost Per Mile Summary')
                    df_expenses_by_cat.to_excel(writer, index=False, sheet_name='Expenses by Category')
                    df_miles.to_excel(writer, index=False, sheet_name='Miles Data')
                st.download_button("Download Excel", output.getvalue(),
                                   f"cost_per_mile_{cpm_start}_{cpm_end}.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("----\nFleet Management System © 2025")# Fleet Management System App (Streamlit)\n\n# [FULL CODE INCLUDED HERE – copy from above assistant message]\n
