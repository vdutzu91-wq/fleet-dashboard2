# Fleet Management Dashboard

A Streamlit application to manage trucks, trailers, drivers, dispatch, income, expenses, loans, and reports. Built for day-to-day fleet ops with CSV/Excel export, bulk upload, and robust reporting.

## Features
- Trucks, Trailers, Drivers management
- Dispatchers and assignments
- Income and Expenses with bulk upload
- Loans tracking and per-truck P&L
- Reports (per-truck, category breakdown, profit/loss)
- Backup/Restore tools (download .db, restore from file)
- Admin Settings (reset individual tables safely)
- Optional OpenRouteService integration for mileage

## Project Structure (typical)

.
├── app.py
├── requirements.txt
├── README.md
└── .streamlit/
└── secrets.toml.example # template for secrets (do not commit real secrets)


## Local Setup

1) Create and activate a virtual environment (optional but recommended)

- Windows (PowerShell)
python -m venv .venv
..venv\Scripts\activate

- macOS/Linux
python3 -m venv .venv
source .venv/bin/activate


2) Install dependencies
pip install -r requirements.txt


3) Run the app
streamlit run app.py


The app will create a SQLite database file `fleet_management.db` (or as configured) in your working directory by default.

## Configuration (Secrets)

Create `.streamlit/secrets.toml` locally if you want to override defaults. Example:

```toml
# .streamlit/secrets.toml
DB_DIR = "."
ORS_API_KEY = ""        # if using OpenRouteService
ADMIN_USER = "admin"    # optional if you added internal login
ADMIN_PASS = "admin123"
On Streamlit Cloud, set these in App → Settings → Secrets.

Deployment (Streamlit Community Cloud)
Push this repository to GitHub (private or public).
On Streamlit Cloud, click “New app” and select this repo and app.py.
In App → Settings → Secrets, add:
toml
Copy
DB_DIR = "/mount/data"      # persistent storage on Streamlit Cloud
ORS_API_KEY = ""            # fill if using mileage API
Deploy. The app will create and persist the database at /mount/data/fleet_management.db.
Backups
Use the built-in Backup/Restore page to download the .db file.
Recommended policy: keep last 7 daily and 4 weekly backups.
You can also export CSV/Excel from the app for critical reports.
Updating the App
Use branches for changes: create a feature branch, test locally, then merge to main.
Streamlit Cloud will deploy the latest main.
Database migrations are idempotent in the app (ensure_* and init_* functions run on start).
Troubleshooting
If you see SQLite “database is locked”: retry; WAL mode is enabled automatically on startup.
If you run into a “foreign key mismatch” when resetting a table, the app uses safe delete helpers that temporarily disable FK checks for that single operation.
If dependencies are missing on deployment, adjust requirements.txt and redeploy.
License
Private/internal project. All rights reserved.


If you’d like, I can also add a short “Contributing” section and a tiny screenshot section later. After you edit and commit this README on GitHub, tell me and we’ll proceed to the app.py edits for DB_DIR + WAL and then deploy to Streamlit Cloud.
