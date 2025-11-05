"""
Database Helper Module for Fleet Management System
This module provides PostgreSQL connection and helper functions
"""

import streamlit as st
import psycopg2
from psycopg2 import sql, extras
import os
from contextlib import contextmanager


def get_db_connection():
    """
    Get PostgreSQL connection using Neon connection string from Streamlit secrets
    or environment variable DATABASE_URL
    """
    try:
        # Try to get from Streamlit secrets first (for Streamlit Cloud)
        if hasattr(st, 'secrets') and 'DATABASE_URL' in st.secrets:
            connection_string = st.secrets['DATABASE_URL']
        else:
            # Fallback to environment variable (for local testing)
            connection_string = os.getenv('DATABASE_URL')
            
        if not connection_string:
            st.error("❌ DATABASE_URL not found in secrets or environment variables")
            st.info("Please add DATABASE_URL to your Streamlit secrets (Streamlit Cloud) or .env file (local)")
            st.stop()
            
        # Create connection with autocommit disabled for transaction control
        conn = psycopg2.connect(connection_string)
        conn.autocommit = False
        return conn
    except Exception as e:
        st.error(f"❌ Failed to connect to PostgreSQL database: {e}")
        st.info("Please check your DATABASE_URL connection string in Streamlit secrets")
        st.stop()


@contextmanager
def get_db_cursor(commit=True):
    """
    Context manager for database operations with automatic commit/rollback
    
    Usage:
        with get_db_cursor() as cur:
            cur.execute("SELECT * FROM trucks")
            results = cur.fetchall()
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        yield cur
        if commit:
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


def execute_with_returning(cursor, query, params):
    """
    Execute INSERT query and return the inserted ID using RETURNING clause
    
    Args:
        cursor: Database cursor
        query: SQL INSERT query (should include RETURNING id_column)
        params: Query parameters
        
    Returns:
        The ID of the inserted row
    """
    cursor.execute(query, params)
    result = cursor.fetchone()
    return result[0] if result else None


def bulk_insert_trucks(df, column_mappings, create_missing_drivers=False, create_missing_dispatchers=False):
    """
    Bulk insert trucks from DataFrame with proper PostgreSQL transaction handling
    
    Args:
        df: Pandas DataFrame containing truck data
        column_mappings: Dictionary mapping data fields to DataFrame columns
        create_missing_drivers: Whether to auto-create missing drivers
        create_missing_dispatchers: Whether to auto-create missing dispatchers
        
    Returns:
        Dictionary with success_count, errors, created_drivers, created_dispatchers
    """
    import pandas as pd
    
    errors = []
    success_count = 0
    created_drivers = 0
    created_dispatchers = 0
    
    # Get single connection for entire bulk operation
    conn = get_db_connection()
    
    try:
        # Use a single cursor for the entire transaction
        cur = conn.cursor()
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Extract truck data from row
                number = row[column_mappings['number']] if column_mappings.get('number') else None
                make = row[column_mappings['make']] if column_mappings.get('make') and pd.notna(row[column_mappings['make']]) else None
                model = row[column_mappings['model']] if column_mappings.get('model') and pd.notna(row[column_mappings['model']]) else None
                year = int(row[column_mappings['year']]) if column_mappings.get('year') and pd.notna(row[column_mappings['year']]) else None
                plate = row[column_mappings['plate']] if column_mappings.get('plate') and pd.notna(row[column_mappings['plate']]) else None
                vin = row[column_mappings['vin']] if column_mappings.get('vin') and pd.notna(row[column_mappings['vin']]) else None
                status = row[column_mappings['status']] if column_mappings.get('status') and pd.notna(row[column_mappings['status']]) else "Active"
                loan_amount = float(row[column_mappings['loan']]) if column_mappings.get('loan') and pd.notna(row[column_mappings['loan']]) else 0.0
                
                # Handle driver assignment
                driver_id = None
                if column_mappings.get('driver') and pd.notna(row[column_mappings['driver']]):
                    ident = str(row[column_mappings['driver']]).strip()
                    
                    # Look up existing driver
                    cur.execute(
                        "SELECT driver_id FROM drivers WHERE name=%s OR license_number=%s",
                        (ident, ident)
                    )
                    found = cur.fetchone()
                    
                    if found:
                        driver_id = found[0]
                    elif create_missing_drivers and ident:
                        # Create driver and get ID using RETURNING
                        cur.execute(
                            """
                            INSERT INTO drivers (name, license_number, status, created_at)
                            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                            RETURNING driver_id
                            """,
                            (ident, None, "Active")
                        )
                        driver_id = cur.fetchone()[0]
                        created_drivers += 1
                
                # Handle dispatcher assignment
                dispatcher_id = None
                if column_mappings.get('dispatcher') and pd.notna(row[column_mappings['dispatcher']]):
                    val = row[column_mappings['dispatcher']]
                    
                    # Try numeric ID first
                    try:
                        dispatcher_id = int(val)
                    except (ValueError, TypeError):
                        # Treat as name
                        name = str(val).strip()
                        cur.execute(
                            "SELECT dispatcher_id FROM dispatchers WHERE name=%s",
                            (name,)
                        )
                        found = cur.fetchone()
                        
                        if found:
                            dispatcher_id = found[0]
                        elif create_missing_dispatchers and name:
                            # Create dispatcher and get ID using RETURNING
                            cur.execute(
                                """
                                INSERT INTO dispatchers (name, created_at)
                                VALUES (%s, CURRENT_TIMESTAMP)
                                RETURNING dispatcher_id
                                """,
                                (name,)
                            )
                            dispatcher_id = cur.fetchone()[0]
                            created_dispatchers += 1
                
                # Insert truck using RETURNING to get the new truck_id
                cur.execute(
                    """
                    INSERT INTO trucks (
                        number, make, model, year, plate, vin, status, 
                        loan_amount, driver_id, dispatcher_id, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    RETURNING truck_id
                    """,
                    (number, make, model, year, plate, vin, status, loan_amount, driver_id, dispatcher_id)
                )
                truck_id = cur.fetchone()[0]
                
                # Record loan history if applicable
                if loan_amount and loan_amount > 0:
                    from datetime import date
                    cur.execute(
                        """
                        INSERT INTO truck_loan_history (
                            truck_id, loan_amount, start_date, note, created_at
                        )
                        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                        """,
                        (truck_id, float(loan_amount), date.today(), "Bulk upload")
                    )
                
                # Record driver assignment if applicable
                if driver_id:
                    from datetime import date
                    cur.execute(
                        """
                        INSERT INTO driver_truck_assignments (
                            driver_id, truck_id, start_date, note, created_at
                        )
                        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                        """,
                        (driver_id, truck_id, date.today(), "Assigned via bulk upload")
                    )
                
                success_count += 1
                
            except Exception as row_e:
                # Log the error but continue with other rows
                errors.append(f"Row {idx+1}: {str(row_e)}")
        
        # Commit the entire transaction if we got here
        conn.commit()
        
    except Exception as e:
        # Rollback on any error
        conn.rollback()
        raise e
    finally:
        # Always close cursor and connection
        try:
            cur.close()
        except:
            pass
        try:
            conn.close()
        except:
            pass
    
    return {
        'success_count': success_count,
        'errors': errors,
        'created_drivers': created_drivers,
        'created_dispatchers': created_dispatchers
    }


def close_all_db_connections():
    """Kept for compatibility - PostgreSQL handles connection pooling"""
    pass


def close_all_db_connections_if_any():
    """Kept for compatibility - PostgreSQL handles connection pooling"""
    pass
