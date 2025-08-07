"""
Database utilities and schema functions for the Agentic AI Assistant.
Contains functions for database operations, schema management, and data formatting.
"""

import sqlite3
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
import os
from pathlib import Path
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections and operations.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
    
    def connect(self) -> sqlite3.Connection:
        """Establish database connection."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            return self.connection
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def __enter__(self):
        """Context manager entry."""
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def get_schema_description(db_path: str) -> str:
    """
    Get a formatted description of the database schema.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        Formatted string describing the database schema
    """
    try:
        with DatabaseManager(db_path) as conn:
            cursor = conn.cursor()
            
            schema_str = ""
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table_name, in tables:
                cursor.execute(f"PRAGMA table_info({table_name});")
                cols = cursor.fetchall()
                col_names = [col[1] for col in cols]
                schema_str += f"\n- {table_name}: columns = {', '.join(col_names)}"
            
            return schema_str.strip()
    except Exception as e:
        logger.error(f"Error getting schema description: {e}")
        return f"Error accessing database schema: {e}"


def get_table_schema(db_path: str, table_name: str) -> Dict[str, Any]:
    """
    Get detailed schema information for a specific table.
    
    Args:
        db_path: Path to the SQLite database
        table_name: Name of the table
        
    Returns:
        Dictionary containing table schema information
    """
    try:
        with DatabaseManager(db_path) as conn:
            cursor = conn.cursor()
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            schema = {
                "table_name": table_name,
                "columns": []
            }
            
            for col in columns:
                column_info = {
                    "name": col[1],
                    "type": col[2],
                    "not_null": bool(col[3]),
                    "default_value": col[4],
                    "primary_key": bool(col[5])
                }
                schema["columns"].append(column_info)
            
            return schema
    except Exception as e:
        logger.error(f"Error getting table schema for {table_name}: {e}")
        return {"table_name": table_name, "error": str(e)}


def get_all_tables(db_path: str) -> List[str]:
    """
    Get list of all tables in the database.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        List of table names
    """
    try:
        with DatabaseManager(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            return tables
    except Exception as e:
        logger.error(f"Error getting tables: {e}")
        return []


def execute_query(db_path: str, query: str) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.
    
    Args:
        db_path: Path to the SQLite database
        query: SQL query to execute
        
    Returns:
        DataFrame containing query results
    """
    try:
        with DatabaseManager(db_path) as conn:
            df = pd.read_sql_query(query, conn)
            return df
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return pd.DataFrame({"Error": [str(e)]})


def get_sample_data(db_path: str, table_name: str, limit: int = 5) -> pd.DataFrame:
    """
    Get sample data from a table.
    
    Args:
        db_path: Path to the SQLite database
        table_name: Name of the table
        limit: Number of rows to return
        
    Returns:
        DataFrame with sample data
    """
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    return execute_query(db_path, query)


def format_money_columns(df: pd.DataFrame, money_keywords: List[str]) -> pd.DataFrame:
    """
    Format money-related columns for display.
    
    Args:
        df: DataFrame to format
        money_keywords: List of keywords that indicate money columns
        
    Returns:
        Formatted DataFrame
    """
    formatted_df = df.copy()
    
    for col in formatted_df.select_dtypes(include='number').columns:
        col_lower = col.lower()
        
        # Format ratio columns as percentages
        if "ratio" in col_lower:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else ""
            )
        
        # Format money columns with commas
        elif any(keyword in col_lower for keyword in money_keywords):
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x:,.0f}" if pd.notnull(x) else ""
            )
    
    return formatted_df


def validate_sql_query(query: str) -> Tuple[bool, str]:
    """
    Basic SQL query validation.
    
    Args:
        query: SQL query to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    query_upper = query.upper()
    
    # Check for dangerous keywords
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return False, f"Query contains dangerous keyword: {keyword}"
    
    # Check for basic SELECT structure
    if "SELECT" not in query_upper:
        return False, "Query must contain SELECT statement"
    
    return True, "Query appears valid"


def get_database_info(db_path: str) -> Dict[str, Any]:
    """
    Get comprehensive database information.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        Dictionary containing database information
    """
    try:
        with DatabaseManager(db_path) as conn:
            cursor = conn.cursor()
            
            # Get database file info
            db_info = {
                "file_path": db_path,
                "file_size": os.path.getsize(db_path) if os.path.exists(db_path) else 0,
                "last_modified": datetime.fromtimestamp(
                    os.path.getmtime(db_path)
                ).isoformat() if os.path.exists(db_path) else None,
                "tables": []
            }
            
            # Get table information
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table_name, in tables:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # Get column count
                cursor.execute(f"PRAGMA table_info({table_name});")
                column_count = len(cursor.fetchall())
                
                table_info = {
                    "name": table_name,
                    "row_count": row_count,
                    "column_count": column_count
                }
                db_info["tables"].append(table_info)
            
            return db_info
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {"error": str(e)}


def backup_database(db_path: str, backup_path: str) -> bool:
    """
    Create a backup of the database.
    
    Args:
        db_path: Path to the original database
        backup_path: Path for the backup file
        
    Returns:
        True if backup successful, False otherwise
    """
    try:
        with DatabaseManager(db_path) as source_conn:
            with sqlite3.connect(backup_path) as backup_conn:
                source_conn.backup(backup_conn)
        logger.info(f"Database backed up to {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False


def get_table_statistics(db_path: str, table_name: str) -> Dict[str, Any]:
    """
    Get statistical information about a table.
    
    Args:
        db_path: Path to the SQLite database
        table_name: Name of the table
        
    Returns:
        Dictionary containing table statistics
    """
    try:
        with DatabaseManager(db_path) as conn:
            cursor = conn.cursor()
            
            # Get basic table info
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            stats = {
                "table_name": table_name,
                "row_count": row_count,
                "column_count": len(columns),
                "columns": []
            }
            
            # Get column statistics
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                
                # Get unique values count for categorical columns
                if col_type.lower() in ['text', 'varchar']:
                    cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name}")
                    unique_count = cursor.fetchone()[0]
                else:
                    unique_count = None
                
                column_stats = {
                    "name": col_name,
                    "type": col_type,
                    "unique_values": unique_count
                }
                stats["columns"].append(column_stats)
            
            return stats
    except Exception as e:
        logger.error(f"Error getting table statistics for {table_name}: {e}")
        return {"table_name": table_name, "error": str(e)}


# Database documentation constants
DATABASE_DOCUMENTATION = """
PnC_Data Table:
- Reserve Class contains insurance business lines such as 'Property', 'Casualty', 'Marine', 'Motor', etc.
- Exposure Year refers to the year in which the insured risk was exposed to potential loss.
- RI Type identifies whether the record is 'Gross' or one of the reinsurance types such as 'Ceded - XOL', 'Ceded - QS', 'Ceded - CAP', 'Ceded - FAC', or 'Ceded - Others'.
- Branch indicates the geographical business unit handling the contract, e.g., 'Europe', 'LATAM', 'North America'.
- Loss Type captures the nature of the loss, and may be one of: 'ATT', 'CAT', 'LARGE', 'THREAT', or 'Disc'.
- Underwriting Year represents the year in which the policy was underwritten or originated.
- Incurred Loss represents the total loss incurred to date, including paid and case reserves.
- Paid Loss is the portion of the Incurred Loss that has already been settled and paid out.
- IBNR is calculated as the difference between Ultimate Loss and Incurred Loss.
- Ultimate Loss is the projected final value of loss.
- Ultimate Premium refers to the projected premium expected to be earned.
- Loss Ratio is calculated as Ultimate Loss divided by Ultimate Premium.
- AvE Incurred = Expected - Actual Incurred.
- AvE Paid = Expected - Actual Paid.
- Budget Premium is the forecasted premium for budgeting.
- Budget Loss is the projected loss for budgeting.
- Earned Premium is the portion of the premium that has been earned.
- Case Reserves = Incurred Loss - Paid Loss.
"""


def get_database_path() -> str:
    """
    Get the default database path.
    
    Returns:
        Path to the default database
    """
    return './Actuarial_Data (2).db'


def is_database_accessible(db_path: str) -> bool:
    """
    Check if the database is accessible and exists.
    
    Args:
        db_path: Path to the database
        
    Returns:
        True if database is accessible, False otherwise
    """
    try:
        if not os.path.exists(db_path):
            return False
        
        with DatabaseManager(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            return True
    except Exception as e:
        logger.error(f"Database accessibility check failed: {e}")
        return False


def get_database_summary(db_path: str) -> Dict[str, Any]:
    """
    Get a summary of database contents for LLM context.
    
    Args:
        db_path: Path to the database
        
    Returns:
        Dictionary with database summary information
    """
    try:
        with DatabaseManager(db_path) as conn:
            cursor = conn.cursor()
            
            # Get table information
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            summary = {
                "total_tables": len(tables),
                "tables": []
            }
            
            for table_name, in tables:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # Get column count
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                column_count = len(columns)
                
                table_summary = {
                    "name": table_name,
                    "rows": row_count,
                    "columns": column_count,
                    "column_names": [col[1] for col in columns]
                }
                summary["tables"].append(table_summary)
            
            return summary
    except Exception as e:
        logger.error(f"Error getting database summary: {e}")
        return {"error": str(e)} 