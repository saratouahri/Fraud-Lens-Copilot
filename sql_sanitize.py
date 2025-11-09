# sql_sanitize.py
import re
import sqlvalidator

def sanitize_sql(sql: str) -> str:
    # Nettoyage basique SANS ajout de LIMIT
    sql = re.sub(r'--.*', '', sql)
    sql = sql.strip().rstrip(";") + ";"
    
    # Validation syntaxique seulement
    try:
        parsed = sqlvalidator.parse(sql)
        if not parsed.is_valid():
            raise ValueError("SQL syntax error")
        return parsed.sql
    except Exception as e:
        raise ValueError(f"Invalid SQL: {str(e)}")