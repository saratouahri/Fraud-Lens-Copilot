# api/main.py — version intelligente alignée avec query_generator.py
import os
import sys
import logging
import traceback
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import duckdb
import pandas as pd
from contextlib import asynccontextmanager

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from query_generator import load_llm, process_query, get_fallback_explanation, _safe_close_llm
from sql_sanitize import sanitize_sql

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudlens-api")

class QueryRequest(BaseModel):
    query: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Initialisation du backend Fraud-Lens")
    app.state.db = None
    app.state.llm = None

    db_path = ROOT_DIR / "paysim.duckdb"
    if db_path.exists():
        app.state.db = duckdb.connect(str(db_path), read_only=True)
        logger.info("✅ Base PaySim connectée")

    try:
        app.state.llm = load_llm()
        logger.info("✅ Modèle LLM chargé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle : {e}")
        traceback.print_exc()

    yield

    if app.state.db:
        app.state.db.close()
    if app.state.llm:
        _safe_close_llm(app.state.llm)
    logger.info("👋 Backend arrêté proprement")

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "db_connected": app.state.db is not None,
        "model_loaded": app.state.llm is not None,
    }
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=1)
@app.post("/analyze")
async def analyze(request: QueryRequest):
    try:
        llm = app.state.llm
        db = app.state.db

        result = process_query(request.query, llm)
        # log utile pour debug
        logger.info("process_query result: %s", result)

        rtype = (result.get("type") or "business").lower()
        explanation = result.get("explanation") or get_fallback_explanation(request.query)

        # === CAS METIER : NE JAMAIS EXECUTER DE SQL ===
        if rtype == "business":
            return {
                "type": "business",
                "explanation": explanation,
                "original_query": request.query,
            }

        # Récupérer et valider la SQL proposée
        sql = (result.get("sql") or "").strip()
        if not sql:
            # Pas de SQL — considère comme réponse métier
            return {
                "type": "business",
                "explanation": explanation or "La requête nécessite une explication métier (pas de SQL fourni).",
                "original_query": request.query,
            }

        # Sécurité : n'exécute que SELECT / WITH
        if not sql.lower().startswith(("select", "with")):
            logger.warning("SQL non autorisée proposée par LLM : %s", sql)
            return {
                "type": "business",
                "explanation": "Le modèle a proposé une requête non exécutable. Reformulez la requête.",
                "original_query": request.query,
            }

        # Nettoyage & exécution
        safe_sql = sanitize_sql(sql)
        df = db.execute(safe_sql).fetchdf()

        if df.empty:
            return {
                "type": "sql" if rtype == "sql" else rtype,
                "sql": safe_sql,
                "results": [],
                "value": 0,
                "metric_name": result.get("metric_name", "Résultats"),
                "explanation": "Aucun résultat trouvé.",
                "original_query": request.query,
            }

        # analytical: retourner la métrique
        if rtype == "analytical":
            # suppose que le SQL renvoie une seule valeur agrégée en première colonne
            value = float(df.iloc[0, 0])
            return {
                "type": "analytical",
                "sql": safe_sql,
                "value": value,
                "metric_name": result.get("metric_name", "Transactions"),
                "explanation": explanation,
                "original_query": request.query,
            }

        # else SQL tabulaire
        return {
            "type": "sql",
            "sql": safe_sql,
            "results": df.head(500).to_dict(orient="records"),
            "explanation": explanation,
            "original_query": request.query,
        }

    except Exception as e:
        logger.exception("Erreur analyse")
        return {
            "type": "business",
            "explanation": f"Erreur inattendue : {str(e)}",
            "original_query": request.query,
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
