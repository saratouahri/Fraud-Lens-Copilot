# query_generator.py — version 100% LLM-driven intelligente
import os
import re
import gc
import json
import logging
import atexit
from pathlib import Path
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudlens-llm")

ROOT_DIR = Path(__file__).parent
MODEL_PATH = ROOT_DIR / "models" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"❌ Modèle non trouvé à {MODEL_PATH}\n"
        "Téléchargez-le depuis : https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    )

N_THREADS = max(1, (os.cpu_count() or 4) - 2)
_global_llm = None

def get_llm():
    """Renvoie le LLM global (chargé une seule fois)"""
    global _global_llm
    if _global_llm is None:
        _global_llm = load_llm()
    return _global_llm


def load_llm():
    logger.info(f"🧠 Chargement du modèle optimisé Mistral depuis {MODEL_PATH}")
    llm = LlamaCpp(
        model_path=str(MODEL_PATH),
        n_ctx=1024,                # réduit le contexte → moins de latence
        n_threads=os.cpu_count(),  # utilise tous les cœurs CPU
        n_batch=128,               # petit batch = réponse plus fluide
        n_gpu_layers=0,            # reste full CPU
        f16_kv=True,               # clé-valeurs en float16 (gain mémoire et temps)
        temperature=0.1,
        top_p=0.9,
        verbose=False,
        use_mlock=False,           # évite les copies inutiles
        use_mmap=True,             # mmap = lecture directe sans chargement complet
        streaming=True,            # génère token par token (réduit latence visible)
    )
    return llm


META_PROMPT = """
<|system|>
Tu es **Fraud-Lens Copilot**, un assistant intelligent spécialisé en analyse de fraude bancaire.
Tu comprends les questions en langage naturel et tu réponds UNIQUEMENT en **JSON strictement valide**.

---

### 🎯 TA MISSION
Analyser la question de l'utilisateur pour déterminer son intention :
1️⃣ **Intention "business"** → question théorique ou explicative  
   (ex : "Explique ce qu’est une transaction frauduleuse", "Comment détecter un compte compromis ?")  
   ➜ type = "business"  
   ➜ sql = ""  
   ➜ metric_name = ""  
   ➜ explanation = courte explication métier claire et informative

2️⃣ **Intention "analytical"** → question de calcul, agrégat ou statistique  
   (ex : "Combien de fraudes ?", "Quel est le taux de fraude ?")  
   ➜ type = "analytical"  
   ➜ générer une requête SQL avec COUNT, SUM, AVG, etc.

3️⃣ **Intention "sql"** → question demandant un échantillon de transactions  
   (ex : "Montre 5 transactions frauduleuses", "Liste les fraudes récentes")  
   ➜ type = "sql"  
   ➜ générer une requête SQL avec SELECT ... LIMIT ...

---

### 🧠 DONNÉES DISPONIBLES
Base DuckDB avec table `transactions`
Champs : step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud  
Valeurs possibles de `type`: 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT', 'PAYMENT'  
`isFraud = 1` indique une fraude confirmée.

---

### 🧮 RÈGLES DE LOGIQUE
- Si la question commence par *combien*, *nombre*, *taux*, *moyenne*, *pourcentage* → analytical  
- Si la question contient *montre*, *liste*, *donne-moi*, *affiche* → sql  
- Si la question contient *explique*, *qu’est-ce que*, *comment détecter*, *définis* → business  
- En cas de doute, prioriser **analytical** si la question mentionne des données concrètes (fraudes, montants, transferts, etc.)  

---

### 🔍 FORMAT DE SORTIE STRICT
Réponds UNIQUEMENT en JSON valide, sans aucun texte additionnel :
{{
  "type": "business" | "analytical" | "sql",
  "sql": "REQUÊTE SQL SI APPLICABLE (vide si type=business)",
  "metric_name": "Nom du métrique si applicable",
  "explanation": "Phrase claire et concise expliquant le sens de la réponse"
}}

---

### ✅ EXEMPLES

Question : "Combien de fraudes dans les transferts ?"
Réponse :
{{
  "type": "analytical",
  "sql": "SELECT COUNT(*) FROM transactions WHERE isFraud = 1 AND type = 'TRANSFER';",
  "metric_name": "Fraudes TRANSFER",
  "explanation": "Nombre total de fraudes détectées dans les transactions de type TRANSFER."
}}

Question : "Explique ce qu’est une transaction frauduleuse"
Réponse :
{{
  "type": "business",
  "sql": "",
  "metric_name": "",
  "explanation": "Une transaction frauduleuse est une opération non autorisée effectuée par un individu malveillant pour détourner de l’argent sans le consentement du titulaire du compte."
}}

Question : "Montre 5 transactions frauduleuses"
Réponse :
{{
  "type": "sql",
  "sql": "SELECT * FROM transactions WHERE isFraud = 1 ORDER BY step DESC LIMIT 5;",
  "metric_name": "Exemples de fraudes",
  "explanation": "Les 5 transactions les plus récentes identifiées comme frauduleuses."
}}

---

<|user|>
{query}
<|assistant|>
"""


# ---- process_query (remplace la version existante) ----
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import re
import logging

logger = logging.getLogger("fraudlens-llm")

def process_query(query: str, llm_instance):
    """Interprétation complète de la requête par le LLM"""
    chain = (
        PromptTemplate(template=META_PROMPT, input_variables=["query"])
        | llm_instance
        | StrOutputParser()
    )

    raw_output = chain.invoke({"query": query}).strip()
    logger.info(f"🧠 Sortie brute du modèle : {raw_output}")

    # 🔍 Recherche d’un JSON dans la sortie
    json_match = re.search(r"\{[\s\S]*\}", raw_output)
    json_text = json_match.group(0) if json_match else None
    result = None

    # 🧩 Étape 1 : Parsing JSON (si trouvé)
    if json_text:
        try:
            result = json.loads(json_text)
        except Exception as e:
            logger.warning(f"⚠️ JSON mal formé : {e}")

    # 🧩 Étape 2 : Si pas de JSON → encapsuler texte
    if not result and len(raw_output.split()) > 3:
        logger.info("✅ Encapsulation automatique de la sortie texte en JSON business")
        result = {
            "type": "business",
            "sql": "",
            "metric_name": "",
            "explanation": raw_output.strip()
        }

    # 🧩 Étape 3 : Fallback ultime
    if not result:
        logger.warning(f"⚠️ Sortie non JSON du modèle : {raw_output}")
        result = {
            "type": "business",
            "sql": "",
            "metric_name": "",
            "explanation": get_fallback_explanation(query),
        }

    # 🧩 Étape 4 : Correction automatique du SQL si besoin
    sql_text = result.get("sql", "")
    if sql_text:
        fixed_sql = sanitize_sql(sql_text)
        if fixed_sql != sql_text:
            logger.info(f"🛠 SQL corrigé automatiquement : {fixed_sql}")
        result["sql"] = fixed_sql

        # Vérifie les parenthèses
        if fixed_sql.count("(") != fixed_sql.count(")"):
            logger.warning(f"⚠️ SQL potentiellement invalide : {fixed_sql}")
            try:
                correction_prompt = f"""
Corrige cette requête SQL DuckDB pour qu’elle soit valide :
{sql_text}
Réponds UNIQUEMENT avec la requête corrigée (pas de texte, pas de JSON).
"""
                corrected = llm_instance.invoke(correction_prompt).strip()
                if corrected.startswith("SELECT"):
                    logger.info(f"✅ SQL réparé par le LLM : {corrected}")
                    result["sql"] = corrected
            except Exception as e:
                logger.error(f"💥 Erreur correction SQL par LLM : {e}")

    return result
def sanitize_sql(sql: str) -> str:
    """Nettoyage léger du SQL pour corriger les erreurs courantes du LLM"""
    sql = sql.strip()
    # Supprimer les doubles parenthèses ou caractères parasites
    sql = re.sub(r"\)\)", ")", sql)
    sql = re.sub(r"\(\(", "(", sql)
    sql = re.sub(r";+", ";", sql)
    # Corriger les alias mal écrits
    sql = re.sub(r"\s+as\s+", " AS ", sql, flags=re.IGNORECASE)
    # Corriger AVG(isFraud)) -> AVG(isFraud)
    sql = sql.replace("))", ")")
    return sql

def get_fallback_explanation(query: str) -> str:
    """Réponse de secours si le modèle échoue"""
    q = query.lower()
    if "taux" in q:
        return "Le taux de fraude = (nombre de fraudes / total des transactions) * 100."
    elif "fraude" in q:
        return (
            "Une transaction frauduleuse correspond à une opération non autorisée, souvent liée à des transferts anormaux."
        )
    return "Je suis un assistant d'analyse de fraude bancaire. Posez-moi une question comme 'Combien de fraudes ?'."


def _safe_close_llm(llm):
    try:
        if hasattr(llm, "_model") and hasattr(llm._model, "close"):
            llm._model.close()
        gc.collect()
    except Exception as e:
        logger.error(f"Erreur fermeture modèle : {e}")


atexit.register(lambda: _safe_close_llm(globals().get("_global_llm", None)))


if __name__ == "__main__":
    llm = load_llm()
    q = "Combien de fraudes dans les transferts ?"
    print(json.dumps(process_query(q, llm), indent=2, ensure_ascii=False))
    _safe_close_llm(llm)
