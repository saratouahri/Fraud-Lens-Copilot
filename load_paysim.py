import duckdb
import pandas as pd

# Charger le dataset (ajustez le chemin si nécessaire)
df = pd.read_csv("C:/Users/sara/Desktop/projets/fraudCopilot/data/PS_20174392719_1491204439457_log.csv") 

# Nettoyage initial
df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])  # Colonnes sensibles ou redondantes
df["step"] = df["step"].astype(int)  # Convertir step en entier

# Créer la base DuckDB
con = duckdb.connect("paysim.duckdb")
con.execute("""
CREATE TABLE transactions (
    step INTEGER,
    type VARCHAR,
    amount FLOAT,
    oldbalanceOrg FLOAT,
    newbalanceOrig FLOAT,
    oldbalanceDest FLOAT,
    newbalanceDest FLOAT,
    isFraud INTEGER
)
""")

# Insérer les données
con.register("df_view", df)
con.execute("INSERT INTO transactions SELECT * FROM df_view")
con.execute("CREATE INDEX idx_fraud ON transactions(isFraud)")
con.execute("CREATE INDEX idx_type ON transactions(type)")
con.close()

print("✅ Dataset PaySim chargé dans paysim.duckdb")