# app.py — version alignée avec LLM intelligent (Fraud-Lens Copilot)
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go

# Configuration Streamlit
st.set_page_config(
    page_title="Fraud-Lens Copilot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HEADER ---
st.title("🔍 Fraud-Lens Copilot")
st.caption("Assistant conversationnel pour l'analyse de fraude bancaire | Dataset PaySim")

# --- Initialisation historique ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour ! Je suis **Fraud-Lens Copilot**, votre expert en détection de fraude bancaire. 🤖\n\n"
                "Je peux :\n"
                "📊 **Analyser les transactions** (ex. 'Combien de fraudes dans les transferts ?')\n"
                "💡 **Expliquer des concepts métier** (ex. 'Comment détecter un compte compromis ?')\n"
                "🧠 **Générer des requêtes SQL** sur les données PaySim\n\n"
                "**Exemples de questions** :\n"
                "- Montre 5 transactions frauduleuses\n"
                "- Explique ce qu’est une transaction frauduleuse\n"
                "- Combien de fraudes dans les transferts ?\n"
                "- Comment détecter un compte compromis ?"
            )
        }
    ]

# --- Sidebar ---
with st.sidebar:
    st.header("📚 Connaissances métier")
    st.markdown("""
    **Fraudes typiques (PaySim)** :
    - 🚨 **Transferts massifs** : >200k vers nouveaux comptes (`oldbalanceDest = 0`)
    - 💸 **Retraits anormaux** : CASH_OUT vidant le compte (`newbalanceOrig < 0`)
    - 🔁 **Schémas répétitifs** : mêmes montants ou heures récurrentes

    **Indicateurs clés** :
    - `isFraud = 1` → Transaction frauduleuse
    - `type` → Type de transaction (TRANSFER, CASH_OUT, etc.)
    - `step` → Pas de temps (1 step = 1 h)
    """)
    st.divider()
    st.subheader("🔧 Statut du backend")

    status_placeholder = st.empty()
    try:
        res = requests.get("http://localhost:8000/health", timeout=2)
        if res.status_code == 200:
            data = res.json()
            status_placeholder.success(
                f"✅ Backend opérationnel\nModèle chargé : {'Oui' if data['model_loaded'] else 'Non'}"
            )
        else:
            status_placeholder.warning("⚠️ Backend partiellement actif")
    except:
        status_placeholder.error("❌ Backend injoignable")

# --- Affichage historique ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("type") == "analytical" and "value" in msg:
            st.metric(label=msg["metric_name"], value=f"{msg['value']:,.0f}")

        elif "results" in msg and msg["results"]:
            st.dataframe(pd.DataFrame(msg["results"]))

        if "chart" in msg and msg["chart"]:
            fig = go.Figure(msg["chart"])
            st.plotly_chart(fig, use_container_width=True)

# --- Entrée utilisateur ---
if prompt := st.chat_input("Posez une question (ex : 'Combien de fraudes dans le dataset ?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyse en cours..."):
            try:
                r = requests.post(
                    "http://localhost:8000/analyze",
                    json={"query": prompt},
                    headers={"Content-Type": "application/json"},
                    timeout=240
                )
                r.raise_for_status()
                data = r.json()

                if data.get("type") == "business":
                    st.markdown("### 💼 Explication métier")
                    with st.container(border=True):
                        st.markdown(f"🧠 **{data['explanation']}**")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"💡 {data['explanation']}",
                        "type": "business"
                    })

                elif data.get("type") == "analytical":
                    st.markdown("### 📊 Résultat analytique")
                    st.metric(label=data["metric_name"], value=f"{data['value']:,.0f}")
                    st.info(data["explanation"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"📊 {data['metric_name']} : {data['value']:,.0f}\n\n{data['explanation']}",
                        "type": "analytical",
                        "value": data["value"],
                        "metric_name": data["metric_name"]
                    })

                else:
                    st.code(data["sql"], language="sql")
                    df = pd.DataFrame(data["results"])
                    st.markdown(f"### 📋 {len(df)} lignes extraites")

                    if len(df) > 500:
                        st.warning("⚠️ Affichage limité aux 500 premières lignes.")
                        df = df.head(500)
                    st.dataframe(df)

                    # Graphiques auto
                    fig = None
                    if "amount" in df.columns and "type" in df.columns:
                        fig = px.box(
                            df, x="type", y="amount",
                            color="isFraud" if "isFraud" in df.columns else None,
                            title="Distribution des montants par type"
                        )
                    elif "step" in df.columns and "isFraud" in df.columns:
                        fraud_timeline = df.groupby("step")["isFraud"].sum().reset_index()
                        fig = px.line(
                            fraud_timeline,
                            x="step", y="isFraud",
                            title="Évolution des fraudes dans le temps"
                        )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state.current_chart = fig.to_dict()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"```sql\n{data['sql']}\n```",
                        "type": "sql",
                        "sql": data["sql"],
                        "results": df.to_dict(orient="records"),
                        "chart": getattr(st.session_state, "current_chart", None)
                    })

            except Exception as e:
                st.error(f"❌ Erreur : {e}")
                st.session_state.messages.append({"role": "assistant", "content": str(e)})

# --- Reset bouton ---
if st.sidebar.button("🔄 Réinitialiser la conversation"):
    st.session_state.messages = []
    st.rerun()

st.markdown("---")
st.caption("🚀 Projet Fraud-Lens Copilot | Backend Mistral | Dataset : PaySim")
