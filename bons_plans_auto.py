import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import time
import json
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

# -------------------------
# CONFIGURATION
# -------------------------
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/TON_WEBHOOK_ICI"  # 🔧 Mets ton webhook ici
HEADERS = {"User-Agent": "Mozilla/5.0"}
SEEN_FILE = "seen.json"

# -------------------------
# OUTILS DE BASE
# -------------------------
def extract_number(text):
    match = re.search(r"\d+", text.replace(" ", ""))
    return int(match.group()) if match else None

def get_description(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        s = BeautifulSoup(r.text, "html.parser")
        desc_elem = s.select_one("[data-qa-id='adview_description_container']")
        return desc_elem.text.strip() if desc_elem else ""
    except:
        return ""

def get_ads(query, location, limit=30):
    search_url = f"https://www.leboncoin.fr/recherche?text={query}&locations={location}"
    response = requests.get(search_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    ads = []
    for ad in soup.select("a[data-qa-id='aditem_container']")[:limit]:
        title = ad.select_one("[data-qa-id='aditem_title']").text.strip()
        price_elem = ad.select_one("[data-qa-id='aditem_price']")
        price = extract_number(price_elem.text) if price_elem else None
        link = "https://www.leboncoin.fr" + ad["href"]
        desc = get_description(link)
        time.sleep(0.3)
        ads.append({
            "titre": title,
            "prix": price,
            "description": desc,
            "lien": link
        })
    return pd.DataFrame(ads)

# -------------------------
# FILTRE INTELLIGENT
# -------------------------
MOTS_POSITIFS = [
    "neuf", "comme neuf", "urgent", "sous garantie", "boîte scellée",
    "avec facture", "très bon état", "excellent état"
]

MOTS_NEGATIFS = [
    "cassé", "hs", "hors service", "à réparer", "écran fissuré", "boîte vide",
    "ne fonctionne pas", "problème", "non fonctionnel", "pour pièces", "incomplet"
]

def filtrer_annonces(df):
    def score_qualité(row):
        texte = (row["titre"] + " " + row["description"]).lower()
        score = 0
        for mot in MOTS_POSITIFS:
            if mot in texte:
                score += 2
        for mot in MOTS_NEGATIFS:
            if mot in texte:
                score -= 3
        return score

    df["qualité_score"] = df.apply(score_qualité, axis=1)
    return df[df["qualité_score"] > 0]

# -------------------------
# MACHINE LEARNING
# -------------------------
def entrainer_modele(df):
    df["texte"] = df["titre"].fillna('') + " " + df["description"].fillna('')
    df = df.dropna(subset=["prix"])
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="french")
    X = vectorizer.fit_transform(df["texte"])
    y = df["prix"]
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model, vectorizer

def detecter_bons_plans(df, model, vectorizer):
    df["texte"] = df["titre"].fillna('') + " " + df["description"].fillna('')
    X_new = vectorizer.transform(df["texte"])
    df["prix_estimé"] = model.predict(X_new)
    df["delta"] = df["prix_estimé"] - df["prix"]
    bons_plans = df[df["prix"] < 0.7 * df["prix_estimé"]].sort_values("delta", ascending=False)
    return bons_plans

# -------------------------
# GESTION DES DOUBLONS
# -------------------------
def charger_annonces_vues():
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, "r") as f:
            return set(json.load(f))
    return set()

def sauvegarder_annonces_vues(liens):
    with open(SEEN_FILE, "w") as f:
        json.dump(list(liens), f)

# -------------------------
# DISCORD ALERTES
# -------------------------
def envoyer_discord(annonce):
    data = {
        "content": (
            f"💸 **BON PLAN DÉTECTÉ !**\n"
            f"**{annonce['titre']}**\n"
            f"Prix réel : {annonce['prix']} €\n"
            f"Valeur estimée : {annonce['prix_estimé']:.0f} €\n"
            f"🔗 {annonce['lien']}"
        )
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=data)
    except Exception as e:
        print(f"Erreur d’envoi Discord : {e}")

# -------------------------
# INTERFACE STREAMLIT
# -------------------------
st.set_page_config(page_title="Leboncoin Bons Plans Auto", page_icon="💸")
st.title("💸 Détecteur automatique de bons plans Leboncoin")
st.caption("Analyse, prédit, et envoie les bons plans sur Discord — automatiquement !")

query = st.text_input("🔎 Mot-clé de recherche :", "ps5")
location = st.text_input("📍 Ville ou région :", "paris")
refresh_rate = st.number_input("⏱️ Intervalle de rafraîchissement (minutes) :", min_value=1, max_value=120, value=10)

if "last_run" not in st.session_state:
    st.session_state.last_run = 0

if st.button("🚀 Lancer la recherche automatique"):
    st.session_state.running = True

if st.session_state.get("running", False):
    current_time = time.time()
    if current_time - st.session_state.last_run > refresh_rate * 60:
        st.session_state.last_run = current_time
        with st.spinner(f"Analyse en cours pour '{query}' à '{location}'..."):
            df = get_ads(query, location)
            if df.empty:
                st.warning("Aucune annonce trouvée.")
            else:
                df = filtrer_annonces(df)
                if df.empty:
                    st.info("Aucune annonce de bonne qualité trouvée.")
                else:
                    model, vectorizer = entrainer_modele(df.dropna(subset=["prix"]))
                    bons_plans = detecter_bons_plans(df, model, vectorizer)
                    vues = charger_annonces_vues()
                    nouveaux = [row for _, row in bons_plans.iterrows() if row["lien"] not in vues]

                    if not nouveaux:
                        st.info("Aucun nouveau bon plan détecté pour le moment.")
                    else:
                        st.success(f"{len(nouveaux)} nouveaux bons plans trouvés ! 🎯")
                        for annonce in nouveaux:
                            st.markdown(
                                f"**[{annonce['titre']}]({annonce['lien']})**  \n"
                                f"💰 Prix : {annonce['prix']} € — Valeur estimée : {annonce['prix_estimé']:.0f} €  \n"
                                f"📊 Score qualité : {annonce['qualité_score']}"
                            )
                            envoyer_discord(annonce)
                            vues.add(annonce["lien"])
                        sauvegarder_annonces_vues(vues)

        st.rerun()

st.info("⏳ L’application se rafraîchira automatiquement toutes les "
        f"{refresh_rate} minutes tant qu’elle reste ouverte.")
