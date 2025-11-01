import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import time
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

# -------------------------
# CONFIGURATION
# -------------------------
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/TON_WEBHOOK_ICI"  # 🔧 Mets ton webhook ici
HEADERS = {"User-Agent": "Mozilla/5.0"}
SEEN_FILE = "seen.json"
QUERY = "ps5"            # 🔧 Mot-clé de recherche
LOCATION = "paris"       # 🔧 Ville ou région
REFRESH_MINUTES = 10     # 🔧 Intervalle entre chaque recherche

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
        title_elem = ad.select_one("[data-qa-id='aditem_title']")
        if not title_elem:
            continue
        title = title_elem.text.strip()
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
# BOUCLE PRINCIPALE
# -------------------------
def main():
    vues = charger_annonces_vues()
    while True:
        print(f"🔎 Scraping Leboncoin pour '{QUERY}' à '{LOCATION}'...")
        df = get_ads(QUERY, LOCATION)
        if df.empty:
            print("⚠️ Aucune annonce trouvée.")
        else:
            df = filtrer_annonces(df)
            if df.empty:
                print("ℹ️ Aucune annonce de bonne qualité.")
            else:
                model, vectorizer = entrainer_modele(df.dropna(subset=["prix"]))
                bons_plans = detecter_bons_plans(df, model, vectorizer)
                nouveaux = [row for _, row in bons_plans.iterrows() if row["lien"] not in vues]

                if not nouveaux:
                    print("ℹ️ Aucun nouveau bon plan détecté pour le moment.")
                else:
                    print(f"✅ {len(nouveaux)} nouveaux bons plans trouvés !")
                    for annonce in nouveaux:
                        print(f"💰 {annonce['titre']} - {annonce['prix']} €")
                        envoyer_discord(annonce)
                        vues.add(annonce["lien"])
                    sauvegarder_annonces_vues(vues)

        print(f"⏳ Pause de {REFRESH_MINUTES} minutes...\n")
        time.sleep(REFRESH_MINUTES * 60)

if __name__ == "__main__":
    main()
