# 💸 Détecteur automatique de bons plans Leboncoin

Ce bot analyse en continu les annonces Leboncoin pour repérer les **bons plans** en fonction du **prix**, du **titre** et de la **description**.  
Il utilise un petit modèle **ML (Ridge Regression + TF-IDF)** pour estimer la vraie valeur d’un objet et envoie automatiquement les **bons plans sur Discord** via un webhook.

---

## 🚀 Fonctionnalités
- 🔎 Recherche dynamique par mot-clé et ville
- 🧠 Détection automatique des sous-évaluations
- ⚙️ Filtre intelligent (ignore “cassé”, “HS”, “à réparer”, etc.)
- 🔔 Envoi des alertes sur Discord
- ⏱️ Rafraîchissement automatique toutes les X minutes
- 💾 Mémoire locale (évite les doublons)

---

## 🧰 Installation locale

```bash
git clone https://github.com/<ton-nom-utilisateur>/leboncoin-bot.git
cd leboncoin-bot
pip install -r requirements.txt
streamlit run bons_plans_auto.py
