#!/usr/bin/env python3
# extract.py — Extraction structurée d'entités (Fiches profils) avec Qwen2.5-3B

# === PROMPTS (modifiables sans toucher à la logique) ===
SYSTEM_PROMPT = """Tu es un système expert d'extraction d'entités nommées.
Ton rôle est d'analyser le texte libre fourni et d'extraire UNIQUEMENT les champs suivants sous la forme d'un objet JSON strict :

{
  "prenom": "Prénom extrait (string)",
  "nom": "Nom de famille extrait (string)",
  "age": Âge sous forme de nombre entier (int) ou null si introuvable,
  "email": "Adresse email extraite (string)"
}

Consignes :
1. Ne devine pas les valeurs manquantes.
2. L'âge doit être un NOMBRE ENTIER (ex: 34).
3. Le prénom et le nom doivent être séparés correctement.
4. Réponds STRICTEMENT en JSON valide, sans texte additionnel, sans markdown."""

FEW_SHOT_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user", 
        "content": "Texte : Je suis Thomas Roche, 45 ans. Contact : thomas.roche@company.org"
    },
    {
        "role": "assistant", 
        "content": '{"prenom": "Thomas", "nom": "Roche", "age": 45, "email": "thomas.roche@company.org"}'
    }
]

# === LOGIQUE ===
import csv
import json
import os
from llama_cpp import Llama

# 1. Vérification du modèle dans le volume partagé
MODEL_PATH = "/models/qwen2.5-3b-instruct-q4_k_m.gguf"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Le fichier modèle est introuvable sous : {MODEL_PATH}")

print("🚀 Chargement du modèle Qwen2.5-3B-Instruct...")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)

# 2. Traitement du fichier CSV source
INPUT_CSV = "prospects.csv"
fiches_profils = []

print(f"📄 Lecture du fichier {INPUT_CSV}...\n")

with open(INPUT_CSV, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        prospect_id = row["id"]
        texte = row["texte_libre"]

        messages = FEW_SHOT_MESSAGES + [
            {"role": "user", "content": f"Texte : {texte}"}
        ]

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=128,
            temperature=0.0,  # Strictement déterministe
            response_format={"type": "json_object"}
        )

        raw_output = response["choices"][0]["message"]["content"]

        try:
            extraction = json.loads(raw_output)
        except json.JSONDecodeError:
            extraction = {"prenom": None, "nom": None, "age": None, "email": None}

        fiche = {
            "id": prospect_id,
            **extraction
        }
        
        fiches_profils.append(fiche)
        print(f"ID {prospect_id} Extrait → {fiche.get('prenom')} {fiche.get('nom')} ({fiche.get('age')} ans) | Email: {fiche.get('email')}")

# 3. Export JSON des fiches profils
OUTPUT_JSON = "fiches_profils.json"
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(fiches_profils, f, ensure_ascii=False, indent=2)

# 4. Export CSV structuré
OUTPUT_CSV = "profils_extraits.csv"
with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    fieldnames = ["id", "prenom", "nom", "age", "email"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(fiches_profils)

print(f"\n✅ Extraction terminée !")
print(f"  - Fiches JSON : {OUTPUT_JSON}")
print(f"  - Export CSV  : {OUTPUT_CSV}")