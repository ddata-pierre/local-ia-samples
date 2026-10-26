# 04-extraction - Extraction d'informations structurées (Entités nommées & Fiches profils)

> **Objectif** : Utiliser un LLM local pour lire des biographies en texte libre issues d'un fichier CSV, extraire de façon structurée les entités clés (**Prénom, Nom, Âge, Email**) et générer automatiquement des fiches profils sous forme de données JSON et CSV exploitables.  
> **Durée** : ~15 min.  
> **Prérequis** : Podman, l'image de base `local-ia/base-llm:latest` et le volume partagé `shared-models` (contenant le modèle `qwen2.5-3b-instruct-q4_k_m.gguf`).

---

## 1. Problématique & Architecture

Dans de nombreux cas d'usage métiers (CRM, Onboarding, RH, parsing de candidatures), les données clients ou prospects arrivent sous forme de texte non structuré (mails de présentation, notes de comptes-rendus, champs "présentation" dans des formulaires).

L'objectif de ce tutoriel est d'implémenter un pipeline d'**Extraction d'Entités Nommées (NER)** capable de :
1. Parcourir un fichier CSV contenant des textes libres.
2. Passer chaque texte au modèle **Qwen2.5-3B-Instruct** avec un prompt d'extraction guidé et un format JSON strict.
3. Générer un fichier de fiches profils structurées en JSON (`fiches_profils.json`) et exporter un fichier CSV nettoyé (`profils_extraits.csv`).

---

## 2. Jeux de données - `assets/prospects.csv`

Créez le fichier `assets/prospects.csv` avec 5 exemples de présentations en texte libre (incluant des formats variables d'âge, des adresses email noyées dans le texte, et des tournures idiomatiques) :

```csv
id,texte_libre
1,"Salut à tous ! Moi c'est Julien Bertrand, j'ai fêté mes 34 ans le mois dernier. Je suis consultant Cloud et vous pouvez me joindre directement sur julien.bertrand@email-pro.fr pour toute question."
2,"Bonjour, je m'appelle Claire Duchemin. J'ai 28 ans et je travaille dans le design produit. Mon adresse de contact est claire.d@design-studio.io."
3,"Candidature spontanée de Marc De La Tour (42 ans). Ancien chef de projet data, joignable via marc.delatour@data-corp.com."
4,"Ravi de rejoindre la communauté. Je suis Sophie N'Guyen, 19 ans, étudiante en alternance. N'hésitez pas à m'écrire à sophie.nguyen99@provider.com."
5,"Compte-rendu d'échange avec M. Antoine Vasseur. Âge : 51 ans. Poste : Directeur financier. Contact principal : a.vasseur@finance-conseil.fr."
```

---

## 3. Script d'extraction - `assets/extract.py`

Le script utilise `csv.DictReader` pour lire le fichier source, puis la méthode `create_chat_completion` avec un prompt **Few-Shot** pour garantir une extraction exacte sans aucune hallucination.

```python
#!/usr/bin/env python3
# extract.py - Extraction structurée d'entités (Fiches profils) avec Qwen2.5-3B

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

# 2. Prompt Système & Directives d'extraction
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

# 3. Exemples Few-Shot (Apprentissage par l'exemple)
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

# 4. Traitement du fichier CSV source
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

# 5. Export JSON des fiches profils
OUTPUT_JSON = "fiches_profils.json"
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(fiches_profils, f, ensure_ascii=False, indent=2)

# 6. Export CSV structuré
OUTPUT_CSV = "profils_extraits.csv"
with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    fieldnames = ["id", "prenom", "nom", "age", "email"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(fiches_profils)

print(f"\n✅ Extraction terminée !")
print(f"  - Fiches JSON : {OUTPUT_JSON}")
print(f"  - Export CSV  : {OUTPUT_CSV}")
```

---

## 4. Dockerfile optimisé - `assets/Dockerfile`

En nous appuyant sur l'image de base `local-ia/base-llm:latest` créée précédemment, le Dockerfile de ce tutoriel est extrêmement rapide à construire :

```dockerfile
FROM local-ia/base-llm:latest

LABEL maintainer="ddata.fr"
LABEL description="Tuto 04 - Extraction d'entités et création de fiches profils"

# Copie du script et du jeu de données
COPY --chown=appuser:appuser prospects.csv /home/appuser/prospects.csv
COPY --chown=appuser:appuser extract.py /home/appuser/extract.py

USER appuser
CMD ["python3", "extract.py"]
```

---

## 5. Build et Exécution avec Podman

### Étape 1 : Construction de l'image du tutoriel (instantané)
```bash
cd 04-extraction/assets
podman build -t local-ia/extraction:tuto04 .
```

### Étape 2 : Lancement en injectant le volume partagé de modèles
```bash
podman run --rm -v shared-models:/models local-ia/extraction:tuto04
```

---

## 6. Résultat attendu

**Sortie Console** :
```text
🚀 Chargement du modèle Qwen2.5-3B-Instruct...
📄 Lecture du fichier prospects.csv...

ID 1 Extrait → Julien Bertrand (34 ans) | Email: julien.bertrand@email-pro.fr
ID 2 Extrait → Claire Duchemin (28 ans) | Email: claire.d@design-studio.io
ID 3 Extrait → Marc De La Tour (42 ans) | Email: marc.delatour@data-corp.com
ID 4 Extrait → Sophie N'Guyen (19 ans) | Email: sophie.nguyen99@provider.com
ID 5 Extrait → Antoine Vasseur (51 ans) | Email: a.vasseur@finance-conseil.fr

✅ Extraction terminée !
  - Fiches JSON : fiches_profils.json
  - Export CSV  : profils_extraits.csv
```

**Contenu de `fiches_profils.json` généré** :
```json
[
  {
    "id": "1",
    "prenom": "Julien",
    "nom": "Bertrand",
    "age": 34,
    "email": "julien.bertrand@email-pro.fr"
  },
  {
    "id": "2",
    "prenom": "Claire",
    "nom": "Duchemin",
    "age": 28,
    "email": "claire.d@design-studio.io"
  },
  {
    "id": "3",
    "prenom": "Marc",
    "nom": "De La Tour",
    "age": 42,
    "email": "marc.delatour@data-corp.com"
  },
  {
    "id": "4",
    "prenom": "Sophie",
    "nom": "N'Guyen",
    "age": 19,
    "email": "sophie.nguyen99@provider.com"
  },
  {
    "id": "5",
    "prenom": "Antoine",
    "nom": "Vasseur",
    "age": 51,
    "email": "a.vasseur@finance-conseil.fr"
  }
]
```

---

## 7. Récupération des fichiers générés

Pour récupérer les fichiers `fiches_profils.json` et `profils_extraits.csv` sur votre machine hôte :

```bash
# Lancement avec un nom temporaire
podman run --name temp-extract -v shared-models:/home/appuser/models:z local-ia/extraction:tuto04

# Copie des données extraites vers l'hôte
podman cp temp-extract:/home/appuser/fiches_profils.json /tmp/fiches_profils.json
podman cp temp-extract:/home/appuser/profils_extraits.csv /tmp/profils_extraits.csv

# Nettoyage du conteneur temporaire
podman rm temp-extract
```

---

## 8. Explications & Bonnes Pratiques

1. **Typage strict dans le prompt** : En précisant qu'`age` doit être un `int` (nombre entier), le modèle convertit de lui-même `"34 ans"` en `34`.
2. **Support des noms composés** : Le prompt Few-Shot apprend au modèle à ne pas casser les noms complexes comme `"De La Tour"` ou `"N'Guyen"`.
3. **Architecture réutilisable** : L'utilisation du volume `shared-models` et de l'image de base `local-ia/base-llm:latest` rend ce tutoriel très économe en espace disque et en temps de build sur votre CPU/N100.

---

## 9. Exercices + Corrections

### Exercice 1 - Ajouter un champ téléphone

**Consigne** : Étendre l'extraction pour capturer aussi le numéro de téléphone (format français : `06 12 34 56 78` ou `+33 6 12 34 56 78`).

**Correction** : Dans `SYSTEM_PROMPT`, ajouter `"telephone": "Numéro de téléphone (string) ou null"` dans le JSON et dans les consignes. Dans `FEW_SHOT_MESSAGES`, ajouter un exemple avec téléphone. Relancer l'extraction.

### Exercice 2 - Comparer modèles (vitesse vs qualité)

**Consigne** : Télécharger `SmolLM2-135M` dans le volume partagé, modifier `MODEL_PATH` dans `extract.py`, relancer et comparer temps d'exécution + qualité d'extraction (noms composés, âges).

**Correction** :
```bash
# Télécharger SmolLM2
podman run --rm -v shared-models:/models local-ia/base-llm:latest \
  python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ggml-org/SmolLM2-135M-GGUF', filename='SmolLM2-135M-Q4_K_M.gguf', local_dir='/models', local_dir_use_symlinks=False)"

# Modifier MODEL_PATH dans extract.py → '/models/SmolLM2-135M-Q4_K_M.gguf'
# Rebuild + run
```

**Observation attendue** : SmolLM2 ~3x plus rapide mais rate les noms composés (`De La Tour` → `De`, `La`, `Tour` séparés) et âges parfois.

---

## 10. Nettoyage

```bash
podman rmi local-ia/extraction:tuto04
rm -f assets/fiches_profils.json assets/profils_extraits.csv
```