# 03-classification - Classification de tickets d'incidents (Modèle de décision)

> **Objectif** : utiliser un LLM local pour classifier automatiquement des tickets d'incidents (département, frustration, urgence).
> **Durée** : ~10 min.
> **Prérequis** : Podman, l'image de base `local-ia/base-llm:latest` et le volume partagé `shared-models` (contenant le modèle `qwen2.5-3b-instruct-q4_k_m.gguf`).

---

## 1. Problématique

On reçoit des tickets utilisateurs en texte libre. Il faut les router automatiquement :
- **Département** : IT, RH, Finance, Commercial, Juridique
- **Frustration utilisateur** : Faible, Moyenne, Élevée, Critique
- **Urgence** : Basse, Normale, Haute, Critique

---

## 2. Données d'exemple - 5 tickets

**Fichier** : `assets/tickets.json`

```json
[
  {
    "id": "INC-001",
    "texte": "Bonjour, mon ordinateur portable ne s'allume plus depuis ce matin. J'ai une présentation client à 14h et je ne peux pas accéder à mes fichiers. C'est vraiment urgent !"
  },
  {
    "id": "INC-002",
    "texte": "Je n'arrive pas à me connecter à l'espace RH pour télécharger ma fiche de paie. Le mot de passe ne fonctionne pas même après réinitialisation. C'est agaçant."
  },
  {
    "id": "INC-003",
    "texte": "Demande de remboursement de frais de déplacement du mois dernier. J'ai soumis le dossier il y a 3 semaines et toujours pas de nouvelles. Je commence à m'inquiéter."
  },
  {
    "id": "INC-004",
    "texte": "Notre contrat avec le fournisseur ACME expire vendredi. Il faut absolument le renouveler avant sinon on perd les conditions tarifaires. Qui s'en occupe ?"
  },
  {
    "id": "INC-005",
    "texte": "L'imprimante du 3ème étage ne fonctionne plus. C'est la 3ème fois ce mois-ci. Personne ne vient la réparer malgré mes relances. Je suis énervé."
  }
]
```

---

## 3. Approche : Prompt structuré (JSON mode)

On force le modèle à répondre **uniquement en JSON valide** avec les 3 champs.

**Prompt système** :
```
Tu es un classificateur de tickets d'incidents. Réponds UNIQUEMENT par un JSON valide avec exactement ces clés :
- "departement": "IT" | "RH" | "Finance" | "Commercial" | "Juridique"
- "frustration": "Faible" | "Moyenne" | "Elevee" | "Critique"
- "urgence": "Basse" | "Normale" | "Haute" | "Critique"

Aucun texte additionnel, pas de markdown, juste le JSON.
```

---

## 4. Dockerfile - Image de classification

**Fichier** : `assets/Dockerfile`

```dockerfile
FROM local-ia/base-llm:latest

LABEL description="Tuto 03 - Classification"

# Plus de téléchargement de modèle ici ! Le volume s'en charge.
COPY --chown=appuser:appuser tickets.json /home/appuser/tickets.json
COPY --chown=appuser:appuser classify.py /home/appuser/classify.py

USER appuser
CMD ["python3", "classify.py"]
```

---

## 5. Script de classification - `classify.py`

```python
#!/usr/bin/env python3
# classify.py - Classification de tickets avec llama.cpp

import json
from llama_cpp import Llama

# 1. Charge le modèle
MODEL_PATH = "/models/qwen2.5-3b-instruct-q4_k_m.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=1024, n_threads=4, verbose=False)

# 2. Prompt système (force JSON)
SYSTEM_PROMPT = """Tu es un classificateur de tickets d'incidents. Réponds UNIQUEMENT par un JSON valide avec exactement ces clés :
- "departement": "IT" | "RH" | "Finance" | "Commercial" | "Juridique"
- "frustration": "Faible" | "Moyenne" | "Elevee" | "Critique"
- "urgence": "Basse" | "Normale" | "Haute" | "Critique"

Aucun texte additionnel, pas de markdown, juste le JSON."""

# 3. Charge les tickets
with open("tickets.json", "r", encoding="utf-8") as f:
    tickets = json.load(f)

# 4. Classifie chaque ticket
resultats = []
for ticket in tickets:
    prompt = f"{SYSTEM_PROMPT}\n\nTicket : {ticket['texte']}\n\nClassification :"
    
    # Génération avec temperature=0 (déterministe)
    reponse = llm(prompt, max_tokens=128, temperature=0, stop=["}"], echo=False)
    texte = reponse["choices"][0]["text"]
    
    # Le stop sur "}" coupe la fin, on referme le JSON
    if not texte.strip().endswith("}"):
        texte = texte.strip() + "}"
    
    try:
        classification = json.loads(texte)
    except json.JSONDecodeError:
        # Fallback si JSON invalide
        classification = {"departement": "IT", "frustration": "Moyenne", "urgence": "Normale"}
    
    resultats.append({
        "id": ticket["id"],
        "texte": ticket["texte"][:80] + "...",
        **classification
    })
    print(f"{ticket['id']} → {classification}")

# 5. Sauvegarde résultats
with open("resultats.json", "w", encoding="utf-8") as f:
    json.dump(resultats, f, ensure_ascii=False, indent=2)

print("\n✅ Résultats sauvés dans resultats.json")
```

---

## 6. Build et lancement

```bash
cd 03-classification/assets
podman build -t local-ia/classification:tuto03 .
podman run --rm -v shared-models:/models local-ia/classification:tuto03
```

**Résultat attendu** (`resultats.json`) :
```json
[
  {
    "id": "INC-001",
    "texte": "Bonjour, mon ordinateur portable ne s'allume plus depuis ce matin...",
    "departement": "IT",
    "frustration": "Elevee",
    "urgence": "Critique"
  },
  {
    "id": "INC-002",
    "texte": "Je n'arrive pas à me connecter à l'espace RH pour télécharger ma fiche de paie...",
    "departement": "RH",
    "frustration": "Moyenne",
    "urgence": "Normale"
  },
  ...
]
```

---

## 7. Explications techniques

| `temperature=0` | Réponses déterministes (même input = même output) |
| `stop=["}"]` | Coupe à la fin du JSON (économise tokens) |
| `n_ctx=1024` | Contexte suffisant pour prompt + ticket |
| `response_format={"type": "json_object"}` | Force JSON valide via API llama.cpp |
| Prompt structuré + Few-Shot | Force le format, évite hallucinations |

---

## 8. Exercices + Corrections

### Exercice 1 - Ajouter un champ "priorité"

**Consigne** : Ajoute un champ `priorite` (P1/P2/P3/P4) basé sur urgence + frustration.

**Correction** : Modifie `SYSTEM_PROMPT` pour inclure `"priorite": "P1" | "P2" | "P3" | "P4"` et la logique de calcul.

### Exercice 2 - Router vers email

**Consigne** : Au lieu d'afficher, envoie un email au bon département (simulation `print`).

**Correction** :
```python
EMAILS = {"IT": "it@entreprise.fr", "RH": "rh@entreprise.fr", ...}
print(f"→ Envoi à {EMAILS[classification['departement']]} : {ticket['id']}")
```

### Exercice 3 - Comparer modèles

**Consigne** : Teste avec `Phi-3-mini-4k-instruct-q4.gguf` (plus capable) vs `Qwen2.5-3B-Instruct` (modèle par défaut).

**Correction** : Télécharge Phi-3 dans le volume partagé, change `MODEL_PATH` dans `classify.py`, rebuild, compare `resultats.json`.

---

## 9. Nettoyage

```bash
podman rmi local-ia/classification:tuto03
rm -f assets/resultats.json
```

---

## 10. Aller plus loin

- Fine-tuning sur tickets historiques (LoRA)
- Ajout confiance score (logprobs)
- Intégration ticketing (Jira, ServiceNow, GLPI)
- Prochaine étape : **04-extraction** - extraction d'entités nommées.