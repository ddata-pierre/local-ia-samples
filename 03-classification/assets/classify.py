#!/usr/bin/env python3
# classify.py - Classification précise avec Few-Shot Prompting

# === PROMPTS (modifiables sans toucher à la logique) ===
SYSTEM_PROMPT = """Tu es un expert en triage de tickets de support d'entreprise.
Analyse le ticket et renvoie STRICTEMENT un objet JSON avec les valeurs exactes :

- "departement": "IT" (matériel/logiciel/accès), "RH" (paie/congés/contrat travail), "Finance" (frais/factures/remboursements), "Commercial" (clients/ventes), "Juridique" (contrats fournisseurs/médias)
- "frustration": 
  * "Faible" (question simple, ton neutre)
  * "Moyenne" (gêne mineure, relance courtoise)
  * "Elevee" (blocage de travail, agacement explicite)
  * "Critique" (colère, menaces, plusieurs relances sans réponse)
- "urgence":
  * "Basse" (demande d'information, pas d'échéance)
  * "Normale" (procédure classique)
  * "Haute" (échéance proche, impact direct sur une tâche)
  * "Critique" (bloquant immédiat pour une échéance à très court terme < 24h)

Réponds UNIQUEMENT avec un JSON valide, sans balises markdown, sans texte supplémentaire."""

FEW_SHOT_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    
    # Exemple 1 : Faible / Basse
    {"role": "user", "content": "Ticket : Où puis-je trouver le calendrier des congés payés de l'année prochaine ?"},
    {"role": "assistant", "content": '{"departement": "RH", "frustration": "Faible", "urgence": "Basse"}'},
    
    # Exemple 2 : Élevée / Haute
    {"role": "user", "content": "Ticket : Mon accès au CRM client est bloqué alors que j'ai un appel de négociation dans 1h. Merci de débloquer vite !"},
    {"role": "assistant", "content": '{"departement": "IT", "frustration": "Elevee", "urgence": "Haute"}'}
]

# === LOGIQUE ===
import json
from llama_cpp import Llama

# 1. Charge le modèle Qwen2.5-3B
MODEL_PATH = "/models/qwen2.5-3b-instruct-q4_k_m.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)

# 2. Charge les tickets
with open("tickets.json", "r", encoding="utf-8") as f:
    tickets = json.load(f)

# 3. Résultats attendus pour évaluation rapide
ATTENDUS = {
    "INC-001": ("IT", "Critique", "Haute"),
    "INC-002": ("RH", "Critique", "Normale"),
    "INC-003": ("Finance", "Critique", "Normale"),
    "INC-004": ("Commercial", "Critique", "Critique"),
    "INC-005": ("IT", "Critique", "Basse"),
    "INC-006": ("Finance", "Faible", "Normale"),
}

resultats = []
for ticket in tickets:
    # Construction du fil de conversation avec Few-Shot
    messages = FEW_SHOT_MESSAGES + [
        {"role": "user", "content": f"Ticket : {ticket['texte']}"}
    ]
    
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=128,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    texte = response["choices"][0]["message"]["content"]
    
    try:
        classification = json.loads(texte)
    except json.JSONDecodeError:
        classification = {"departement": "IT", "frustration": "Moyenne", "urgence": "Normale"}
    
    resultats.append({
        "id": ticket["id"],
        "texte": ticket["texte"][:80] + "...",
        **classification
    })
    print(f"{ticket['id']} → {classification}")

# 4. Sauvegarde
with open("resultats.json", "w", encoding="utf-8") as f:
    json.dump(resultats, f, ensure_ascii=False, indent=2)

# 5. Mini-évaluation (pédagogique)
correct = sum(
    1 for r in resultats 
    if (r["departement"], r["frustration"], r["urgence"]) == ATTENDUS[r["id"]]
)
print(f"\n📊 Précision : {correct}/{len(resultats)} ({correct/len(resultats)*100:.0f}%)")
print("\n✅ Résultats sauvés dans resultats.json")