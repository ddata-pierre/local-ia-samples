#!/usr/bin/env python3
# rag_app.py - Application RAG locale avec preuve d'origine

import os
import glob
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama

# 1. Chargement du modèle LLM local
MODEL_PATH = "/models/qwen2.5-3b-instruct-q4_k_m.gguf"
print("🔄 Chargement du modèle LLM...")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)

# 2. Chargement et indexation de la base de connaissances
DOCS_DIR = "/home/appuser/docs"
documents = []
filenames = []

print("📚 Indexation de la base de connaissances...")
for filepath in glob.glob(os.path.join(DOCS_DIR, "*.md")):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        documents.append(content)
        filenames.append(os.path.basename(filepath))

# Vectorisation TF-IDF légère pour la recherche par similarité
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)


def chercher_contexte(question, top_k=1):
    """Recherche le document le plus pertinent pour la question posée."""
    query_vec = vectorizer.transform([question])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    
    return filenames[best_idx], documents[best_idx], best_score


def poser_question(question):
    """Exécute le pipeline RAG complet pour une question."""
    print(f"\n==================================================")
    print(f"❓ Question : {question}")
    
    # Étape 1 : Retrieval (Recherche)
    filename, contexte, score = chercher_contexte(question)
    print(f"🔍 Document trouvé : {filename} (score de pertinence : {score:.2f})")
    
    # Étape 2 : Construction d'un prompt plus direct et tolérant
    system_prompt = f"""Tu es un assistant technique utile. Réponds à la question de l'utilisateur en te basant sur le document fourni ci-dessous ({filename}).

CONTEXTE :
---
{contexte}
---

Consigne : Donne la réponse clairement en te basant uniquement sur ce texte."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # Étape 3 : Génération LLM
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=150,
        temperature=0.0
    )
    
    reponse_texte = response["choices"][0]["message"]["content"]
    print(f"💡 Réponse RAG :\n{reponse_texte}")


# 5. Tests de démonstration avec preuves d'origine
if __name__ == "__main__":
    # Test 1 : Preuve via la clé de chiffrement du serveur ZEUS
    poser_question("Quelle est la clé de chiffrement du serveur ZEUS et quel est son port SSH ?")
    
    # Test 2 : Preuve via le délai de préavis spécifique de 2026
    poser_question("Quel est le délai de préavis pour poser des congés payés ?")
    
    # Test 3 : Preuve via le mot de passe Wi-Fi
    poser_question("Quel est le mot de passe du réseau Wi-Fi visiteurs ?")