# 01-ollama - Ollama en conteneur (Podman)

> **Objectif** : faire tourner Ollama + modèles en conteneur, comprendre l'API REST.
> **Durée** : ~10 min.
> **Prérequis** : Podman installé, 16 Go RAM, modèles < 1 Go.

---

## 1. Pourquoi Podman + Ollama ?

| Avantages | Inconvénients |
|-----------|---------------|
| Isolation complète, pas de pollution hôte | Légère surcouche (Dockerfile) |
| Persistance via volumes nommés | Port mapping requis |
| Reproductible (image versionnée) | Nécessite Podman/Docker |

> Alternative locale : tutoriel 02 (llama.cpp brut).

---

## 2. Dockerfile - Image Ollama avec modèles préchargés

**Fichier** : `Dockerfile` (à la racine de `01-ollama/`)

```dockerfile
# Image de base officielle Ollama (basée sur Debian slim)
FROM docker.io/ollama/ollama:latest

# Auteur / métadonnées
LABEL maintainer="ddata.fr"
LABEL description="Ollama avec modèles tiny pour N100"

# Variable d'environnement : modèles à pull au build (cache layers)
ENV OLLAMA_MODELS="smollm2:135m qwen2.5:0.5b tinyllama:1.1b"

# Script d'initialisation : pull les modèles pendant le build
# Utilise un script shell explicite pour éviter les problèmes de continuation de ligne
RUN set -eux; \
    for m in $OLLAMA_MODELS; do \
        ollama serve & \
        sleep 3 && \
        ollama pull "$m" && \
        pkill ollama; \
    done

# Expose le port API REST
EXPOSE 11434

# Ajout ping pour démonstration hors réseau
RUN apt-get update \
    && apt-get install -y --no-install-recommends iputils-ping \
    && rm -rf /var/lib/apt/lists/*
    
# Point d'entrée : lance le serveur Ollama (ENTRYPOINT du base = /bin/ollama)
CMD ["serve"]
```

> **Build** (une seule fois) :
> ```bash
> podman build -t local-ia/ollama:tiny .
> ```
> Temps ~2–3 min (téléchargement modèles ~1 Go). Les layers sont cachés : rebuild instantané si Dockerfile inchangé.

---

## 3. Lancer le conteneur

```bash
# Crée un volume nommé pour persister les modèles (+ configs)
podman volume create ollama-data

# Lance le conteneur en détaché (-d), mappe port 11434, monte le volume
podman run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  local-ia/ollama:tiny
```

**Explications** :
| Option | Rôle |
|--------|------|
| `-d` | Détaché (arrière-plan) |
| `--name ollama` | Nom explicite pour `podman stop ollama` |
| `-p 11434:11434` | Port hôte:conteneur (API accessible sur localhost:11434) |
| `-v ollama-data:/root/.ollama` | Persistance modèles dans volume nommé |
| `local-ia/ollama:tiny` | Image construite à l'étape 2 |

**Vérifier** :
```bash
podman logs ollama           # logs du serveur
curl http://localhost:11434/api/tags  # liste modèles (JSON)
```

---

## 3bis. Démonstration : fonctionnement 100% local (sans réseau)

Pour prouver qu'aucune connexion externe n'est nécessaire **après le build**, on lance le conteneur **sans réseau** :

```bash
# Arrête le conteneur précédent si besoin
podman stop ollama && podman rm ollama

# Lance SANS réseau (--network none)
# Note: --network none ne supporte PAS le port mapping (-p) dans Podman
podman run -d \
  --name ollama-local \
  --network none \
  -v ollama-data:/root/.ollama \
  local-ia/ollama:tiny
```

**Vérifier l'isolation** :
```bash
# Depuis le conteneur : modèle tourne en local (pas de pull, pas de réseau)
podman exec ollama-local /bin/ollama run smollm2:135m "test"
# → génère une réponse (spinner visible), modèles déjà dans l'image

# Aucun accès réseau externe (ping pas installé, pas de DNS, pas d'IP)
podman exec ollama-local ping -c 1 8.8.8.8 2>&1 || echo "ping échoué (attendu : pas de réseau, ping absent)"
```

**Explications** :
| Option | Rôle |
|--------|------|
| `--network none` | Pas d'interface réseau dans le conteneur (isolé total) |
| Modèles dans l'image | Aucun `pull` au runtime, inférence 100% locale |
| Pas de `-p` | Port mapping impossible avec `--network none` (limite Podman) |

> **Preuve** : le conteneur démarre, le modèle répond, mais **aucune connexion sortante n'est possible**. L'inférence est 100% locale.

> **Pour l'API REST** : utilisez le conteneur avec réseau (étape 3). Les modèles sont déjà dans l'image, donc **zéro téléchargement au démarrage**.

---

## 4. Utilisation CLI - `podman exec`

Si vous avez fait le test sans réseau pensez à relancer le conteneur *ollama* (étape 3)

```bash
# REPL interactive dans le conteneur (équivalent `ollama run`)
podman exec -it ollama ollama run qwen2.5:0.5b
```

**Explications** :
- `-it` : interactif + TTY (nécessaire pour REPL)
- `ollama run qwen2.5:0.5b` : commande *dans* le conteneur
- `Ctrl+D` ou `/bye` pour quitter

---

## 5. API REST - Exemples pas à pas (depuis l'hôte)

### 5.1 Chat non-streaming (conversation multi-tours)

**Commande** :
```bash
curl -s http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:0.5b",
    "messages": [
      {"role": "system", "content": "Tu es un assistant concis."},
      {"role": "user", "content": "C'\''est quoi un embedding ?"}
    ],
    "stream": false
  }' | jq -r '.message.content'
```

**Explications** :
- `curl -s` : silencieux (pas de barre de progression)
- `-H "Content-Type: application/json"` : header JSON
- `-d '{...}'` : payload JSON (modèle, messages, stream=false)
- `| jq -r '.message.content'` : extrait le champ texte de la réponse

**Flux** : client → POST /api/chat → Ollama → modèle → JSON response → `jq` filtre.

### 5.2 Generate (complétion simple, sans historique)

**Différence clé** : `/api/generate` = complétion unique (prompt brut, pas de rôles), `/api/chat` = conversation structurée.

#### 5.2.1 Réponse pour humain (texte libre)

```bash
curl -s http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:0.5b",
    "prompt": "La capitale de la France est",
    "stream": false
  }' | jq -r '.response'
```

**Sortie** : `Paris. C'est une ville...` (phrase complète, langage naturel)

#### 5.2.2 Réponse pour machine (JSON strict, exploitable)

Pour obtenir une réponse parsable par un autre process, on force le format via le prompt système + options :

```bash
curl -s http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:0.5b",
    "prompt": "Réponds UNIQUEMENT par un JSON valide : {\"capital\": \"...\"}. Question : Quelle est la capitale de la France ?",
    "stream": false,
    "options": {
      "temperature": 0,
      "num_predict": 16
    }
  }' | jq '.response | fromjson'
```

**Sortie** : `{"capital": "Paris"}` (JSON valide, `jq` le parse avec `fromjson`)

**Paramètres clés** :
- `temperature: 0` : déterministe (toujours même réponse)
- `num_predict: 16` : limite tokens (économique, évite verbiage)
- Prompt explicite : exige format JSON exact

> **Usage machine** : `capital=$(curl ... | jq -r '.response | fromjson | .capital')` → variable shell = `Paris`

### 5.3 Streaming (temps réel, token par token)

**Commande** :
```bash
curl -s http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:0.5b",
    "messages": [{"role": "user", "content": "Écris un haïku sur le code."}],
    "stream": true
  }' | while read line; do
      echo "$line" | jq -r '.message.content // empty' | tr -d '\n'
  done
echo
```

**Explications** :
- `stream: true` : renvoie du NDJSON (une ligne JSON par token)
- `while read line` : lit chaque ligne au fur et à mesure (flux continu)
- `jq -r '.message.content // empty'` : extrait le token, ignore lignes sans contenu
- `tr -d '\n'` : supprime newlines pour affichage fluide
- `echo` final : ajoute le newline de fin

**Pourquoi cette boucle** : `curl` sort le flux brut NDJSON ; la boucle traite chaque ligne JSON au fil de l'eau.

---

## 6. Personnaliser un modèle (Modelfile)

Crée un fichier `Modelfile` (sur l'hôte) :

```dockerfile
# Modelfile : définit un modèle dérivé
FROM qwen2.5:0.5b                              # modèle de base (déjà dans l'image)
SYSTEM "Tu es un expert data engineering. Réponds en français, max 3 phrases."  # prompt système
PARAMETER temperature 0.3                      # créativité (0 = déterministe, 1 = créatif)
PARAMETER top_p 0.9                            # noyau d'échantillonnage
```

Puis crée le modèle dans le conteneur :

```bash
# Copie le Modelfile dans le conteneur et crée le modèle
podman cp Modelfile ollama:/tmp/Modelfile
podman exec ollama ollama create mon-expert -f /tmp/Modelfile
podman exec -it ollama ollama run mon-expert   # test
```

---

## 7. Gestion des données (volume)

```bash
# Liste volumes
podman volume ls

# Inspecte le volume (point de montage hôte)
podman volume inspect ollama-data

# Sauvegarde (export tar)
podman run --rm -v ollama-data:/data -v $(pwd):/backup alpine tar czf /backup/ollama-backup.tar.gz -C /data .

# Restauration
podman run --rm -v ollama-data:/data -v $(pwd):/backup alpine tar xzf /backup/ollama-backup.tar.gz -C /data
```

---

## 8. Exercices + Corrections

### Exercice 1 - Modèle « Pirate »

**Consigne** : Crée un Modelfile qui force le ton « pirate ».

**Correction** :
```dockerfile
# Modelfile.pirate
FROM qwen2.5:0.5b
SYSTEM "Tu es un pirate. Parle comme un pirate : 'Arrr', 'matelot', 'trésor', etc. Réponds en français."
PARAMETER temperature 0.8
PARAMETER top_p 0.95
```
```bash
podman cp Modelfile.pirate ollama:/tmp/Modelfile.pirate
podman exec ollama ollama create pirate -f /tmp/Modelfile.pirate
podman exec -it ollama ollama run pirate
# Test : "Bonjour, comment ça va ?" → "Arrr matelot ! Ça va bien, le vent est bon..."
```

---

### Exercice 2 - Comparer modèles

**Consigne** : Pose la même question à `smollm2:135m` et `qwen2.5:0.5b`, note la différence.

**Correction** :
```bash
# Question identique
Q="Explique la différence entre HTTP et HTTPS en 2 phrases."

# Modèle 1 (très petit)
podman exec ollama ollama run smollm2:135m "$Q"
# → Réponse courte, parfois imprécise, vocabulaire limité

# Modèle 2 (meilleur)
podman exec ollama ollama run qwen2.5:0.5b "$Q"
# → Réponse structurée, termes techniques corrects (TLS, certificat, chiffrement)
```
**Observation attendue** : Qwen2.5 0.5B >> SmolLM2 135M en qualité, mais 4x plus lent.

---

### Exercice 3 - API Python (client `ollama`)

**Consigne** : Installe `ollama` (`pip install ollama`) et réécris le `curl` chat en Python.

**Correction** :
```bash
# Dans un venv (recommandé)
python3 -m venv .venv && source .venv/bin/activate
pip install ollama
```

```python
# client.py
import ollama

# Client pointe vers localhost:11434 (défaut)
client = ollama.Client(host="http://localhost:11434")

# Chat non-streaming
resp = client.chat(
    model="qwen2.5:0.5b",
    messages=[
        {"role": "system", "content": "Tu es un assistant concis."},
        {"role": "user", "content": "C'est quoi un embedding ?"}
    ],
    stream=False
)
print(resp["message"]["content"])

# Chat streaming
stream = client.chat(
    model="qwen2.5:0.5b",
    messages=[{"role": "user", "content": "Compte de 1 à 5"}],
    stream=True
)
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
print()
```

```bash
python3 client.py
```

---

## 9. Nettoyage

```bash
# Arrête et supprime le conteneur
podman stop ollama && podman rm ollama

# Supprime l'image (libère ~2 Go)
podman rmi local-ia/ollama:tiny

# Supprime le volume (supprime modèles + configs)
podman volume rm ollama-data
```

---

## 10. Aller plus loin

- [Ollama API docs](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Modelfile reference](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- [Podman docs](https://docs.podman.io/)
- Prochaine étape : **02-raw-llama** - même chose sans Ollama, avec `llama.cpp` en conteneur.