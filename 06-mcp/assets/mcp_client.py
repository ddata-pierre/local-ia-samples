#!/usr/bin/env python3
# mcp_client.py — Client MCP + Inférence LLM locale

# === PROMPTS (modifiables sans toucher à la logique) ===
# Le prompt système est construit dynamiquement avec les outils découverts
TOOL_CALL_FORMAT = """Tu es un assistant connecté à un serveur MCP.
Tu as accès aux outils suivants :
{tools_json}

Pour utiliser un outil, réponds UNIQUEMENT sous la forme JSON suivante :
{{"outil": "nom_de_l_outil", "arguments": {{"parametre": "valeur"}}}}"""

FINAL_SYNTHESIS_PROMPT = "Rédige une réponse claire et courtoise à l'utilisateur à partir du résultat de l'outil."

# === LOGIQUE ===
import asyncio
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from llama_cpp import Llama

# 1. Charger le LLM local
MODEL_PATH = "/models/qwen2.5-3b-instruct-q4_k_m.gguf"
print("🔄 Chargement du modèle LLM local...")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)


async def executer_demonstration_mcp():
    # Configuration pour lancer le serveur MCP en arrière-plan via STDIO
    server_params = StdioServerParameters(
        command="python3",
        args=["/home/appuser/mcp_server.py"],
        env=None
    )

    print("🔌 Connexion au serveur MCP...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Étape A : Initialisation de la session MCP
            await session.initialize()
            
            # Étape B : Découverte dynamique des outils MCP
            mcp_tools = await session.list_tools()
            tools_description = []
            
            for tool in mcp_tools.tools:
                tools_description.append({
                    "nom": tool.name,
                    "description": tool.description,
                    "schema": tool.inputSchema
                })
            
            print(f"✅ Outils découverts sur le serveur MCP : {[t['nom'] for t in tools_description]}")

            # Étape C : Soumission d'une question nécessitant l'outil
            question = "Combien de serveurs N100 reste-t-il en stock et où sont-ils ?"
            print(f"\n❓ Question utilisateur : '{question}'")

            # Construction du prompt avec la définition des outils découverts
            system_prompt = TOOL_CALL_FORMAT.format(
                tools_json=json.dumps(tools_description, indent=2, ensure_ascii=False)
            )

            # Appel LLM pour choisir l'outil
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.0
            )

            choix_llm = response["choices"][0]["message"]["content"]
            print(f"🤖 Décision du LLM (Appel d'outil) :\n{choix_llm}")

            # Étape D : Exécution de l'outil via le protocole MCP
            try:
                action = json.loads(choix_llm)
                tool_name = action["outil"]
                tool_args = action["arguments"]

                print(f"⚙️ Exécution de l'outil MCP '{tool_name}' avec {tool_args}...")
                
                # Exécution via la session MCP
                mcp_result = await session.call_tool(tool_name, tool_args)
                result_text = mcp_result.content[0].text
                print(f"📊 Résultat renvoyé par le Serveur MCP :\n{result_text}")

                # Étape E : Synthèse finale par le LLM avec le résultat de l'outil
                final_response = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": FINAL_SYNTHESIS_PROMPT},
                        {"role": "user", "content": f"Question : {question}\nRésultat de l'outil MCP : {result_text}"}
                    ],
                    temperature=0.3
                )
                
                print(f"\n💬 Réponse finale à l'utilisateur :\n{final_response['choices'][0]['message']['content']}")

            except (json.JSONDecodeError, KeyError) as e:
                print(f"❌ Erreur lors de l'analyse de la décision de l'outil : {e}")

if __name__ == "__main__":
    asyncio.run(executer_demonstration_mcp())