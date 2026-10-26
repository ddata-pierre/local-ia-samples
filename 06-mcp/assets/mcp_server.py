#!/usr/bin/env python3
# mcp_server.py - Mini Serveur MCP avec FastMCP

from mcp.server.fastmcp import FastMCP

# Initialisation du serveur MCP
mcp = FastMCP("DData-Enterprise-Tools")

# Base de données fictive pour la démonstration
STOCKS = {
    "SERVEUR-N100": {"stock": 14, "emplacement": "Entrepôt Marseille"},
    "ECRAN-4K": {"stock": 0, "emplacement": "Rupture de stock"},
    "CABLE-ETH-10M": {"stock": 150, "emplacement": "Entrepôt Aix"}
}

METEO_SITES = {
    "marseille": "Soleil, 22°C, Vent 15 km/h",
    "aix": "Nuageux, 19°C, Vent faible",
    "paris": "Pluie, 12°C, Vent 30 km/h"
}

@mcp.tool()
def verifier_stock(article_id: str) -> str:
    """Vérifie la disponibilité et l'emplacement d'un article dans les entrepôts.
    
    Args:
        article_id: L'identifiant exact de l'article (ex: SERVEUR-N100, ECRAN-4K, CABLE-ETH-10M).
    """
    article = article_id.upper()
    if article in STOCKS:
        info = STOCKS[article]
        return f"Article {article} : {info['stock']} unités disponibles ({info['emplacement']})."
    return f"Article {article} non trouvé dans le catalogue."

@mcp.tool()
def obtenir_meteo_site(ville: str) -> str:
    """Obtient les conditions météo actuelles pour un site d'entreprise.
    
    Args:
        ville: Nom de la ville du site (marseille, aix, paris).
    """
    v = ville.lower()
    if v in METEO_SITES:
        return f"Météo actuelle pour le site de {ville.capitalize()} : {METEO_SITES[v]}"
    return f"Aucun site répertorié dans la ville de {ville}."

if __name__ == "__main__":
    # Lance le serveur sur les flux d'entrée/sortie standard (STDIO)
    mcp.run(transport="stdio")