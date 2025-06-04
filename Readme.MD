# Assistant IA pour LimeSurvey avec LangChain et Streamlit

Ce projet vise à créer un assistant intelligent capable d'analyser et de répondre en langage naturel à des questions concernant les données des enquêtes LimeSurvey. L'assistant utilise l'API RemoteControl 2 (RC2) de LimeSurvey pour l'extraction de données et est construit avec LangChain pour l'orchestration de l'agent IA, et Streamlit pour l'interface utilisateur.

## Fonctionnalités Principales

L'assistant est conçu pour :
*   Comprendre les requêtes utilisateur en langage naturel.
*   Récupérer les structures des enquêtes LimeSurvey (titres, groupes de questions, questions).
*   Récupérer les données de réponse aux enquêtes.
*   Effectuer des analyses thématiques sur le contenu des enquêtes.
*   Analyser les réponses à des questions spécifiques (distributions, sentiments si les données le permettent).
*   Fournir des synthèses et des conclusions basées sur les données extraites.

## Architecture

*   **Agent Principal (Orchestrateur) :** Un agent LangChain (`main_api_agent.py`) utilisant le modèle GPT d'OpenAI (configurable, `gpt-4o` recommandé) et des outils spécifiques pour interagir avec l'API LimeSurvey.
*   **Client API LimeSurvey :** Un module Python dédié (`tools/limesurvey_api_client.py`) pour toutes les communications avec l'API RC2 de LimeSurvey.
*   **Agent d'Analyse de Données :** Un LLM "spécialiste" (guidé par `prompts/data_analyzer_prompt.txt`) appelé via un outil de l'agent principal pour effectuer des analyses de données ou thématiques.
*   **Interface Utilisateur :** Une application web interactive construite avec Streamlit (`streamlit_app.py`).
*   **Prompts :** Des prompts système détaillés (`prompts/`) guident le comportement des agents.
*   **Configuration :** Gérée via un fichier `.env` et `config/llm_config.py`.

## Prérequis

*   Python 3.10 ou supérieur
*   Une instance LimeSurvey (Community Edition Version 6.x ou compatible avec l'API RC2) accessible via son URL d'API.
*   Un compte API LimeSurvey avec les permissions nécessaires pour :
    *   Obtenir/libérer une clé de session.
    *   Lister les sondages.
    *   Obtenir les propriétés des sondages (titre, etc.).
    *   Lister les groupes de questions.
    *   Lister les questions (avec leurs options de réponse).
    *   Exporter les réponses.
*   Une clé API OpenAI.

## Installation

1.  **Cloner le dépôt (si applicable) :**
    ```bash
    git clone [URL_DE_VOTRE_DEPOT_GIT]
    cd limesurvey_autogen 
    ```

2.  **Créer et activer un environnement virtuel Python :**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate 
    # Pour Windows : .venv\Scripts\activate
    ```

3.  **Installer les dépendances :**
    Créez un fichier `requirements.txt` (voir section ci-dessous) puis installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Créer un fichier `.env` :**
    À la racine du projet, créez un fichier nommé `.env` et remplissez-le avec vos informations. Inspirez-vous du modèle suivant :

    ```dotenv
    # Clé API pour OpenAI
    OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

    # URL et identifiants de l'API LimeSurvey RemoteControl 2
    LIMESURVEY_API_URL="http://votre-limesurvey.com/index.php/admin/remotecontrol"
    LIMESURVEY_API_USER="votre_utilisateur_api_limesurvey"
    LIMESURVEY_API_PASSWORD="votre_mot_de_passe_api_limesurvey"

    # Optionnel: Configuration du modèle LLM si différent de celui par défaut dans llm_config.py
    # DEFAULT_MODEL_NAME="gpt-4o" 
    ```
    **Note de Sécurité :** N'ajoutez jamais votre fichier `.env` à un système de contrôle de version public (comme Git). Assurez-vous qu'il est listé dans votre fichier `.gitignore`.

2.  **Vérifier la configuration du modèle LLM :**
    Le modèle LLM principal est configuré dans `config/llm_config.py`. Par défaut, il est réglé sur `gpt-4o`. Assurez-vous que votre clé API OpenAI a accès à ce modèle ou modifiez-le en conséquence.

## Fichier `requirements.txt` (Exemple)

Voici une liste des dépendances principales. **Il est fortement recommandé de générer votre propre fichier `requirements.txt` à partir de votre environnement virtuel actif avec `pip freeze > requirements.txt`** pour garantir les bonnes versions.

```txt
python-dotenv
langchain
langchain-openai
langgraph
streamlit
requests
pydantic>=2.0,<3.0 # Pour être compatible avec les dernières versions de LangChain
# Pillow # Optionnel, pour la visualisation du graphe LangGraph
# termcolor # Souvent utilisé par LangChain pour les logs colorés
```
*Adaptez cette liste en fonction des paquets exacts que vous avez installés.*

## Utilisation

1.  **Activer l'environnement virtuel :**
    ```bash
    source .venv/bin/activate 
    # Pour Windows : .venv\Scripts\activate
    ```

2.  **Lancer l'application Streamlit :**
    ```bash
    streamlit run streamlit_app.py
    ```
    L'application devrait s'ouvrir dans votre navigateur web par défaut à une adresse locale (généralement `http://localhost:8501`).

3.  **Interagir avec l'assistant :**
    Utilisez le champ de saisie dans l'interface Streamlit pour poser vos questions concernant les données LimeSurvey.

## Tester l'Agent en Ligne de Commande (Optionnel)

Le fichier `main_api_agent.py` contient un bloc `if __name__ == "__main__":` qui permet d'exécuter des tests prédéfinis en ligne de commande. Ces tests sont utiles pour le débogage direct de l'agent sans passer par l'interface Streamlit.

Pour lancer ces tests :
```bash
python3 main_api_agent.py
```
Les logs détaillés (y compris les pensées de l'agent et les appels d'API) s'afficheront dans la console. Les résultats finaux des tests sont également sauvegardés dans `main_api_agent_batch_test_results.json`.

## Structure du Projet

```
limesurvey_autogen/
├── .env                     # Variables d'environnement (NON VERSIONNÉ)
├── config/
│   └── llm_config.py        # Configuration du LLM
├── tools/
│   └── limesurvey_api_client.py # Client API LimeSurvey
├── prompts/
│   ├── api_main_agent_prompt.txt  # Prompt de l'agent orchestrateur
│   └── data_analyzer_prompt.txt   # Prompt de l'agent d'analyse
├── main_api_agent.py          # Logique de l'agent principal et tests en CLI
├── streamlit_app.py           # Application Streamlit
├── requirements.txt           # Dépendances Python
└── README.md                  # Ce fichier
```

## Limitations et Pistes d'Amélioration

*   **Gestion des Erreurs :** L'agent tente de gérer les erreurs d'API, mais des cas non prévus peuvent survenir.
*   **Précision de l'Analyseur de Données :** L'exactitude de l'analyse des réponses dépend de la qualité des prompts et de la capacité du LLM à interpréter les données fournies (notamment l'identification des colonnes de questions et l'utilisation des mappings de codes).
*   **Support Multilingue :** L'agent peut être configuré pour utiliser des langues spécifiques pour certaines API, mais une gestion multilingue complète de l'interface et des analyses nécessiterait plus de développement.
*   **Analyses Complexes :** Pour des analyses statistiques très poussées ou des croisements de données complexes entre enquêtes, des outils plus spécialisés pourraient être nécessaires en complément.

Les futures améliorations pourraient inclure :
*   Une meilleure gestion de l'identification des codes de question pour l'analyseur.
*   L'extraction et le passage automatiques des mappings de codes de réponse à l'analyseur.
*   Une interface Streamlit plus riche avec sélection de sondages/questions et visualisations.

## Dépannage

*   **`ValueError: Single '}' encountered in format string` :** Vérifiez les fichiers de prompts (`prompts/`) pour des accolades non doublées. Les accolades littérales doivent être `{{` ou `}}`.
*   **Erreurs d'API LimeSurvey :** Consultez les logs de `tools.limesurvey_api_client.py` (niveau DEBUG) pour voir les requêtes et réponses brutes. Vérifiez les permissions de votre utilisateur API LimeSurvey et l'état de votre instance LimeSurvey. Le message `'At least one column must be selected for export.'` indique souvent que le paramètre `aFields` pour `export_responses` doit être `null` (Python `None`) plutôt qu'un tableau vide `[]` pour exporter tous les champs.
*   **Erreurs de clé API OpenAI :** Assurez-vous que `OPENAI_API_KEY` est correctement défini dans `.env` et que la clé est valide et a accès aux modèles requis.
*   **Avertissements Pydantic :** Si vous voyez des avertissements concernant `langchain_core.pydantic_v1`, mettez à jour vos imports dans les fichiers Python (comme `main_api_agent.py`) de `from langchain_core.pydantic_v1 import BaseModel, Field` vers `from pydantic import BaseModel, Field`, en vous assurant d'avoir `pydantic` v2+ installé.

---
Développé par : Marco (Intermag Business Optimizer Solutions) 