# config/llm_config.py
import os
from dotenv import load_dotenv
import logging # Ajout du logging

# Configuration du logger pour ce module
logger = logging.getLogger(__name__)

# Charger les variables d'environnement à partir du fichier .env
# Cela rendra les clés API disponibles via os.environ.get()
if load_dotenv():
    logger.debug(".env file loaded.")
else:
    logger.debug(".env file not found or not loaded. Relying on environment variables directly.")


# Récupérer la clé API OpenAI depuis les variables d'environnement
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    # Lever une exception si la clé API OpenAI est indispensable et non trouvée.
    error_message = "CRITICAL: OPENAI_API_KEY not found. Ensure it is defined in your .env file or environment."
    logger.critical(error_message)
    raise ValueError(error_message)
else:
    logger.info("OpenAI API Key loaded successfully.")

# Configuration pour les modèles OpenAI
# C'est une liste, car certains frameworks (comme Autogen) peuvent gérer plusieurs configurations.
# Pour LangChain, nous utiliserons généralement la première configuration de cette liste.
config_list_openai = [
    {
        # Modèle recommandé pour l'agent principal pour ses capacités de raisonnement et d'utilisation d'outils.
        # gpt-4o-mini peut être trop limitant pour des tâches complexes d'agent.
        "model": "gpt-4o", # Assurez-vous que ce modèle est disponible pour votre clé API
        # Alternatives : "gpt-4-turbo", "gpt-4-turbo-preview"
        # Ancien modèle pour tests : "gpt-4o-mini"
        "api_key": openai_api_key,
        # Vous pouvez ajouter d'autres paramètres ici si nécessaire pour ChatOpenAI,
        # comme "temperature", bien qu'il soit souvent préférable de le définir
        # lors de l'instanciation du client LLM.
    }
    # Vous pourriez ajouter d'autres configurations de modèles ici si nécessaire.
]

# Fonction pour récupérer la configuration LLM principale
def get_main_llm_config():
    if not config_list_openai:
        error_message = "No OpenAI configuration is defined in config_list_openai."
        logger.error(error_message)
        raise ValueError(error_message)
    # Retourne le premier dictionnaire de configuration
    return config_list_openai[0]

# Fonction pour récupérer le LLM LangChain directement
def get_langchain_llm(temperature=0.0, streaming=False, model_name: str = None, max_tokens: int = None):
    from langchain_openai import ChatOpenAI # Importation locale pour éviter les dépendances circulaires si ce module est importé tôt
    
    llm_params = get_main_llm_config()
    
    current_model_name = model_name if model_name else llm_params["model"]
    logger.info(f"Initializing ChatOpenAI with model: {current_model_name}, temperature: {temperature}, streaming: {streaming}")

    config_args = {
        "model_name": current_model_name,
        "openai_api_key": llm_params["api_key"],
        "temperature": temperature,
        "streaming": streaming
    }
    if max_tokens is not None:
        config_args["max_tokens"] = max_tokens
        
    return ChatOpenAI(**config_args)

if __name__ == '__main__':
    # Configuration simple du logging pour les tests de ce module
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("--- Running llm_config.py tests ---")
    try:
        main_config = get_main_llm_config()
        logger.info(f"Main LLM configuration: {main_config}")

        # Test d'initialisation du LLM LangChain
        llm = get_langchain_llm()
        logger.info(f"LangChain LLM initialized with model: {llm.model_name}")
        
        # Vérifier les variables d'environnement de LimeSurvey API
        limesurvey_api_url = os.environ.get("LIMESURVEY_API_URL")
        limesurvey_api_user = os.environ.get("LIMESURVEY_API_USER")
        # LIMESURVEY_API_PASSWORD peut être une chaîne vide, donc on vérifie juste sa présence
        limesurvey_api_password_exists = "LIMESURVEY_API_PASSWORD" in os.environ

        if not all([limesurvey_api_url, limesurvey_api_user, limesurvey_api_password_exists]):
            logger.warning("One or more LimeSurvey API environment variables are not fully defined.")
            logger.warning(f"  LIMESURVEY_API_URL: {'Defined' if limesurvey_api_url else 'NOT DEFINED'}")
            logger.warning(f"  LIMESURVEY_API_USER: {'Defined' if limesurvey_api_user else 'NOT DEFINED'}")
            logger.warning(f"  LIMESURVEY_API_PASSWORD: {'Exists in env' if limesurvey_api_password_exists else 'NOT IN ENV'}")
        else:
            logger.info("LimeSurvey API environment variables (URL, USER, PASSWORD_EXISTS) seem to be loaded.")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during llm_config.py tests: {e}", exc_info=True)
    logger.info("--- llm_config.py tests finished ---")