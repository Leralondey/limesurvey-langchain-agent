# streamlit_app.py
import streamlit as st
import json
from dotenv import load_dotenv
import logging
import atexit # Pour le nettoyage

# Importer les composants clés de votre agent existant
from main_api_agent import app as langgraph_app
from main_api_agent import MainAgentGraphState
from langchain_core.messages import HumanMessage, AIMessage

from tools.limesurvey_api_client import get_session_key_api, release_session_key_api

# Configuration du logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
# Streamlit a son propre logger, donc on configure le logger racine pour nos modules.
# Pour voir les logs dans la console où Streamlit est lancé:
logging.basicConfig(level=logging.INFO, format=log_format, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__) # Logger spécifique à ce fichier Streamlit

# Charger les variables d'environnement
if load_dotenv():
    logger.info(".env file loaded by Streamlit app.")
else:
    logger.warning(".env file not found. Ensure API keys and other configs are in environment.")


# --- Configuration de la Page Streamlit ---
st.set_page_config(page_title="Assistant IA LimeSurvey", layout="wide")
st.title("🤖 Assistant IA pour l'exploration des données LimeSurvey")
st.caption("Posez des questions en langage naturel sur vos enquêtes LimeSurvey.")

# --- Gestion de l'État de Session Streamlit ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # Historique des messages de la conversation actuelle
    logger.info("Historique des messages Streamlit initialisé.")

if "limesurvey_session_key" not in st.session_state:
    st.session_state.limesurvey_session_key = None
    logger.info("Clé de session LimeSurvey dans Streamlit initialisée à None.")

# --- Initialisation de la Session API LimeSurvey (une seule fois par session Streamlit) ---
if st.session_state.limesurvey_session_key is None:
    logger.info("Tentative d'obtention de la clé de session LimeSurvey pour la session Streamlit...")
    with st.spinner("Connexion à l'API LimeSurvey..."):
        try:
            key = get_session_key_api()
            if key:
                st.session_state.limesurvey_session_key = key
                logger.info(f"Clé de session LimeSurvey obtenue : {key[:10]}...")
                # st.sidebar.success("Connecté à l'API LimeSurvey.") # Optionnel, si vous utilisez une sidebar
            else:
                error_message = "Échec de l'obtention de la clé de session LimeSurvey. Vérifiez les logs du client API."
                logger.error(error_message)
                st.error(error_message + " L'application ne peut pas continuer.")
                st.stop() 
        except Exception as e:
            error_message = f"Erreur critique lors de la connexion à LimeSurvey : {str(e)}"
            logger.critical(error_message, exc_info=True)
            st.error(error_message + " L'application ne peut pas continuer.")
            st.stop()

# --- Nettoyage de la session API à la fin de l'exécution du script ---
# Cela ne fonctionne que lorsque le script Python s'arrête,
# pas nécessairement quand l'utilisateur ferme l'onglet.
def cleanup_limesurvey_session_on_exit():
    if st.session_state.get("limesurvey_session_key"): # Utiliser .get pour éviter KeyError si non défini
        logger.info("Nettoyage atexit : Libération de la clé de session LimeSurvey...")
        release_session_key_api()
        st.session_state.limesurvey_session_key = None # Mettre à None dans l'état de session
        logger.info("Clé de session LimeSurvey libérée (atexit).")

if "atexit_registered" not in st.session_state:
    atexit.register(cleanup_limesurvey_session_on_exit)
    st.session_state.atexit_registered = True
    logger.info("Fonction de nettoyage atexit enregistrée.")


# --- Affichage de l'Historique des Messages ---
for msg_idx, message_obj in enumerate(st.session_state.messages):
    role = "user" if message_obj.type == "human" else "assistant"
    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
        content_to_display = message_obj.content
        if content_to_display:
             st.markdown(content_to_display)


# --- Champ de Saisie Utilisateur et Logique d'Appel de l'Agent ---
if user_prompt := st.chat_input("Posez votre question ici..."):
    logger.info(f"Requête utilisateur reçue: '{user_prompt}'")
    
    # Ajouter le message utilisateur à l'historique Streamlit et l'afficher
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_prompt)

    # Préparer l'état pour l'appel au graphe LangGraph
    # L'historique complet (st.session_state.messages) est passé.
    graph_input_state = MainAgentGraphState(
        user_query=user_prompt,
        messages=st.session_state.messages, # Contient déjà le dernier HumanMessage
        final_output=None
    )
    
    logger.debug(f"État initial pour LangGraph: {graph_input_state}")

    # Afficher un message "en cours de traitement" et appeler l'agent
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty() # Pour afficher les "pensées" ou la réponse finale
        with st.spinner("L'agent IA réfléchit et consulte les données..."):
            try:
                # Invoquer le graphe LangGraph
                # La configuration de récursion est pour le graphe lui-même, pas pour l'AgentExecutor.
                # Note: `app.stream()` pourrait être utilisé ici pour des mises à jour en temps réel des étapes de l'agent,
                # mais cela nécessite une gestion plus complexe de l'affichage des ToolMessages.
                # Pour l'instant, `invoke` est plus simple pour obtenir la réponse finale.
                
                logger.info("Invocation de LangGraph app...")
                final_graph_state_result = langgraph_app.invoke(graph_input_state, config={"recursion_limit": 15})
                logger.info(f"Résultat de LangGraph app: {str(final_graph_state_result)[:500]}...")

                if final_graph_state_result and "final_output" in final_graph_state_result:
                    ai_response_content = final_graph_state_result["final_output"]
                    if not ai_response_content: # Si final_output est vide
                         ai_response_content = "L'agent n'a pas produit de réponse textuelle explicite mais a peut-être terminé ses actions."
                else:
                    ai_response_content = "L'agent n'a pas retourné de sortie dans le format attendu."
                    logger.error(f"Structure de réponse inattendue de LangGraph: {final_graph_state_result}")

            except Exception as e:
                logger.error(f"Erreur critique lors de l'invocation de LangGraph: {e}", exc_info=True)
                ai_response_content = f"Désolé, une erreur technique est survenue lors du traitement de votre demande : {str(e)}"
        
        # Afficher la réponse de l'IA
        message_placeholder.markdown(ai_response_content)

    # Ajouter la réponse de l'IA à l'historique Streamlit
    st.session_state.messages.append(AIMessage(content=ai_response_content))
    logger.info("Réponse de l'IA ajoutée à l'historique Streamlit.")

# Optionnel : Bouton pour réinitialiser la conversation
if st.sidebar.button("Réinitialiser la conversation"):
    st.session_state.messages = []
    logger.info("Historique de conversation Streamlit réinitialisé par l'utilisateur.")
    # La clé de session LimeSurvey N'EST PAS réinitialisée ici, elle reste active pour la session Streamlit.
    st.rerun()

st.sidebar.info("Cette application utilise un agent IA pour interroger l'API LimeSurvey. Les logs de l'agent (y compris les appels d'outils) sont visibles dans la console où Streamlit est lancé.")