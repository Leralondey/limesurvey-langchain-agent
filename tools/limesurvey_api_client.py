# tools/limesurvey_api_client.py
import requests
import json
import os
import base64
from typing import Optional, List, Dict, Any, Union
import logging
import time

# Configuration du logger
logger = logging.getLogger(__name__)

LIMESURVEY_API_URL = os.environ.get("LIMESURVEY_API_URL")
LIMESURVEY_API_USER = os.environ.get("LIMESURVEY_API_USER")
LIMESURVEY_API_PASSWORD = os.environ.get("LIMESURVEY_API_PASSWORD")

_session_key: Optional[str] = None
_http_session: Optional[requests.Session] = None
_session_key_last_obtained_time: Optional[float] = None
SESSION_KEY_EXPIRATION_PERIOD_SECONDS = 2 * 60 * 60  # 2 heures

ApiResult = Union[List[Any], Dict[str, Any], str]

def _get_http_session() -> requests.Session:
    """Initialise et retourne une session requests persistante."""
    global _http_session
    if _http_session is None:
        logger.debug("Initializing new HTTP session for LimeSurvey API client.")
        _http_session = requests.Session()
        _http_session.headers.update({'Content-Type': 'application/json'})
    return _http_session

def _mask_sensitive_params(method: str, params: list) -> list:
    """Masque les paramètres sensibles comme le mot de passe dans les logs."""
    if method == "get_session_key" and len(params) >= 2:
        masked_params = list(params)
        masked_params[1] = "****MASKED_PASSWORD****"
        return masked_params
    return params

def _handle_api_error_response(response_json: Dict[str, Any], method: str, logged_params: list) -> Dict[str, Any]:
    """Gère les erreurs JSON structurées retournées par l'API LimeSurvey."""
    error_content = response_json.get('error', 'Unknown error structure in API JSON response')
    error_message_detail = error_content.get('status', str(error_content)) if isinstance(error_content, dict) else str(error_content)
    logger.warning(f"LimeSurvey API Error (from JSON response): Method='{method}', Error='{error_message_detail}'")
    
    error_to_return = error_content if isinstance(error_content, dict) else error_message_detail
    return {"error": error_to_return, "method_called": method, "params_sent_for_log": logged_params}

def _make_api_request(method: str, params: list, attempt: int = 1) -> Dict[str, Any]:
    """
    Effectue un appel API JSON-RPC à LimeSurvey avec gestion des erreurs et de la session.
    """
    if not LIMESURVEY_API_URL:
        logger.error("LIMESURVEY_API_URL is not configured.")
        return {"error": "LIMESURVEY_API_URL is not configured."}

    if method != "get_session_key":
        current_session_key = get_session_key_api()
        if not current_session_key:
            return {"error": f"Failed to obtain/refresh session key for method {method}."}
        if params and params[0] == "__SESSION_KEY_PLACEHOLDER__":
            params[0] = current_session_key

    payload = {"method": method, "params": params, "id": 1}
    logged_params_for_display = _mask_sensitive_params(method, params)
    logger.debug(f"API Request: Method='{method}', Params='{str(logged_params_for_display)[:200]}...' (Attempt {attempt})")

    http_session = _get_http_session()
    response_text_content = "N/A (response object not created)" 

    try:
        response = http_session.post(LIMESURVEY_API_URL, data=json.dumps(payload), timeout=45)
        response_text_content = response.text 
        
        response.raise_for_status() 
        
        try:
            response_json = response.json()
        except json.JSONDecodeError as jde:
            logger.error(f"JSONDecodeError for method {method}. HTTP Status: {response.status_code}. Raw response text (first 500 chars): '{response_text_content[:500]}'. Error: {jde}")
            return {
                "error": f"Invalid JSON response from API (HTTP {response.status_code}). See logs for raw text.",
                "method_called": method,
                "raw_response_excerpt": response_text_content[:200],
                "original_exception": str(jde)
            }

        if response_json.get("error") is not None:
            api_error_details = _handle_api_error_response(response_json, method, logged_params_for_display)
            error_str = str(api_error_details.get("error")).lower()
            if "invalid session key" in error_str and method != "get_session_key" and attempt == 1:
                logger.warning(f"Invalid session key detected for method {method}. Attempting to refresh and retry ONCE.")
                release_session_key_api(clear_locally_only=True)
                
                params_for_retry = list(params)
                if params_for_retry and params_for_retry[0] == _session_key: 
                    params_for_retry[0] = "__SESSION_KEY_PLACEHOLDER__"
                elif params_for_retry and params_for_retry[0] != "__SESSION_KEY_PLACEHOLDER__":
                    logger.warning(f"Unexpected first parameter in retry logic for method {method}. Expected session key or placeholder.")
                
                return _make_api_request(method, params_for_retry, attempt=2)
            return api_error_details

        logger.debug(f"API Response: Method='{method}', Result Excerpt='{str(response_json.get('result'))[:200]}...'")
        return {"result": response_json.get("result")}

    except requests.exceptions.HTTPError as e: 
        logger.error(f"API HTTP Error: Method='{method}', Status='{e.response.status_code}', Response Text (first 500 chars): '{response_text_content[:500]}', Error='{e}'")
        return {"error": f"HTTP Error {e.response.status_code}: {e.response.reason}. Check logs for response text.", 
                "method_called": method, "raw_response_excerpt": response_text_content[:200]}
    except requests.exceptions.Timeout as e:
        logger.error(f"API Request Timeout: Method='{method}', URL='{LIMESURVEY_API_URL}', Error='{e}'")
        return {"error": f"Request timed out: {str(e)}", "method_called": method}
    except requests.exceptions.RequestException as e: 
        logger.error(f"API Request Exception: Method='{method}', Error='{e}'", exc_info=True)
        return {"error": f"Request failed: {str(e)}", "method_called": method}

def get_session_key_api() -> Optional[str]:
    """Obtient ou réutilise une clé de session LimeSurvey, en gérant l'expiration."""
    global _session_key, _session_key_last_obtained_time

    current_time = time.time()
    if _session_key and _session_key_last_obtained_time:
        if (current_time - _session_key_last_obtained_time) < SESSION_KEY_EXPIRATION_PERIOD_SECONDS:
            logger.info("Using existing non-expired session key.")
            return _session_key
        else:
            logger.info("Existing session key has expired. Clearing local key to obtain a new one.")
            _session_key = None 
            _session_key_last_obtained_time = None

    if not LIMESURVEY_API_USER or LIMESURVEY_API_PASSWORD is None:
        logger.error("LIMESURVEY_API_USER or LIMESURVEY_API_PASSWORD not configured (or password is None).")
        return None

    logger.info("Attempting to obtain a new session key...")
    response_data = _make_api_request("get_session_key", [LIMESURVEY_API_USER, LIMESURVEY_API_PASSWORD])

    if "result" in response_data and isinstance(response_data["result"], str):
        _session_key = response_data["result"]
        _session_key_last_obtained_time = time.time()
        logger.info(f"New session key obtained: {_session_key[:10]}...")
        return _session_key
    else:
        error_detail = response_data.get('error', 'Unknown error or invalid result format for session key')
        logger.error(f"Could not get session key. Detail: {error_detail}")
        _session_key = None
        _session_key_last_obtained_time = None
        return None

def release_session_key_api(clear_locally_only: bool = False) -> None:
    """Libère la clé de session LimeSurvey."""
    global _session_key, _session_key_last_obtained_time
    if not _session_key:
        logger.info("No session key to release or already released.")
        return

    if clear_locally_only:
        logger.info("Clearing session key locally without API call (e.g., for retry logic).")
    else:
        logger.info(f"Attempting to release session key: {_session_key[:10]}...")
        response = _make_api_request("release_session_key", [_session_key])
        if "error" in response:
            logger.warning(f"Error releasing session key via API: {response.get('error')}")
        else:
            logger.info("Session key released successfully via API.")
    
    _session_key = None
    _session_key_last_obtained_time = None

def list_surveys_api() -> ApiResult:
    response_data = _make_api_request("list_surveys", ["__SESSION_KEY_PLACEHOLDER__", None, None])
    return response_data.get("result") if "result" in response_data else response_data

def get_survey_title_api(survey_id: int, language: Optional[str] = None) -> ApiResult:
    params = ["__SESSION_KEY_PLACEHOLDER__", survey_id, ['surveyls_title']]
    if language:
        params.append(language)
    response_data = _make_api_request("get_language_properties", params)
    return response_data.get("result") if "result" in response_data else response_data

def list_groups_api(survey_id: int) -> ApiResult:
    response_data = _make_api_request("list_groups", ["__SESSION_KEY_PLACEHOLDER__", survey_id])
    return response_data.get("result") if "result" in response_data else response_data

def list_questions_api(survey_id: int, group_id: Optional[int] = None, language: Optional[str] = None) -> ApiResult:
    params = ["__SESSION_KEY_PLACEHOLDER__", survey_id, group_id, language]
    response_data = _make_api_request("list_questions", params)
    return response_data.get("result") if "result" in response_data else response_data

def export_responses_api(
    survey_id: int, document_type: str = 'json',
    language: Optional[str] = None,
    completion_status: str = 'all',
    heading_type: str = 'code',
    response_type: str = 'long'
) -> ApiResult:
    params = [
        "__SESSION_KEY_PLACEHOLDER__", survey_id, document_type,
        language, completion_status, heading_type, response_type,
        None,  # sFromResponseID (null for all)
        None,  # sToResponseID (null for all)
        None   # aFields (null for all fields) <--- CORRECTION ICI (None au lieu de [])
    ]
    response_data = _make_api_request("export_responses", params)

    if "result" in response_data and isinstance(response_data["result"], str):
        base64_encoded_data = response_data["result"]
        if not base64_encoded_data: 
            logger.warning(f"export_responses for SID {survey_id} returned an empty string result.")
            # Retourner un format qui ressemble à un export vide mais avec un indicateur d'avertissement,
            # ou une erreur plus explicite si préféré.
            # Pour la cohérence, un export vide pourrait être {"responses": []} après décodage.
            # Mais comme l'API retourne une chaîne vide au lieu d'un Base64 de {"responses": []}, c'est un cas spécial.
            return {"error": "API returned an empty string for exported responses, possibly indicating no responses or an API issue.", 
                    "details": "Expected Base64 encoded JSON data."}
        try:
            decoded_bytes = base64.b64decode(base64_encoded_data)
            decoded_str = decoded_bytes.decode('utf-8')
            parsed_json = json.loads(decoded_str)
            return parsed_json
        except (base64.binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.error(f"Could not decode/parse Base64 responses for SID {survey_id}: {e}", exc_info=True)
            return {"error": f"Failed to decode/parse Base64 responses: {str(e)}", "base64_result_excerpt": base64_encoded_data[:100]}
        except Exception as e: 
            logger.error(f"Unexpected error during Base64 response decoding for SID {survey_id}: {e}", exc_info=True)
            return {"error": f"Unexpected error decoding Base64 responses: {str(e)}"}
    elif "error" in response_data: 
        return response_data 
    else: 
        logger.warning(f"Unexpected or non-string result from export_responses API for SID {survey_id}. Response: {str(response_data)[:300]}")
        return {"error": "Unexpected or non-string result from export_responses API", "details": response_data}

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(levelname)-8s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    logger.info("--- Starting LimeSurvey API Client Self-Tests ---")
    if not all([LIMESURVEY_API_URL, LIMESURVEY_API_USER, LIMESURVEY_API_PASSWORD is not None]):
        logger.critical("CRITICAL: LIMESURVEY_API_URL, _USER, or _PASSWORD not configured. Tests aborted.")
    else:
        logger.info(f"Attempting to connect to LimeSurvey API at: {LIMESURVEY_API_URL}")
        
        logger.info("--- Test: Listing surveys (implicitly tests get_session_key) ---")
        surveys = list_surveys_api()

        if isinstance(surveys, list) and surveys:
            logger.info(f"Found {len(surveys)} surveys. First survey: {surveys[0]}")
            test_sid = None
            active_sids_with_titles = {s['sid']: s.get('surveyls_title', f"SID {s['sid']}") for s in surveys if s.get('active') == 'Y'}
            
            if not active_sids_with_titles:
                logger.warning("No active surveys found to test export. Using first available survey if any.")
                if surveys:
                    test_sid = surveys[0]['sid']
                    survey_title_for_log = surveys[0].get('surveyls_title', f"SID {test_sid}")
                else: # Pas de sondages du tout
                    logger.error("No surveys found at all.")
                    survey_title_for_log = "N/A" # Pour éviter une erreur plus tard
            else: 
                test_sid = list(active_sids_with_titles.keys())[0]
                survey_title_for_log = active_sids_with_titles[test_sid]
                logger.info(f"Found active surveys. Will use SID {test_sid} ('{survey_title_for_log}') for export test.")

            if test_sid: # Continuer les tests seulement si un SID a été trouvé
                logger.info(f"Using SID {test_sid} ('{survey_title_for_log}') for subsequent detailed tests.")

                logger.info(f"--- Test: Get survey title for SID {test_sid} ---")
                title_props = get_survey_title_api(test_sid)
                if isinstance(title_props, dict) and not title_props.get("error"):
                    logger.info(f"Survey Title (default lang): '{title_props.get('surveyls_title', 'N/A')}'")
                else:
                    logger.warning(f"Could not get survey title or error: {title_props}")

                logger.info(f"--- Test: Listing groups for SID {test_sid} ---")
                groups = list_groups_api(test_sid)
                if isinstance(groups, list) and groups:
                    logger.info(f"Found {len(groups)} groups. First group: {groups[0].get('group_name', 'N/A')}")
                    test_gid = groups[0]['gid']

                    logger.info(f"--- Test: Listing questions for SID {test_sid}, GID {test_gid} ---")
                    questions = list_questions_api(test_sid, test_gid)
                    if isinstance(questions, list) and questions:
                        logger.info(f"Found {len(questions)} questions in GID {test_gid}. First question: {questions[0].get('title', 'N/A')}")
                    elif isinstance(questions, list) and not questions:
                        logger.info(f"No questions found in group GID {test_gid} or group is empty.")
                    else:
                        logger.error(f"Error listing questions: {questions}")
                elif isinstance(groups, list) and not groups:
                    logger.info(f"No groups found for survey SID {test_sid}.")
                else:
                    logger.error(f"Error listing groups: {groups}")

                logger.info(f"--- Test: Exporting responses for SID {test_sid} (document_type='json', heading_type='full') ---")
                responses_export = export_responses_api(survey_id=test_sid, document_type='json', heading_type='full')
                
                if isinstance(responses_export, dict) and "responses" in responses_export: # Succès attendu
                    num_responses = len(responses_export["responses"])
                    logger.info(f"Successfully exported {num_responses} response entries.")
                    if num_responses > 0 and isinstance(responses_export["responses"], list) and responses_export["responses"][0]:
                        first_resp_keys = list(responses_export["responses"][0].keys())[:5]
                        logger.info(f"First response entry keys (first 5): {first_resp_keys}")
                    elif num_responses == 0 :
                        logger.info("Export successful, but no responses were returned (0 entries). This could be normal if the survey is empty.")
                elif isinstance(responses_export, dict) and "error" in responses_export: # Erreur attendue de l'API ou du client
                    logger.error(f"Error exporting responses: {responses_export}") 
                else: # Cas inattendu
                    logger.warning(f"Exported responses format was unexpected: {str(responses_export)[:500]}...")
            else:
                logger.error("No valid SID found to proceed with detailed tests.")
        
        elif isinstance(surveys, dict) and "error" in surveys: 
             logger.error(f"Error listing surveys: {surveys}")
        else:
            logger.info(f"No surveys found or an issue occurred during list_surveys: {surveys}")
        
        logger.info("--- Test: Releasing session key ---")
        release_session_key_api()
    logger.info("--- LimeSurvey API Client Self-Tests Finished ---")