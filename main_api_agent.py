# main_api_agent.py
import os
import json
import logging
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional, Any
import operator

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage # AJOUT DE SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import Tool, tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field 

from langgraph.graph import StateGraph, END

from config.llm_config import get_langchain_llm
from tools.limesurvey_api_client import (
    list_surveys_api,
    get_survey_title_api,
    list_groups_api,
    list_questions_api,
    export_responses_api,
    get_session_key_api,
    release_session_key_api,
    ApiResult
)

load_dotenv()

logger = logging.getLogger(__name__)
logger.info("--- LimeSurvey API Agent Script Starting ---")

try:
    llm = get_langchain_llm(temperature=0.0, max_tokens=3000)
    logger.info(f"LLM (ChatOpenAI) configured with model: {llm.model_name}")
except Exception as e:
    logger.critical(f"Failed to initialize LLM: {e}", exc_info=True)
    raise

class SurveyIdInput(BaseModel):
    survey_id: int = Field(description="The Survey ID (SID) of the survey.")

class GetSurveyTitleInput(SurveyIdInput):
    language: Optional[str] = Field(None, description="Optional language code (e.g., 'en', 'fr') for the survey title.")

class ListQuestionsInput(SurveyIdInput):
    group_id: Optional[int] = Field(None, description="Optional Group ID (GID) to filter questions.")
    language: Optional[str] = Field(None, description="Optional language code for question details.")

class ExportResponsesInput(SurveyIdInput):
    language: Optional[str] = Field(None, description="Language for response export (e.g., 'en', 'fr'). Defaults to base language.")
    completion_status: str = Field("all", description="Filter responses by completion status ('complete', 'incomplete', 'all'). Default 'all'.")
    response_type: str = Field("long", description="Type of responses ('short' for codes, 'long' for full text). Default 'long'.")
    heading_type: str = Field("code", description="Type of headings for response data ('code', 'full', 'abbreviated'). Default 'code'.")

class DataAnalysisServiceInput(BaseModel):
    task_type: str = Field(description="The type of analysis to perform: 'thematic_analysis_of_survey_texts' or 'response_data_analysis'.")
    data: Dict[str, Any] = Field(description="A dictionary containing the data needed for analysis. Structure depends on task_type. For 'response_data_analysis', this can include 'question_code'.")
    original_user_request_context: str = Field(description="The original user query or context for the analysis request.")

def get_analysis_from_data_analyzer_service(
    task_type: str,
    data: Dict[str, Any],
    original_user_request_context: str
) -> str:
    input_dict_for_log = {"task_type": task_type, "data": data, "original_user_request_context": original_user_request_context}
    logger.debug(f"DataAnalyzerAgent Service Logic called with input (excerpt): {str(input_dict_for_log)[:300]}")
    
    try:
        with open("prompts/data_analyzer_prompt.txt", "r", encoding="utf-8") as f:
            data_analyzer_system_prompt_template = f.read()
    except FileNotFoundError:
        logger.error("CRITICAL: prompts/data_analyzer_prompt.txt not found for DataAnalyzer Service.")
        return json.dumps({"error": "DataAnalyzer prompt file not found."})

    if task_type == "response_data_analysis" and "raw_response_data" in data:
        if not isinstance(data["raw_response_data"], str):
            try:
                data["raw_response_data"] = json.dumps(data["raw_response_data"])
            except TypeError as e:
                error_msg_data = f"DataAnalyzer: 'raw_response_data' in input data is not JSON serializable: {e}"
                logger.error(error_msg_data)
                return json.dumps({"error": error_msg_data})
    
    user_message_for_analyzer = f"""
    Original User Request Context: {original_user_request_context}
    Task Type: {task_type}
    Data for analysis:
    {json.dumps(data, indent=2)}

    Please perform the requested analysis based on your system prompt instructions for this task type.
    Ensure your output is a human-readable text summary.
    """

    analyzer_messages = [
        SystemMessage(content=data_analyzer_system_prompt_template), # CORRIGÉ: Utiliser SystemMessage
        HumanMessage(content=user_message_for_analyzer)
    ]

    try:
        analyzer_llm = get_langchain_llm(temperature=0.1, max_tokens=2000) 
        response = analyzer_llm.invoke(analyzer_messages)
        analysis_output = response.content.strip()
        
        if analysis_output.endswith("\nTERMINATE"):
            analysis_output = analysis_output[:-len("\nTERMINATE")].strip()
        if analysis_output.endswith("TERMINATE"):
            analysis_output = analysis_output[:-len("TERMINATE")].strip()
            
        logger.debug(f"DataAnalyzer Service Logic produced: {analysis_output[:300]}...")
        return analysis_output 
    except Exception as e:
        error_msg = f"Error in DataAnalyzer Service LLM call: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({"error": error_msg})

@tool
def list_surveys_tool() -> str:
    """Retrieves a list of all available surveys from LimeSurvey, including their Survey ID (SID), title, and active status. Use this to find the SID of a survey if the user provides its name. This tool is called without arguments. Output is a JSON string list of survey dictionaries."""
    logger.info("Executing list_surveys_tool")
    result = list_surveys_api()
    return json.dumps(result)

@tool(args_schema=GetSurveyTitleInput)
def get_survey_title_tool(survey_id: int, language: Optional[str] = None) -> str:
    """Retrieves the official title and other language-specific properties of a specific survey given its SID. If language is not provided, the survey's base language title is returned. Output: JSON string of survey properties."""
    logger.info(f"Executing get_survey_title_tool for SID: {survey_id}, Lang: {language}")
    result = get_survey_title_api(survey_id=survey_id, language=language)
    return json.dumps(result)

@tool(args_schema=SurveyIdInput)
def list_survey_groups_tool(survey_id: int) -> str:
    """Retrieves the list of question groups for a specific survey SID. Output: JSON string list of group dictionaries."""
    logger.info(f"Executing list_survey_groups_tool for SID: {survey_id}")
    result = list_groups_api(survey_id=survey_id)
    return json.dumps(result)

@tool(args_schema=ListQuestionsInput)
def list_survey_questions_tool(survey_id: int, group_id: Optional[int] = None, language: Optional[str] = None) -> str:
    """Retrieves the list of questions for a specific survey SID. Can be filtered by an optional group_id and/or language. Output: JSON string list of question dictionaries."""
    logger.info(f"Executing list_survey_questions_tool for SID: {survey_id}, GID: {group_id}, Lang: {language}")
    result = list_questions_api(survey_id=survey_id, group_id=group_id, language=language)
    return json.dumps(result)

@tool(args_schema=ExportResponsesInput)
def export_survey_responses_tool(
    survey_id: int,
    language: Optional[str] = None,
    completion_status: str = 'all',
    response_type: str = 'long',
    heading_type: str = 'code'
) -> str:
    """Exports responses for a specific survey SID. Output: JSON string containing the survey responses."""
    logger.info(f"Executing export_survey_responses_tool for SID: {survey_id} (Lang: {language}, Status: {completion_status}, RespType: {response_type}, HeadType: {heading_type})")
    result = export_responses_api(
        survey_id=survey_id, document_type='json', language=language,
        completion_status=completion_status, response_type=response_type, heading_type=heading_type
    )
    return json.dumps(result)

@tool(args_schema=DataAnalysisServiceInput)
def data_analysis_tool(task_type: str, data: Dict[str, Any], original_user_request_context: str) -> str:
    """
    Use this tool for:
    a) Thematic analysis of survey texts (survey title, list of group titles, list of question texts).
       Input for thematic analysis: task_type="thematic_analysis_of_survey_texts", data={"survey_title": "...", "group_titles": [...], "question_texts": [...]}, original_user_request_context="..."
    b) Analysis of specific survey response data (from export_survey_responses_tool).
       Input for response data analysis: task_type="response_data_analysis", data={"raw_response_data": "[JSON string from export_survey_responses_tool]", "survey_title": "...", "question_text_or_context": "...", "question_code": "G01Q01 (optional)", "response_code_mappings": {...} (optional)}, original_user_request_context="..."
    Output: Human-readable text analysis.
    """
    logger.info(f"Executing data_analysis_tool for task_type: {task_type}")
    return get_analysis_from_data_analyzer_service(
        task_type=task_type,
        data=data,
        original_user_request_context=original_user_request_context
    )

tools = [
    list_surveys_tool,
    get_survey_title_tool,
    list_survey_groups_tool,
    list_survey_questions_tool,
    export_survey_responses_tool,
    data_analysis_tool
]
logger.info(f"LangChain tools defined: {[t.name for t in tools]}")

try:
    with open("prompts/api_main_agent_prompt.txt", "r", encoding="utf-8") as f:
        api_main_agent_system_prompt = f.read()
    logger.info("Successfully loaded api_main_agent_prompt.txt")
except FileNotFoundError:
    logger.error("CRITICAL: prompts/api_main_agent_prompt.txt not found. Using fallback system prompt.")
    api_main_agent_system_prompt = "You are a helpful assistant. Use your tools to answer questions about LimeSurvey."

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", api_main_agent_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
logger.debug("Agent prompt template created.")

langchain_agent_runnable = create_openai_tools_agent(llm, tools, prompt_template)
logger.info("OpenAI Tools Agent runnable created.")

agent_executor = AgentExecutor(
    agent=langchain_agent_runnable,
    tools=tools,
    verbose=True, 
    max_iterations=10, 
    handle_parsing_errors=True,
    return_intermediate_steps=False 
)
logger.info(f"AgentExecutor created (max_iterations={agent_executor.max_iterations}).")

class MainAgentGraphState(TypedDict):
    user_query: str 
    messages: Annotated[Sequence[BaseMessage], operator.add] 
    final_output: Optional[str] 

workflow = StateGraph(MainAgentGraphState)

def agent_executor_node(state: MainAgentGraphState):
    logger.info(f"--- AgentExecutor Node Invoked --- Query: '{state['user_query']}'")
    agent_input_payload = {"messages": state["messages"]}
    
    if state["messages"] and logger.isEnabledFor(logging.DEBUG):
        history_preview = []
        for m in state['messages'][-5:]: 
            content_preview = str(m.content)[:70] + "..." if len(str(m.content)) > 70 else str(m.content)
            if isinstance(m, ToolMessage):
                history_preview.append(f"({m.type} ID: {m.tool_call_id}) Name: {m.name} | Content: {content_preview}")
            elif isinstance(m, AIMessage) and m.tool_calls:
                tool_call_names = [tc.get('name', 'N/A') for tc in m.tool_calls if isinstance(tc, dict)] 
                history_preview.append(f"({m.type}) Calls: {tool_call_names} | Content: {content_preview}")
            else:
                history_preview.append(f"({m.type}) {content_preview}")
        logger.debug(f"AgentExecutor input messages (last 5 preview): {history_preview}")

    response_dict = agent_executor.invoke(agent_input_payload)
    final_agent_response_content = response_dict.get("output", "AgentExecutor did not provide an 'output' field.")
    logger.info(f"AgentExecutor final output for this invocation (excerpt): {final_agent_response_content[:200]}...")
    return {"final_output": final_agent_response_content}

workflow.add_node("main_agent_node", agent_executor_node)
workflow.set_entry_point("main_agent_node")
workflow.add_edge("main_agent_node", END)

app = workflow.compile() # 'app' est ici votre graphe LangGraph compilé
logger.info("LangGraph (simple executor shell) workflow compiled.")

try:
    from PIL import Image
    img_bytes = app.get_graph().draw_mermaid_png()
    if img_bytes:
        with open("langgraph_main_api_agent_flow.png", "wb") as f:
            f.write(img_bytes)
        logger.info("Graph visualization saved to langgraph_main_api_agent_flow.png")
except ImportError:
    logger.warning("Pillow (PIL) not installed, skipping graph visualization generation.")
except Exception as e:
    logger.warning(f"Could not generate graph visualization: {e}. Ensure graphviz and related dependencies are installed.")

def message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    msg_dict: Dict[str, Any] = {"type": message.type, "content": str(message.content)}
    if isinstance(message, AIMessage):
        if message.tool_calls:
            serializable_tool_calls = []
            for tc in message.tool_calls:
                if isinstance(tc, dict): 
                    serializable_tool_calls.append(tc)
                else: 
                    serializable_tool_calls.append({"id": getattr(tc, 'id', None), "name": getattr(tc, 'name', None), "args": getattr(tc, 'args', {})})
            msg_dict["tool_calls"] = serializable_tool_calls
        if message.invalid_tool_calls:
             msg_dict["invalid_tool_calls"] = message.invalid_tool_calls
        if message.additional_kwargs:
            msg_dict["additional_kwargs"] = message.additional_kwargs
        if message.response_metadata:
            msg_dict["response_metadata"] = message.response_metadata
    elif isinstance(message, ToolMessage):
        msg_dict["tool_call_id"] = message.tool_call_id
        if hasattr(message, 'name'): 
            msg_dict["name"] = message.name 
    return msg_dict

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)-8s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger.info("--- Main API Agent Batch Tests Starting ---")
    
    test_queries = [
        {
            "id": "Test_01_ListAllSurveys",
            "query": "Can you list all the surveys available in the system?"
        },
        {
            "id": "Test_02_Themes_Inomax",
            "query": "Tell me about the main themes covered in the survey named 'Inomax'. Please also list its official title."
        },
        {
            "id": "Test_03_MarketGrowthAnalysis_Inomax",
            "query": "Provide an analysis of the responses to the question 'We feel that our main markets are growing' in the Inomax survey."
        },
        {
            "id": "Test_04_NonExistentSurvey",
            "query": "What are the conclusions for the 'TotallyNonExistentSurveyFooBar' survey?"
        }
    ]
        
    all_tests_results = []
    
    session_key_obtained_for_batch = get_session_key_api()
    if not session_key_obtained_for_batch:
        logger.critical("CRITICAL: Could not get LimeSurvey API session key for the batch. Aborting all tests.")
    else:
        logger.info(f"LimeSurvey API session key obtained for batch run: {session_key_obtained_for_batch[:10]}...")
        for test_case in test_queries:
            logger.info(f"\n\n{'='*30} EXECUTING TEST: {test_case['id']} {'='*30}")
            initial_messages_for_test: List[BaseMessage] = [HumanMessage(content=test_case["query"])]
            logger.info(f"QUERY: \"{test_case['query']}\"")
            final_answer = "Error: Default error message, test execution problem."
            logged_messages_for_this_test = initial_messages_for_test.copy()

            try:
                graph_initial_state = MainAgentGraphState(
                    user_query=test_case["query"],
                    messages=initial_messages_for_test,
                    final_output=None
                )
                final_graph_state_result = app.invoke(graph_initial_state, config={"recursion_limit": 15}) # Augmentation de la limite de récursion pour le graphe
                final_answer = final_graph_state_result.get("final_output", "Agent did not provide a final_output in graph state.")
                logged_messages_for_this_test.append(AIMessage(content=final_answer))
                logger.info(f"Agent execution finished for test {test_case['id']}.")
            except Exception as e:
                logger.error(f"Error during Agent/Graph invocation for test {test_case['id']}: {str(e)}", exc_info=True)
                final_answer = f"An error occurred: {str(e)}"
                logged_messages_for_this_test.append(AIMessage(content=final_answer))

            serializable_messages_for_log = [message_to_dict(m) for m in logged_messages_for_this_test]
            all_tests_results.append({
                "test_id": test_case['id'],
                "query": test_case['query'],
                "final_answer_produced": final_answer,
                "conversation_history_for_log": serializable_messages_for_log
            })
            logger.info(f"\n{'-'*20} FINAL ANSWER FOR {test_case['id']} {'-'*20}\n{final_answer}\n{'-'* (40 + len(test_case['id']))}")
            logger.info(f"--- Test {test_case['id']} completed ---")

        logger.info("Releasing LimeSurvey API session key after all tests.")
        release_session_key_api()

    master_log_file_name = "main_api_agent_batch_test_results.json"
    try:
        with open(master_log_file_name, "w", encoding="utf-8") as f:
            json.dump(all_tests_results, f, indent=2, ensure_ascii=False)
        logger.info(f"All test results saved to: {master_log_file_name}")
    except Exception as e:
        logger.error(f"Error saving master log file: {e}", exc_info=True)
    logger.info("--- Main API Agent Batch Tests Completed ---")