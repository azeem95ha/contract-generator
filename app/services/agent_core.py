# contract_api/app/services/agent_core.py
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import json
import requests # Keep if any tool directly uses it, though Tavily is primary for web
# from bs4 import BeautifulSoup # Keep if any tool directly uses it
from urllib.parse import quote_plus
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import io
import uuid
from pathlib import Path
from tavily import TavilyClient
from typing import List, Literal, Optional, Dict, Any

from flask import current_app
from .utils import extract_text_from_pdf_path, chunk_text

# --- Global variables for agent components ---
model = None
client_chroma = None
collection = None
tavily_client = None
agent_executor = None
tools_list = []
tools_dict = {}
_agent_initialized = False

class ContractState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.contract_type: Optional[str] = None
        self.contract_context: Optional[str] = None
        self.template_research: Optional[str] = None
        self.rag_research: Optional[str] = None
        self.required_information: dict = {}
        self.information_complete: bool = False
        self.contract_sections: Optional[str] = None
        self.contract_draft: Optional[str] = None
        self.reviewer_notes: Optional[str] = None
        self.final_draft: Optional[str] = None
        self.current_stage: str = "contract_type_identification"

contract_state = ContractState()

# --- Tool Definitions ---

@tool
def add_contract_template(pdf_file_path: str, contract_type: str, description: str = "") -> str:
    """
    Add a PDF contract template to the knowledge base for RAG retrieval.
    Args:
        pdf_file_path: Path to the PDF contract template file (temporarily saved on server).
        contract_type: Type of contract (e.g., "employment", "service agreement").
        description: Optional description of the contract template.
    Returns:
        Confirmation message about the template being added.
    """
    global collection
    if collection is None:
        return "Error: ChromaDB collection not initialized."
    try:
        if not os.path.exists(pdf_file_path):
            return f"Error: PDF file not found at {pdf_file_path}"

        text_content = extract_text_from_pdf_path(pdf_file_path)

        if not text_content or not text_content.strip():
            return "Error: No text content could be extracted from the PDF"

        chunks = chunk_text(text_content)
        if not chunks:
            return "Error: Text content was empty after chunking."

        filename = Path(pdf_file_path).name
        base_metadata = {
            "source": filename,
            "contract_type": contract_type.lower(),
            "description": description,
            "total_chunks": len(chunks)
        }

        chunk_ids, chunk_texts, chunk_metadatas = [], [], []
        for i, chunk_content in enumerate(chunks):
            if not chunk_content or not chunk_content.strip():
                continue
            chunk_id = f"{filename}_{contract_type}_{uuid.uuid4().hex[:8]}_{i}"
            chunk_metadata = {**base_metadata, "chunk_index": i, "chunk_id": chunk_id}

            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk_content)
            chunk_metadatas.append(chunk_metadata)

        if not chunk_texts:
             return "Error: No valid text chunks to add to the database."

        collection.add(documents=chunk_texts, ids=chunk_ids, metadatas=chunk_metadatas)
        return f"Successfully added {len(chunk_texts)} chunks from '{filename}' of type '{contract_type}' to the knowledge base."
    except Exception as e:
        current_app.logger.error(f"Error in add_contract_template: {e}", exc_info=True)
        return f"Error adding contract template: {str(e)}"

@tool
def search_contract_knowledge_base(query: str, contract_type: Optional[str] = None, top_k: int = 5) -> str:
    """
    Search the contract template knowledge base using RAG for relevant information.
    Args:
        query: Search query describing what information is needed.
        contract_type: Optional filter by contract type.
        top_k: Number of relevant chunks to retrieve.
    Returns:
        Relevant contract template information and examples.
    """
    global collection
    if collection is None:
        return "Error: ChromaDB collection not initialized."
    try:
        search_query_text = query
        if contract_type:
            search_query_text = f"{contract_type} contract {query}"

        where_clause = {}
        if contract_type:
            where_clause["contract_type"] = contract_type.lower()

        results = collection.query(
            query_texts=[search_query_text],
            n_results=top_k,
            where=where_clause if where_clause else None # Pass None if empty for ChromaDB
        )

        if not results or not results.get('documents') or not results['documents'][0]:
            return f"No relevant contract templates found for query: '{query}' with type: '{contract_type or 'any'}'. Try a broader search or add relevant templates."

        rag_output = f"RAG KNOWLEDGE BASE SEARCH RESULTS for '{query}':\n\n"
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results.get('metadatas') and results['metadatas'] else [{}] * len(documents)
        distances = results['distances'][0] if results.get('distances') and results['distances'] else [0.0] * len(documents)

        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
            rag_output += f"Result {i} (Relevance Score: {1-dist:.3f}):\n"
            rag_output += f"  Source File: {meta.get('source', 'N/A')}\n"
            rag_output += f"  Contract Type (in DB): {meta.get('contract_type', 'N/A')}\n"
            rag_output += f"  Description: {meta.get('description', 'N/A')}\n"
            rag_output += f"  Content Snippet:\n---\n{doc.strip()}\n---\n\n"
        return rag_output
    except Exception as e:
        current_app.logger.error(f"Error in search_contract_knowledge_base: {e}", exc_info=True)
        return f"Error searching contract knowledge base: {str(e)}"

@tool
def list_contract_templates() -> str:
    """
    List all contract templates currently in the knowledge base, summarized by source and type.
    Returns:
        Summary of available contract templates and chunk counts.
    """
    global collection
    if collection is None:
        return "Error: ChromaDB collection not initialized."
    try:
        all_db_entries = collection.get(include=["metadatas"])
        if not all_db_entries or not all_db_entries['ids']:
            return "No contract templates found in the knowledge base."

        templates_summary = {}
        for metadata_item in all_db_entries['metadatas']:
            source_file = metadata_item.get('source', 'Unknown Source')
            doc_contract_type = metadata_item.get('contract_type', 'Uncategorized')
            doc_description = metadata_item.get('description', 'No description provided.')
            
            template_key = f"Source: {source_file} (Type: {doc_contract_type})"
            if template_key not in templates_summary:
                templates_summary[template_key] = {'count': 0, 'description': doc_description}
            templates_summary[template_key]['count'] += 1
            
        if not templates_summary:
            return "No contract templates with identifiable metadata found."

        output_str = "Available Contract Templates (Grouped by Source & Type):\n\n"
        for idx, (name, data) in enumerate(templates_summary.items(), 1):
            output_str += f"{idx}. {name}\n"
            output_str += f"   Description: {data['description']}\n"
            output_str += f"   Number of Chunks: {data['count']}\n\n"
        
        output_str += f"Total unique template groups: {len(templates_summary)}\n"
        output_str += f"Total chunks in database: {collection.count()}"
        return output_str
    except Exception as e:
        current_app.logger.error(f"Error in list_contract_templates: {e}", exc_info=True)
        return f"Error listing contract templates: {str(e)}"

@tool
def identify_contract_type(user_input: str) -> str:
    """
    Analyze user input to identify and clarify the contract type needed. Sets the contract type in the state if identified.
    Args:
        user_input: User's description of what they need.
    Returns:
        Response asking for clarification or confirming the contract type and readiness for next steps.
    """
    global model, contract_state
    if model is None: return "Error: LLM model not initialized."

    prompt_text = f"""
    Analyze the following user input to determine if they have clearly specified a contract type:
    User Input: "{user_input}"
    Common contract types: employment contract, service agreement, lease agreement, purchase agreement, non-disclosure agreement (NDA), consulting agreement, freelance contract, partnership agreement, licensing agreement, etc.

    If the contract type is clear and specific:
    - Confirm the contract type. Example: "Okay, you need a [Identified Contract Type]."
    - Indicate readiness to proceed with template research for this type.

    If the contract type is unclear or too vague:
    - Ask specific clarifying questions. Example: "Could you please specify what kind of agreement you're looking for?"
    - You can provide examples of contract types if helpful.

    Be helpful and guide the user to provide a clear, specific contract type.
    """
    llm_response = model.invoke([HumanMessage(content=prompt_text)])

    type_determination_prompt = f"""
    Based on the user's input and the initial analysis, can you identify a specific contract type name?
    User input: "{user_input}"
    Initial LLM Analysis: "{llm_response.content}"

    If a clear contract type (e.g., "employment contract", "NDA", "service agreement") can be confidently identified from the analysis, respond with ONLY the contract type name.
    If it's still not clear enough or ambiguous, respond with "UNCLEAR".
    """
    type_result_message = model.invoke([HumanMessage(content=type_determination_prompt)])
    identified_type_str = type_result_message.content.strip()

    if "UNCLEAR" not in identified_type_str.upper() and identified_type_str:
        contract_state.contract_type = identified_type_str
        contract_state.current_stage = "template_research"
        return f"{llm_response.content}\n\nGreat! I've identified that you need a {contract_state.contract_type}. I will now search for official templates and requirements for this, and also check my knowledge base for relevant contract examples."
    else:
        return llm_response.content

@tool
def search_contract_templates(contract_type: Optional[str] = None) -> str:
    """
    Search web (using Tavily) and the RAG knowledge base for official contract templates, clauses, and requirements.
    This tool should only be used after a contract type has been identified.
    Args:
        contract_type: The type of contract to search for. If None, uses the type stored in the contract_state.
    Returns:
        A combined summary of research findings from both web search and the RAG knowledge base.
    """
    global model, contract_state, tavily_client
    if model is None: return "Error: LLM model not initialized."
    if tavily_client is None: current_app.logger.warning("Tavily client not initialized; web search will be skipped.")

    effective_contract_type = contract_type or contract_state.contract_type
    if not effective_contract_type:
        return "Error: Contract type must be identified before searching for templates. Please use 'identify_contract_type' first or provide a contract_type argument."

    # if contract_state.current_stage not in ["template_research", "type_identified", "information_collection"]: # Allow re-searching
    #     return f"Warning: Template research is usually done after contract type identification. Current stage: {contract_state.current_stage}. Proceeding with search for '{effective_contract_type}'."

    web_research_summary = "WEB SEARCH (TAVILY):\n"
    if tavily_client:
        try:
            tavily_queries = [
                f"official {effective_contract_type} template legal requirements",
                f"standard clauses for {effective_contract_type}",
                f"government guidelines {effective_contract_type} sample"
            ]
            all_tavily_results_content = []
            for t_query in tavily_queries:
                tavily_response = tavily_client.search(query=t_query, search_depth="advanced", max_results=3) # Using advanced for potentially better snippets
                if tavily_response and 'results' in tavily_response:
                    for res_item in tavily_response['results']:
                        all_tavily_results_content.append(f"Title: {res_item.get('title', 'N/A')}\nURL: {res_item.get('url', 'N/A')}\nContent: {res_item.get('content', 'N/A')[:1000]}...\n---\n") # Truncate content

            if all_tavily_results_content:
                web_research_summary += "".join(all_tavily_results_content)
                # Optional: Summarize further with LLM if too long
                # summarize_prompt = f"Summarize the key findings about {effective_contract_type} from these web search results:\n{''.join(all_tavily_results_content)}"
                # web_research_summary = model.invoke([HumanMessage(content=summarize_prompt)]).content

            else:
                web_research_summary += "No relevant results found from web search.\n"
        except Exception as e:
            web_research_summary += f"Error during Tavily web search: {str(e)}\n"
            current_app.logger.error(f"Tavily search error for {effective_contract_type}: {e}", exc_info=True)
    else:
        web_research_summary += "Tavily client not available for web search.\n"

    rag_research_summary = f"RAG KNOWLEDGE BASE SEARCH for '{effective_contract_type}':\n"
    try:
        rag_query = f"clauses and sections for {effective_contract_type}"
        rag_search_result = search_contract_knowledge_base.func(
            query=rag_query,
            contract_type=effective_contract_type,
            top_k=5 # Get more RAG results
        )
        rag_research_summary += rag_search_result
    except Exception as e:
        rag_research_summary += f"Error searching knowledge base: {str(e)}\n"
        current_app.logger.error(f"RAG search error during template search for {effective_contract_type}: {e}", exc_info=True)

    contract_state.template_research = web_research_summary # Store the detailed findings
    contract_state.rag_research = rag_research_summary     # Store the detailed RAG findings
    contract_state.current_stage = "information_collection"

    final_output_summary = (
        f"COMPREHENSIVE TEMPLATE AND REQUIREMENTS RESEARCH FOR: {effective_contract_type.upper()}\n\n"
        f"== Web Search Summary ==\n{web_research_summary}\n\n"
        f"== Knowledge Base (RAG) Summary ==\n{rag_research_summary}\n\n"
        "This research will be used to guide information collection and contract drafting. "
        "I am now ready to collect the specific details needed for your contract."
    )
    return final_output_summary

@tool
def collect_contract_information(user_input: str, contract_type: Optional[str] = None) -> str:
    """
    Collect specific information needed for the contract from the user, based on the identified contract type.
    It updates the internal state with collected information and asks for more if needed, or confirms completion.
    Args:
        user_input: User's response containing some of the requested information.
        contract_type: The type of contract. If None, uses the type from contract_state.
    Returns:
        A response asking for more information or confirming that information collection is complete.
    """
    global model, contract_state
    if model is None: return "Error: LLM model not initialized."

    effective_contract_type = contract_type or contract_state.contract_type
    if not effective_contract_type:
        return "Error: Contract type is not set. Please identify the contract type first."

    # Predefined required information templates for different contract types
    required_info_map = {
        "employment contract": ["employer name and registered address", "employee full name and address", "job title and primary responsibilities", "employment start date", "probationary period (if any)", "salary or wage amount and payment frequency", "working hours and schedule", "annual leave entitlement", "notice period for termination", "confidentiality clause details (if any specific)", "governing law/jurisdiction"],
        "service agreement": ["service provider name and address", "client name and address", "detailed description of services to be provided", "start date and end date of services (or project duration)", "payment terms (e.g., hourly rate, fixed fee, payment schedule)", "deliverables and acceptance criteria", "confidentiality obligations", "intellectual property ownership for work product", "termination conditions and notice period", "liability limitations", "governing law"],
        "lease agreement": ["landlord full name and address", "tenant full name(s) and address(es)", "full address of the leased property", "description of the property (e.g., type, size)", "lease term (start and end dates)", "rent amount and payment due date/method", "security deposit amount and conditions for return", "responsibilities for utilities (e.g., electricity, water, gas)", "maintenance and repair responsibilities (landlord vs. tenant)", "rules and regulations (e.g., pets, smoking, alterations)", "conditions for lease termination or renewal"],
        "non-disclosure agreement (nda)": ["disclosing party name and address", "receiving party name and address", "definition of confidential information", "purpose of disclosure", "obligations of the receiving party (e.g., non-use, non-disclosure, security measures)", "exclusions from confidential information (e.g., publicly known, independently developed)", "term of the agreement (duration of confidentiality)", "return or destruction of confidential information upon termination", "governing law"],
        # Add more contract types and their specific required fields
        "default": ["names and addresses of all parties involved", "clear description of the subject matter of the contract", "key obligations of each party", "payment amounts, terms, and schedule (if applicable)", "duration or term of the contract", "conditions for termination", "governing law/jurisdiction"]
    }
    specific_required_info = required_info_map.get(effective_contract_type.lower(), required_info_map["default"])

    if user_input and user_input.strip():
        extraction_guidance_prompt = f"""
        You are an expert at extracting structured information from user text for a '{effective_contract_type}'.
        The user has provided the following input: "{user_input}"

        The information categories we are trying to fill are: {specific_required_info}
        The information already collected is: {contract_state.required_information}

        Carefully review the user's input and extract any new or updated information that fits into these categories.
        Present the extracted information as a JSON object where keys are the category names (or similar, if the user uses different phrasing for a known category) and values are the extracted details.
        If the user's input doesn't provide information for any of these specific categories, or if it's too vague for a specific category, return an empty JSON object {{}}.
        Be precise. For example, if a category is "employer name and registered address", try to get both.
        """
        try:
            extraction_response = model.invoke([HumanMessage(content=extraction_guidance_prompt)])
            extracted_content = extraction_response.content
            json_match = json_match = extracted_content[extracted_content.find('{'):extracted_content.rfind('}')+1] # Basic JSON extraction
            if json_match:
                newly_extracted_info = json.loads(json_match)
                for key, value in newly_extracted_info.items():
                    if value: # Only update if value is not empty
                        contract_state.required_information[key.lower().replace(" ", "_")] = value # Normalize key
            else: # Fallback for non-JSON or if LLM couldn't structure it
                contract_state.required_information[f"unstructured_input_{len(contract_state.required_information)}"] = user_input
        except (json.JSONDecodeError, Exception) as e:
            current_app.logger.error(f"Error parsing LLM response for info extraction: {e}, content: {extracted_content}")
            contract_state.required_information[f"extraction_error_input_{len(contract_state.required_information)}"] = user_input

    missing_info = [info_item for info_item in specific_required_info if info_item.lower().replace(" ", "_") not in contract_state.required_information or not contract_state.required_information[info_item.lower().replace(" ", "_")]]

    if not missing_info:
        contract_state.information_complete = True
        contract_state.current_stage = "section_planning"
        collected_summary = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k,v in contract_state.required_information.items()])
        return f"COMPLETE: All required information for the {effective_contract_type} seems to be collected.\nSummary:\n{collected_summary}\n\nI will now proceed to plan the contract sections."
    else:
        # Ask for a few missing items at a time
        questions_to_ask = missing_info[:2] # Ask for 1 or 2 items
        question_string = f"For the {effective_contract_type}, I still need some information. Could you please provide:\n"
        for item in questions_to_ask:
            question_string += f"- {item.title()}?\n"
        if len(missing_info) > 2:
            question_string += f"\n(There are {len(missing_info) - 2} more items after these.)"
        
        current_info_summary = "\nInformation collected so far:\n" + "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k,v in contract_state.required_information.items()]) if contract_state.required_information else "\nNo information collected yet."
        return question_string + current_info_summary

@tool
def plan_contract_sections(context: Optional[str] = None, template_research: Optional[str] = None, rag_research: Optional[str] = None) -> str:
    """
    Plan and outline the sections that should be included in the contract.
    This uses the contract type, collected user information, web research, and RAG knowledge base findings.
    This tool should only be used after information collection is complete.
    Args:
        context: General context about the contract (usually set by `investigate_contract_requirements` or inferred).
        template_research: Summary of web search findings for templates.
        rag_research: Summary of RAG knowledge base findings.
    Returns:
        A structured outline of contract sections with brief descriptions for each.
    """
    global model, contract_state
    if model is None: return "Error: LLM model not initialized."

    if not contract_state.information_complete:
        return "Error: Cannot plan contract sections until all required information has been collected. Please complete the information gathering step."
    if not contract_state.contract_type:
        return "Error: Contract type is not set. Cannot plan sections."

    used_context = context or contract_state.contract_context or f"General context for a {contract_state.contract_type}."
    used_template_research = template_research or contract_state.template_research or "No specific web template research available."
    used_rag_research = rag_research or contract_state.rag_research or "No specific RAG knowledge base research available."

    prompt_for_section_planning = f"""
    You are a legal assistant tasked with creating a detailed section outline for a '{contract_state.contract_type}'.
    All necessary information has been collected from the user:
    Collected User Information: {json.dumps(contract_state.required_information, indent=2)}

    Additionally, research has been conducted:
    Web Template Research Summary:
    ---
    {used_template_research}
    ---

    RAG Knowledge Base Examples Summary:
    ---
    {used_rag_research}
    ---

    General Contract Context:
    ---
    {used_context}
    ---

    Based on ALL the above (user information, web research, RAG examples, and general context), create a comprehensive and logically ordered list of sections for this '{contract_state.contract_type}'.
    For each section, provide:
    1. The section title (e.g., "1. Definitions", "2. Scope of Services").
    2. A brief description of what this section should cover, incorporating relevant details from the user's information and research findings.
    3. Mention any key clauses or sub-points that should be included within that section, especially if highlighted in the research or RAG examples.

    Note that the sources hirarchy is as follows: rag_research --> template_research -> user_information
    The outline should be well-structured, professional, and serve as a clear blueprint for drafting the full contract.
    Prioritize standard legal structure and common practices observed in the research.
    Ensure the planned sections will allow for the inclusion of all collected user information.
    """
    section_plan_response = model.invoke([HumanMessage(content=prompt_for_section_planning)])
    contract_state.contract_sections = section_plan_response.content
    contract_state.current_stage = "drafting"
    return f"Contract sections have been planned:\n\n{contract_state.contract_sections}\n\nI am now ready to draft the contract based on this plan."

@tool
def draft_contract(sections: Optional[str] = None, context: Optional[str] = None) -> str:
    """
    Generate a complete contract draft based on the planned sections, collected information, research, and knowledge base examples.
    This tool should only be used after contract sections have been planned.
    Args:
        sections: The planned contract sections outline. If None, uses the outline from contract_state.
        context: General context about the contract. If None, uses context from contract_state.
    Returns:
        The full text of the drafted contract.
    """
    global model, contract_state
    if model is None: return "Error: LLM model not initialized."

    if not contract_state.contract_sections:
        return "Error: Cannot draft contract without a planned section outline. Please use 'plan_contract_sections' first."
    if not contract_state.contract_type or not contract_state.information_complete:
        return "Error: Contract type not set or information collection is incomplete. Cannot draft."

    used_sections = sections or contract_state.contract_sections
    used_context = context or contract_state.contract_context or f"Drafting a {contract_state.contract_type}."
    used_template_research = contract_state.template_research or "No specific web template research available for drafting reference."
    used_rag_research = contract_state.rag_research or "No specific RAG knowledge base research available for drafting reference."

    prompt_for_drafting = f"""
    You are a highly skilled legal drafter. Your task is to generate a complete, professional contract draft for a '{contract_state.contract_type}'.

    You have the following resources:
    1.  Detailed Section Plan:
        ---
        {used_sections}
        ---
    2.  Collected User Information (to be incorporated into the draft):
        ---
        {json.dumps(contract_state.required_information, indent=2)}
        ---
    3.  Reference: Web Template Research Summary:
        ---
        {used_template_research}
        ---
    4.  Reference: RAG Knowledge Base Examples Summary (contains examples of clauses and language):
        ---
        {used_rag_research}
        ---
    5.  General Contract Context:
        ---
        {used_context}
        ---

    Instructions for Drafting:
    -   Follow the `Detailed Section Plan` meticulously. Each planned section must be addressed.
    -   Incorporate ALL `Collected User Information` into the appropriate sections of the contract. If information for a placeholder in the section plan is available from user info, use it.
    -   Use professional, clear, and precise legal language.
    -   Refer to the `Web Template Research Summary` and `RAG Knowledge Base Examples Summary` for standard clauses, terminology, and structural patterns relevant to a '{contract_state.contract_type}'. Emulate the style and completeness found in these examples where appropriate.
    -   Ensure the contract is internally consistent and covers all essential aspects typically found in such an agreement.
    -   Include standard boilerplate clauses if appropriate for this type of contract (e.g., Entire Agreement, Severability, Notices, Governing Law, Dispute Resolution), referencing the research for common practices.
    -   Format the contract clearly with numbered sections and sub-sections.
    -   Conclude with appropriate signature blocks for all parties involved (use placeholders like "[Disclosing Party Name]" or "[Employee Name]" if the actual names are part of the collected info, otherwise generic placeholders).

    Produce the full text of the contract draft. Do not include any commentary outside of the contract text itself.
    """
    draft_response = model.invoke([HumanMessage(content=prompt_for_drafting)])
    contract_state.contract_draft = draft_response.content
    contract_state.current_stage = "reviewing"
    return f"The contract draft has been generated:\n\n---\n{contract_state.contract_draft}\n---\n\nThis draft is now ready for review."

@tool
def review_contract(draft: Optional[str] = None) -> str:
    """
    Review the provided contract draft and offer detailed feedback, suggestions for improvement, and identify potential issues.
    Args:
        draft: The contract draft text to be reviewed. If None, uses the draft from contract_state.
    Returns:
        A comprehensive review of the contract draft.
    """
    global model, contract_state
    if model is None: return "Error: LLM model not initialized."

    draft_to_review = draft or contract_state.contract_draft
    if not draft_to_review:
        return "Error: No contract draft available to review. Please draft the contract first."

    prompt_for_review = f"""
    You are an experienced legal reviewer. Please critically review the following contract draft for a '{contract_state.contract_type}':

    Contract Draft:
    ---
    {draft_to_review}
    ---

    Collected User Information (for context on what was intended):
    ---
    {json.dumps(contract_state.required_information, indent=2)}
    ---
    Reference: Web Template Research Summary (for common standards):
    ---
    {contract_state.template_research or "N/A"}
    ---
    Reference: RAG Knowledge Base Examples Summary (for example clauses/structures):
    ---
    {contract_state.rag_research or "N/A"}
    ---

    Please provide a detailed review focusing on:
    1.  Completeness: Are there any essential clauses or terms missing for a typical '{contract_state.contract_type}', considering the user's information and research?
    2.  Clarity and Ambiguity: Are there any sections or phrases that are unclear, ambiguous, or could lead to misinterpretation?
    3.  Accuracy: Does the draft accurately reflect the collected user information? Are there any inconsistencies?
    4.  Potential Risks or Gaps: Are there any obvious legal risks, omissions, or areas that might be problematic for any party?
    5.  Language and Tone: Is the legal language professional, precise, and appropriate?
    6.  Suggestions for Improvement: Provide specific, actionable suggestions for rephrasing, adding, or removing content.
    7.  Overall Assessment: A brief summary of the draft's quality.

    Structure your review clearly, perhaps by section of the contract or by category of feedback.
    """
    review_notes_response = model.invoke([HumanMessage(content=prompt_for_review)])
    contract_state.reviewer_notes = review_notes_response.content
    # contract_state.current_stage remains "reviewing" or could move to "awaiting_finalization_input"
    return f"Contract review complete. Here are the notes:\n\n{contract_state.reviewer_notes}"

@tool
def finalize_contract(draft: Optional[str] = None, review_notes: Optional[str] = None, user_feedback_on_review: Optional[str] = None) -> str:
    """
    Create the final version of the contract by incorporating reviewer notes and any additional user feedback into the draft.
    Args:
        draft: The contract draft to be finalized. If None, uses the draft from contract_state.
        review_notes: The feedback from the review process. If None, uses notes from contract_state.
        user_feedback_on_review: Any additional specific instructions or changes requested by the user after seeing the review.
    Returns:
        The final, polished contract text. Also saves it to a file.
    """
    global model, contract_state
    if model is None: return "Error: LLM model not initialized."

    original_draft = draft or contract_state.contract_draft
    if not original_draft:
        return "Error: No contract draft available to finalize."

    used_review_notes = review_notes or contract_state.reviewer_notes or "No specific review notes provided. Finalize based on best judgment."
    additional_user_instructions = user_feedback_on_review or "No additional user feedback on review provided."

    prompt_for_finalization = f"""
    You are tasked with producing the FINAL version of a '{contract_state.contract_type}'.
    You have the following materials:

    1.  Original Contract Draft:
        ---
        {original_draft}
        ---
    2.  Reviewer's Notes and Suggestions:
        ---
        {used_review_notes}
        ---
    3.  Additional User Feedback/Instructions (after reviewing the draft and the reviewer's notes):
        ---
        {additional_user_instructions}
        ---
    4.  Original Collected User Information (for reference if needed to clarify intent):
        ---
        {json.dumps(contract_state.required_information, indent=2)}
        ---

    Your goal is to:
    -   Carefully consider all points raised in the `Reviewer's Notes` and `Additional User Feedback`.
    -   Intelligently incorporate necessary revisions, additions, or deletions into the `Original Contract Draft` to address the feedback.
    -   Resolve any inconsistencies or ambiguities highlighted.
    -   Ensure the final contract is legally sound (to the best of your AI ability), clear, complete, and accurately reflects the user's intent as per the `Original Collected User Information`.
    -   Maintain professional legal language and formatting.
    -   If feedback is conflicting, use your best judgment to produce the most reasonable outcome or highlight the conflict if irresolvable.

    Produce only the full text of the polished, final contract. Do not include any prefatory remarks or your own commentary on the process.
    """
    final_draft_response = model.invoke([HumanMessage(content=prompt_for_finalization)])
    contract_state.final_draft = final_draft_response.content
    contract_state.current_stage = "finalized"

    # Save the final draft to a file
    # Ensure UPLOAD_FOLDER is correctly configured and accessible
    upload_folder = current_app.config.get('UPLOAD_FOLDER', './data/uploads') # Fallback
    final_contract_filename = f"final_{contract_state.contract_type.replace(' ', '_')}_{uuid.uuid4().hex[:6]}.md"
    final_file_path = os.path.join(upload_folder, final_contract_filename)
    
    try:
        os.makedirs(upload_folder, exist_ok=True)
        with open(final_file_path, "w", encoding="utf-8") as f:
            f.write(contract_state.final_draft)
        save_message = f"Final contract generated and saved to '{final_file_path}'."
    except Exception as e:
        current_app.logger.error(f"Error saving final contract to {final_file_path}: {e}", exc_info=True)
        save_message = f"Final contract generated, but failed to save to file: {str(e)}."

    return f"{save_message}\n\nFinal Contract Text:\n---\n{contract_state.final_draft}\n---"

@tool
def get_contract_status() -> dict:
    """
    Get the current status of the contract generation process, including current stage and availability of artifacts.
    Returns:
        A dictionary summarizing the current progress and state.
    """
    global contract_state
    status_dict = {
        "current_stage": contract_state.current_stage,
        "contract_type": contract_state.contract_type,
        "information_collected_summary": contract_state.required_information if contract_state.required_information else "No information collected yet.",
        "information_collection_complete": contract_state.information_complete,
        "template_research_available": bool(contract_state.template_research),
        "rag_research_available": bool(contract_state.rag_research),
        "sections_planned_available": bool(contract_state.contract_sections),
        "draft_contract_available": bool(contract_state.contract_draft),
        "reviewer_notes_available": bool(contract_state.reviewer_notes),
        "final_draft_available": bool(contract_state.final_draft),
    }
    # Add a brief summary of what each stage means
    stage_descriptions = {
        "contract_type_identification": "Waiting for user to specify the type of contract needed.",
        "template_research": "Researching templates and requirements for the identified contract type.",
        "information_collection": "Collecting specific details from the user for the contract.",
        "section_planning": "Organizing the structure and sections of the contract.",
        "drafting": "Writing the first draft of the contract.",
        "reviewing": "Contract draft is ready for review, or review is in progress.",
        "finalized": "Contract has been finalized."
    }
    status_dict["current_stage_description"] = stage_descriptions.get(contract_state.current_stage, "Unknown stage.")
    return status_dict

# --- Initialization Function (ensure this is complete and correct) ---
def init_agent_components():
    global model, client_chroma, collection, tavily_client, agent_executor, tools_list, tools_dict, _agent_initialized

    if _agent_initialized:
        current_app.logger.info("Agent components already initialized.")
        return

    current_app.logger.info("Initializing agent components...")
    try:
        google_api_key = current_app.config.get('GOOGLE_API_KEY')
        tavily_api_key = current_app.config.get('TAVILY_API_KEY')
        chroma_db_path = current_app.config.get('CHROMA_DB_PATH')

        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in configuration.")
        if not chroma_db_path:
            raise ValueError("CHROMA_DB_PATH not found in configuration.")

        model = ChatGoogleGenerativeAI(
            model=current_app.config.get("GOOGLE_MODEL_NAME", "gemini-1.5-flash-latest"), # Use a config var
            google_api_key=google_api_key,
            temperature=0.7, # Adjust as needed
        )

        os.makedirs(chroma_db_path, exist_ok=True)
        client_chroma = chromadb.PersistentClient(path=chroma_db_path)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=current_app.config.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2") # Use a config var
        )
        collection = client_chroma.get_or_create_collection(
            name=current_app.config.get("CHROMA_COLLECTION_NAME", "contract_templates_api_v2"), # Use a config var
            embedding_function=sentence_transformer_ef,
            metadata={"hnsw:space": "cosine"}
        )
        current_app.logger.info(f"ChromaDB collection '{collection.name}' loaded/created with {collection.count()} items.")

        if tavily_api_key:
            tavily_client = TavilyClient(api_key=tavily_api_key)
        else:
            current_app.logger.warning("TAVILY_API_KEY not found. Web search tool (Tavily) will be disabled.")
            tavily_client = None

        tools_list = [
            identify_contract_type,
            search_contract_templates,
            collect_contract_information,
            plan_contract_sections,
            draft_contract,
            review_contract,
            finalize_contract,
            get_contract_status,
            add_contract_template,
            search_contract_knowledge_base,
            list_contract_templates
        ]
        tools_dict = {t.name: t for t in tools_list}

        system_prompt_str = """You are a professional AI Legal Contract Generation Assistant. Your goal is to guide the user through creating a comprehensive and legally sound contract. You have access to web search (Tavily), a RAG knowledge base of contract templates, and tools to manage the contract generation lifecycle.

        Core Workflow:
        1.  `identify_contract_type`: Start by clearly identifying the specific type of contract the user needs.
        2.  `search_contract_templates`: Once type is known, research official templates online (web search) AND consult the RAG knowledge base for existing examples, clauses, and structures. Present findings to the user.
        3.  `collect_contract_information`: Gather all necessary specific details from the user for their particular contract. Be thorough.
        4.  `plan_contract_sections`: Based on research and collected info, outline the contract's sections.
        5.  `draft_contract`: Generate the full contract draft using the plan, info, and referencing language/clauses from research/RAG.
        6.  `review_contract`: Critically review the draft, or guide the user to review it, offering suggestions.
        7.  `finalize_contract`: Incorporate review feedback and user's final changes to produce the polished contract.

        Knowledge Base Tools:
        -   `add_contract_template`: Allows adding new PDF contract templates to your RAG knowledge base. User will upload file, you'll get path.
        -   `search_contract_knowledge_base`: Directly search the RAG knowledge base for specific clauses or examples.
        -   `list_contract_templates`: Show the user what templates are currently in the RAG knowledge base.

        General Tools:
        -   `get_contract_status`: Inform the user about the current progress and what information/artifacts are available.

        Interaction Guidelines:
        -   Be methodical. Follow the workflow steps.
        -   Explain what you are doing at each step.
        -   When using RAG or web search, clearly state that you are doing so.
        -   Always confirm critical information with the user.
        -   If the RAG knowledge base lacks relevant examples for a specific contract type or clause, state this and rely more on general legal principles and web search if available.
        -   Your primary function is to assist in drafting; you are not a substitute for legal advice from a qualified human lawyer. Include a disclaimer if generating a final document.
        """
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt_str),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_tool_calling_agent(model, tools_list, prompt_template)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools_list,
            verbose=current_app.config.get("LANGCHAIN_VERBOSE", True), # Control verbosity via config
            handle_parsing_errors=True, # Important for robustness
            max_iterations=current_app.config.get("AGENT_MAX_ITERATIONS", 15) # Prevent long loops
        )
        
        _agent_initialized = True
        current_app.logger.info("Agent components initialized successfully.")

    except Exception as e:
        current_app.logger.error(f"CRITICAL: Failed to initialize agent components: {e}", exc_info=True)
        _agent_initialized = False
        raise # Re-raise critical initialization errors to prevent app running in broken state

def is_initialized():
    return _agent_initialized

def get_agent_executor():
    if not _agent_initialized:
        current_app.logger.error("Attempted to get agent_executor but components are not initialized!")
        raise RuntimeError("Agent components are not initialized. API cannot function.")
    return agent_executor

# Ensure tools are callable directly if needed for non-agent interactions (e.g., direct RAG management API endpoints)
# This is already handled by them being standard functions decorated with @tool.