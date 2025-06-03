from flask import render_template, request, jsonify, current_app
import markdown
from werkzeug.utils import secure_filename
import os
import uuid
from typing import List, Dict, Any


from . import api_bp
from .schemas import (
    ChatMessageInput, AgentResponse, AddTemplateForm, RagSearchQuery,
    StatusResponse, TemplateListResponse, ErrorResponse
)
from app.services import agent_core # Import the module
from app.services.utils import extract_text_from_pdf_bytes # For direct PDF byte processing if needed
from langchain_core.messages import HumanMessage, AIMessage # For chat history format

# Ensure agent components are initialized when this blueprint is registered
# This is a crucial step. It should ideally happen once when the app starts.
# The call in run.py or app/__init__.py's create_app should handle this.
if not agent_core.is_initialized():
    # This might be too late if current_app context is needed for init.
    # It's better to initialize earlier.
    # agent_core.init_agent_components()
    # For now, assume it's initialized by create_app or run.py
    print("WARNING: agent_core might not be initialized at routes.py import time. Ensure init happens in create_app.")


def _format_chat_history_for_agent(history_dicts: List[Dict[str, str]]) -> List:
    """Converts API chat history (list of dicts) to Langchain Message objects."""
    lc_messages = []
    for msg_dict in history_dicts:
        if msg_dict.get("role") == "human":
            lc_messages.append(HumanMessage(content=msg_dict.get("content", "")))
        elif msg_dict.get("role") == "ai" or msg_dict.get("role") == "assistant": # Langchain uses 'ai'
            lc_messages.append(AIMessage(content=msg_dict.get("content", "")))
    return lc_messages

def _format_chat_history_for_api(lc_messages: List) -> List[Dict[str, str]]:
    """Converts Langchain Message objects back to API chat history (list of dicts)."""
    api_history = []
    for lc_msg in lc_messages:
        role = "unknown"
        if isinstance(lc_msg, HumanMessage):
            role = "human"
        elif isinstance(lc_msg, AIMessage):
            role = "ai"
        api_history.append({"role": role, "content": lc_msg.content})
    return api_history


@api_bp.route('/chat', methods=['POST'])
def chat_with_agent():
    if not agent_core.is_initialized():
        return jsonify(ErrorResponse(error="Service Unavailable", message="Agent core not initialized.").model_dump()), 503

    try:
        data = ChatMessageInput(**request.json)
    except Exception as e: # Pydantic validation error
        return jsonify(ErrorResponse(error="Bad Request", message="Invalid input data.", details=str(e)).model_dump()), 400

    agent_executor = agent_core.get_agent_executor()
    if not agent_executor:
         return jsonify(ErrorResponse(error="Service Unavailable", message="Agent executor not available.").model_dump()), 503

    formatted_chat_history = _format_chat_history_for_agent(data.chat_history)

    try:
        response = agent_executor.invoke({
            "input": data.user_input,
            "chat_history": formatted_chat_history
        })
        output_message = response.get('output', "No response from agent.")

        # Update chat history for API response
        updated_api_history = data.chat_history + [
            {"role": "human", "content": data.user_input},
            {"role": "ai", "content": output_message}
        ]
        # Keep history manageable (optional, client can also do this)
        if len(updated_api_history) > 20:
            updated_api_history = updated_api_history[-20:]


        return jsonify(AgentResponse(
            output=output_message,
            chat_history=updated_api_history,
            current_stage=agent_core.contract_state.current_stage, # Access global state
            contract_state_summary=agent_core.get_contract_status.func() # Get current full status
        ).model_dump()), 200

    except Exception as e:
        current_app.logger.error(f"Error in agent execution: {e}", exc_info=True)
        return jsonify(ErrorResponse(error="Internal Server Error", message=str(e)).model_dump()), 500


@api_bp.route('/rag/templates', methods=['POST'])
def add_template_route():
    if not agent_core.is_initialized():
        return jsonify(ErrorResponse(error="Service Unavailable", message="Agent core not initialized.").model_dump()), 503
    if 'pdf_file' not in request.files:
        return jsonify(ErrorResponse(error="Bad Request", message="No PDF file part in the request.").model_dump()), 400
    
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify(ErrorResponse(error="Bad Request", message="No PDF file selected.").model_dump()), 400

    try:
        # Use Pydantic for form data validation if possible, or direct request.form access
        contract_type = request.form.get('contract_type')
        description = request.form.get('description', "")
        if not contract_type:
            return jsonify(ErrorResponse(error="Bad Request", message="Missing 'contract_type' form field.").model_dump()), 400
        
        # form_data = AddTemplateForm(contract_type=contract_type, description=description) # For validation
    except Exception as e: # Pydantic validation error for form fields
        return jsonify(ErrorResponse(error="Bad Request", message="Invalid form data.", details=str(e)).model_dump()), 400


    if file: # Already checked filename != ''
        filename = secure_filename(file.filename)
        # Create a unique sub-folder for each upload or unique filename
        temp_filename = f"{uuid.uuid4().hex}_{filename}"
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True) # Ensure upload folder exists
        temp_file_path = os.path.join(upload_folder, temp_filename)
        
        try:
            file.save(temp_file_path)
            # Call the tool function (which is part of agent_core.tools_list or agent_core.tools_dict)
            # Ensure add_contract_template is available via agent_core
            result_message = agent_core.add_contract_template.func(
                pdf_file_path=temp_file_path,
                contract_type=contract_type, # form_data.contract_type,
                description=description # form_data.description
            )
            if "Error:" in result_message:
                 return jsonify(ErrorResponse(error="Processing Error", message=result_message).model_dump()), 422 # Unprocessable Entity
            return jsonify({"message": result_message}), 201
        except Exception as e:
            current_app.logger.error(f"Error processing uploaded PDF: {e}", exc_info=True)
            return jsonify(ErrorResponse(error="Internal Server Error", message=f"Failed to process PDF: {str(e)}").model_dump()), 500
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError as e:
                    current_app.logger.error(f"Error deleting temp file {temp_file_path}: {e}")
    
    return jsonify(ErrorResponse(error="Bad Request", message="File processing failed unexpectedly.").model_dump()), 400


@api_bp.route('/rag/templates', methods=['GET'])
def list_templates_route():
    if not agent_core.is_initialized():
        return jsonify(ErrorResponse(error="Service Unavailable", message="Agent core not initialized.").model_dump()), 503
    try:
        templates_summary = agent_core.list_contract_templates.func()
        return jsonify(TemplateListResponse(templates_summary=templates_summary).model_dump()), 200
    except Exception as e:
        current_app.logger.error(f"Error listing templates: {e}", exc_info=True)
        return jsonify(ErrorResponse(error="Internal Server Error", message=str(e)).model_dump()), 500


@api_bp.route('/rag/search', methods=['GET']) # Using GET with query params
def search_knowledge_base_route():
    if not agent_core.is_initialized():
        return jsonify(ErrorResponse(error="Service Unavailable", message="Agent core not initialized.").model_dump()), 503
    try:
        # Pydantic can also validate query parameters if you use a library like flask-pydantic
        # For now, direct access:
        query_params = RagSearchQuery(
            query=request.args.get('query'),
            contract_type=request.args.get('contract_type'),
            top_k=int(request.args.get('top_k', 5))
        )
    except Exception as e: # Pydantic validation for query params
        return jsonify(ErrorResponse(error="Bad Request", message="Invalid query parameters.", details=str(e)).model_dump()), 400

    if not query_params.query:
        return jsonify(ErrorResponse(error="Bad Request", message="Missing 'query' parameter.").model_dump()), 400

    try:
        search_results = agent_core.search_contract_knowledge_base.func(
            query=query_params.query,
            contract_type=query_params.contract_type,
            top_k=query_params.top_k
        )
        return jsonify({"results": search_results}), 200
    except Exception as e:
        current_app.logger.error(f"Error searching knowledge base: {e}", exc_info=True)
        return jsonify(ErrorResponse(error="Internal Server Error", message=str(e)).model_dump()), 500


@api_bp.route('/status', methods=['GET'])
def get_status_route():
    if not agent_core.is_initialized():
        return jsonify(ErrorResponse(error="Service Unavailable", message="Agent core not initialized.").model_dump()), 503
    try:
        status_data = agent_core.get_contract_status.func() # This tool now returns a dict
        return jsonify(StatusResponse(status=status_data).model_dump()), 200
    except Exception as e:
        current_app.logger.error(f"Error getting status: {e}", exc_info=True)
        return jsonify(ErrorResponse(error="Internal Server Error", message=str(e)).model_dump()), 500

@api_bp.route('/reset_state', methods=['POST'])
def reset_state_route():
    """ An endpoint to reset the global contract_state.
        USE WITH CAUTION, especially in multi-user scenarios without proper session management.
    """
    if not agent_core.is_initialized():
        return jsonify(ErrorResponse(error="Service Unavailable", message="Agent core not initialized.").model_dump()), 503
    try:
        agent_core.contract_state.reset()
        current_app.logger.info("Global contract state has been reset via API.")
        return jsonify({"message": "Contract state has been reset."}), 200
    except Exception as e:
        current_app.logger.error(f"Error resetting state: {e}", exc_info=True)
        return jsonify(ErrorResponse(error="Internal Server Error", message=str(e)).model_dump()), 500
    
@api_bp.route('/display_contract', methods=['GET'])
def display_final_contract():
    if not agent_core.is_initialized() or not agent_core.contract_state:
        # You might want a nicer error page template here too
        return render_template('contract_display.html', contract_html_content="<p>Error: Contract generation service not ready or no contract state available.</p>")

    final_draft_markdown = agent_core.contract_state.final_draft

    if not final_draft_markdown:
        contract_html = "<p>No finalized contract is available yet. Please complete the contract generation process.</p>"
    else:
        try:
            # Convert Markdown to HTML using the 'markdown' library
            # You can add extensions for more features like tables, fenced code blocks, etc.
            # pip install markdown pygments (for code highlighting if needed)
            # html = markdown.markdown(final_draft_markdown, extensions=['fenced_code', 'tables', 'sane_lists'])
            contract_html = markdown.markdown(final_draft_markdown, extensions=['extra', 'sane_lists'])
        except Exception as e:
            current_app.logger.error(f"Error converting contract markdown to HTML: {e}")
            contract_html = f"<p>Error displaying contract: Could not parse content. Please check the contract source.</p><pre>{final_draft_markdown}</pre>" # Show raw if parsing fails

    return render_template('contract_display.html', contract_html_content=contract_html)