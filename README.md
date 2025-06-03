# Flask Contract Generation API

A Flask-based API application for an AI-powered contract generation assistant. It utilizes Large Language Models (LLMs) via Langchain, Retrieval Augmented Generation (RAG) with a ChromaDB vector store, and Tavily for web search capabilities to interactively create various types of contracts.

This API powers an assistant that guides users through the contract lifecycle, from type identification and information gathering to drafting and finalization.

## ✨ Features

*   **Interactive Chat Interface:** Core interaction via a conversational AI agent.
*   **Langchain Agent & Tools:** Modular tools for specific tasks (identifying contract type, RAG search, web search, drafting, etc.).
*   **Knowledge Base Management:**
    *   Add PDF contract templates to a persistent ChromaDB vector store.
    *   List available templates.
    *   Search the knowledge base using semantic search (RAG).
*   **Web Search Integration:** Uses Tavily to fetch up-to-date information and official template requirements.
*   **Stateful Contract Generation:** Manages the state of the contract generation process (e.g., current stage, collected information).
*   **Markdown Support:** Agent responses and final contracts can be in Markdown.
*   **Simple HTML UI:** Includes a basic HTML/JavaScript interface for direct API interaction and contract viewing.

## 📋 Prerequisites

*   Python 3.8+
*   Git
*   Access to Google Generative AI (Gemini) and Tavily API keys.

## 🚀 Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/azeem95ha/contract-generator.git
    cd contract-generator
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your API keys and other configurations. **Refer to `.env.example` for the required variables.** Crucially, you will need:
        *   `SECRET_KEY`: A strong, random string for Flask.
        *   `GOOGLE_API_KEY`: Your API key for Google Generative AI.
        *   `TAVILY_API_KEY`: Your API key for Tavily search.

5.  **Initialize Agent Components (if not automatic on first run):**
    The application attempts to initialize AI components on startup. If you encounter issues, ensure the `init_agent_components()` function in `app/services/agent_core.py` is being called correctly (e.g., from `app/__init__.py` within an app context).

6.  **Run the application:**
    ```bash
    flask run
    # Or, if you prefer:
    # python run.py
    ```
    The API will typically be available at `http://localhost:5000/api/v1`.
    The HTML interface will be at `http://localhost:5000/`.

## 📖 API Documentation

For detailed API endpoint specifications, request/response formats, and usage examples, please see the [Full API Documentation](./API_DOCUMENTATION.md). *(You would replace this with a link to your MkDocs site once deployed, or keep it as a local file if you prefer).*

**Key API Endpoints:**
*   `POST /api/v1/chat`: Interact with the contract generation agent.
*   `POST /api/v1/rag/templates`: Add a new PDF contract template.
*   `GET /api/v1/rag/templates`: List available templates.
*   `GET /api/v1/rag/search`: Search the knowledge base.
*   `GET /api/v1/status`: Get the current contract generation status.
*   `POST /api/v1/reset_state`: Reset the agent's contract state.

**HTML Interface Routes:**
*   `GET /`: Main HTML interface.
*   `GET /view_contract`: Displays the finalized contract in Markdown.

## 🧪 Running Tests

*(TODO: Add instructions on how to run tests once a test suite is implemented.)*

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.
*(For more detailed guidelines, a `CONTRIBUTING.md` file can be added).*