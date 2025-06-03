# Flask Contract Generation API

A Flask-based API application for an AI-powered contract generation assistant. It uses LLMs, RAG with ChromaDB, and web search to create contracts interactively.

## ✨ Features

*   Interactive chat interface for contract generation.
*   Knowledge base management (add, list, search PDF contract templates).
*   Retrieval Augmented Generation (RAG) for contextual responses.
*   Web search integration for up-to-date information.
*   State management for the contract generation lifecycle.

## 📋 Prerequisites

*   Python 3.8+
*   Git

## 🚀 Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
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
    *   Edit the `.env` file and add your API keys and other configurations:
        ```
        FLASK_APP=run.py
        FLASK_ENV=development
        SECRET_KEY='your_very_strong_random_secret_key'
        GOOGLE_API_KEY='your_google_ai_api_key'
        TAVILY_API_KEY='your_tavily_api_key'
        # Optional LangSmith keys
        # LANGCHAIN_TRACING_V2="true"
        # LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
        # LANGCHAIN_API_KEY="your_langsmith_api_key"
        # LANGCHAIN_PROJECT="Contract-Gen-API"
        ```

5.  **Run the application:**
    ```bash
    flask run
    # Or: python run.py
    ```
    The API will be available at `http://localhost:5000`.
    The HTML interface will be at `http://localhost:5000/`.

## 📖 API Documentation

For detailed API endpoint specifications, request/response formats, and usage examples, please see the [Full API Documentation](./docs/index.md) (or link to your deployed MkDocs site).

*(A brief summary of key endpoints can go here if desired)*

## 🧪 Running Tests

*(Instructions on how to run tests once you've added them)*

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

##🤝 Contributing

*(Link to CONTRIBUTING.md or brief contribution guidelines)*