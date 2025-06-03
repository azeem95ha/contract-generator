import os
from dotenv import load_dotenv

load_dotenv() # Load .env file

basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, '..', 'data', 'uploads') # For PDF uploads
CHROMA_DB_PATH = os.path.join(basedir, '..', 'data', 'contract_templates_db')

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-very-secret-dev-key'
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
    UPLOAD_FOLDER = UPLOAD_FOLDER
    CHROMA_DB_PATH = CHROMA_DB_PATH
    LANGCHAIN_TRACING_V2 = "true" # Common for LangSmith
    LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY') # If using LangSmith
    LANGCHAIN_PROJECT = os.environ.get('LANGCHAIN_PROJECT') or "Contract-Gen-API"
    GOOGLE_MODEL_NAME = 'gemini-2.5-flash-preview-05-20'


    @staticmethod
    def init_app(app):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        if not Config.GOOGLE_API_KEY:
            print("WARNING: GOOGLE_API_KEY not set in .env file!")
        if not Config.TAVILY_API_KEY:
            print("WARNING: TAVILY_API_KEY not set in .env file!")

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
    # Add any test-specific configs, e.g., in-memory DB if applicable

class ProductionConfig(Config):
    DEBUG = False
    # Add production-specific configs

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
