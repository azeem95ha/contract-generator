from flask import Flask, render_template
from .config import config
from .extensions import cors # Example extension
# Import agent_core here if you want to ensure its modules are loaded early
# from .services import agent_core

def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # Initialize extensions
    cors.init_app(app, resources={r"/api/*": {"origins": "*"}}) # Basic CORS for API

    # Initialize agent components (if not done in run.py or elsewhere)
    # This ensures all necessary components like LLM, DB, tools are ready
    # from .services import agent_core # Defer import to avoid circular issues if agent_core imports app
    # if not agent_core.is_initialized():
    #     agent_core.init_agent_components()


    # Register Blueprints
    from .api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api/v1')

    # Example: Make agent_executor accessible globally or via app.extensions
    # from .services.agent_core import agent_executor, contract_state, tools_dict
    # app.extensions['agent_executor'] = agent_executor
    # app.extensions['contract_state'] = contract_state
    # app.extensions['tools_dict'] = tools_dict # For direct tool calls if needed

    @app.route('/health')
    def health_check():
        return "API is healthy!", 200
    
    @app.route('/')
    def index():
        return render_template('index.html')

    return app
