import os
from app import create_app
from app.services import agent_core # Import agent_core to ensure it's initialized


if __name__ == '__main__':
    app = create_app(os.getenv('FLASK_CONFIG') or 'default')    
    with app.app_context():
        if not agent_core.is_initialized():
            print("Initializing agent components from run.py...")
            agent_core.init_agent_components()

            app.run(debug=True, host='0.0.0.0', port=5000)
        else:
            print("Agent components are already initialized.")
            app.run(debug=True, host='0.0.0.0', port=5000)

