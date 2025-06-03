import os
from app import create_app
from app.services import agent_core # Import agent_core to ensure it's initialized

app = create_app(os.getenv('FLASK_CONFIG') or 'default')

with app.app_context():
    if not agent_core.is_initialized():
        print("Initializing agent components from run.py...")
        agent_core.init_agent_components()

    if __name__ == '__main__':
        app.run(debug=True, host='0.0.0.0', port=5000)
