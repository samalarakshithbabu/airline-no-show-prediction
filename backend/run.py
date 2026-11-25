#!/usr/bin/env python3
"""
Application entry point for running the Flask backend.
"""

import os
import sys

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.app import create_app

# Create application
app = create_app()

if __name__ == '__main__':
    # Load models if available
    model_dir = os.environ.get('MODEL_DIR', 'models')
    if os.path.exists(model_dir):
        print(f"Loading models from {model_dir}...")
        if hasattr(app, 'prediction_service'):
            app.prediction_service.load_all_models()
            print(f"Loaded {len(app.prediction_service.models)} models")
    
    # Run the application
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug_default = 'false' if os.environ.get('FLASK_ENV', 'development').lower() == 'production' else 'true'
    debug = os.environ.get('DEBUG', debug_default).lower() == 'true'
    
    print(f"\nStarting server on {host}:{port}")
    print("API Documentation available at http://localhost:5000/api/docs")
    
    app.run(host=host, port=port, debug=debug)