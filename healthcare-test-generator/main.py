#!/usr/bin/env python3
"""
Healthcare Test Case Generator - Cloud Run Entry Point

This file serves as the entry point for deploying the Healthcare Test Case Generator
to Google Cloud Run using gcloud CLI.
"""

import os
from google.adk.web import AdkWebServer

# Environment configuration
SERVE_WEB_INTERFACE = True  # Enable the ADK web UI
PORT = int(os.environ.get("PORT", 8080))  # Cloud Run sets PORT environment variable

def main():
    """Main entry point for Cloud Run deployment."""
    # Set up ADK web server configuration
    server_config = {
        "port": PORT,
        "host": "0.0.0.0",  # Required for Cloud Run
        "agents_source_dir": ".",  # Current directory contains the agent
        "serve_web_interface": SERVE_WEB_INTERFACE,
    }
    
    # Start the ADK web server
    server = AdkWebServer(**server_config)
    server.run()

if __name__ == "__main__":
    main()
