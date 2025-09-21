#!/usr/bin/env python3
"""
Example usage of the Healthcare Test Case Generator Agent

This demonstrates how to use the healthcare test generator agent
with Google ADK to process PDF documents from GCS.
"""

from . import root_agent

def main():
    """Example of using the healthcare test generator agent."""
    
    print("🏥 Healthcare Test Case Generator Agent")
    print("=" * 50)
    print(f"Agent Name: {root_agent.name}")
    print(f"Description: {root_agent.description}")
    print(f"Available Tools: {len(root_agent.tools)}")
    
    print("\n🛠️ Available Tools:")
    for i, tool in enumerate(root_agent.tools, 1):
        print(f"{i}. {tool.__name__}")
    
    print("\n📋 Usage with ADK:")
    print("1. Import the agent: from healthcare_test_generator import root_agent")
    print("2. Use with ADK Runner to process PRD documents from GCS")
    print("3. The agent will automatically:")
    print("   - List PRD documents from GCS bucket (hackathon-11)")
    print("   - Extract requirements from PDF documents using LLM")
    print("   - Validate compliance with healthcare standards")
    print("   - Generate comprehensive test cases")
    print("   - Create traceability matrix")
    print("   - Export test cases to CSV in datetime-stamped folder in GCS bucket")
    print("   - Export traceability matrix to PDF in datetime-stamped folder in GCS bucket")
    print("   - Export test cases to Jira (if JIRA_PROJECT_KEY is set)")
    print("   - Store additional artifacts in GCS")
    print("\n📝 Note: Jira extraction is commented out to focus on PRD processing, but Jira export is active")
    
    print("\n🔧 Environment Variables Required:")
    print("- GOOGLE_CLOUD_PROJECT: Your GCP project ID")
    print("- GOOGLE_CLOUD_LOCATION: GCP location (e.g., us-central1)")
    print("- GCS_BUCKET: GCS bucket name (default: hackathon-11)")
    print("- JIRA_PROJECT_KEY: Jira project key for test case export (optional)")
    print("- JIRA_URL, JIRA_USERNAME, JIRA_API_TOKEN: Jira credentials (optional)")
    
    print("\n✅ Agent is ready to use with Google ADK!")

if __name__ == "__main__":
    main()
