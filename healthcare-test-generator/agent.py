# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Healthcare Test Case Generator Agent

Main agent that orchestrates the generation of healthcare test cases from requirements.
Follows the Google ADK samples structure pattern.
"""

from google.adk import Agent
from .tools import (
    # Document processing functions
    extract_pdf_text,
    analyze_requirements_from_text,
    process_requirements_document,
    list_prd_documents,
    # Analysis and validation functions
    validate_compliance_requirements,
    # Export functions
    export_to_jira,
    store_in_gcs,
    export_test_cases_to_csv,
    export_traceability_matrix_to_pdf,
    # Agentic orchestration functions
    get_workflow_status,
    validate_processing_step,
    # MALFORMED DATA CORRECTION
    fix_malformed_function_output,
    fix_malformed_adk_call,
    # USER-CONTROLLED TEST CASE GENERATION FUNCTIONS
    list_extracted_requirements,
    generate_test_cases_for_requirements,
    create_test_case_reports,
    create_traceability_report
    # Note: process_healthcare_prd_to_test_cases removed to enable step-by-step agentic flow
)
# Note: Prompts are used directly in the tools, not in the agent definition

# Define the main healthcare test generator agent with agentic orchestration
root_agent = Agent(
    model="gemini-2.5-pro",
    name="healthcare_test_generator",
    description="AI-powered healthcare test case generator that converts healthcare software requirements into compliant, traceable test cases. Supports FDA, HIPAA, IEC 62304, and other healthcare compliance standards. Generates professional test documentation and traceability matrices for regulatory audits.",
        tools=[
            # === DOCUMENT DISCOVERY & INPUT ===
            list_prd_documents,              # List available PRD documents in GCS bucket
            
            # === DOCUMENT PROCESSING ===
            extract_pdf_text,                # Extract raw text from PDF documents
            analyze_requirements_from_text,  # Analyze text and extract structured requirements
            process_requirements_document,   # Complete PDF -> requirements workflow
            
            # === COMPLIANCE & VALIDATION ===
            validate_compliance_requirements, # Validate requirements against healthcare standards
            validate_processing_step,        # Validate intermediate processing results
            
            # === MALFORMED DATA CORRECTION ===
            fix_malformed_function_output,   # 🔧 Fix malformed outputs using simple validation
            fix_malformed_adk_call,          # 🔧 Fix malformed ADK function calls with complex JSON
            
            # === TEST CASE GENERATION (USER-CONTROLLED WORKFLOW) ===
            list_extracted_requirements,      # 📋 List requirements for user selection
            generate_test_cases_for_requirements, # 🎯 Generate test cases for selected requirements
            create_test_case_reports,         # 📄 Create PDF and CSV reports from test cases
            create_traceability_report,       # 🔗 Create traceability matrix PDF
            
            # === WORKFLOW MONITORING ===
            get_workflow_status,             # Check status of workflow processing
            
            # === EXPORT & OUTPUT ===
            export_test_cases_to_csv,        # Export test cases to CSV format
            export_traceability_matrix_to_pdf, # Export traceability matrix to PDF
            export_to_jira,                  # Export test cases to Jira (optional)
            store_in_gcs                     # Store artifacts in GCS
            
            # Note: Removed process_healthcare_prd_to_test_cases to enable true agentic orchestration
            # Users can now see each individual step and the agent will orchestrate them transparently
        ]
)
