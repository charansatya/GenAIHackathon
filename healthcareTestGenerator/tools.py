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
Healthcare Test Case Generator Tools

Custom tools for document processing, Jira integration, GCS operations,
and healthcare-specific functionality using Google GenAI.
"""

import logging
import json
import os
import sys
# Simplified type hints for ADK compatibility
from datetime import datetime, timedelta
import requests
from google.cloud import storage
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from typing import Union, Optional

# Simple JSON validation and correction utilities
import pandas as pd
from pathlib import Path
from google import genai
from google.genai import types
from jinja2 import Template
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(log_level: str = None, log_file: str = None):
    """
    Configure comprehensive logging for debugging and monitoring.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
    """
    # Get log level from environment or default to INFO
    level = log_level or os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure root logger
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a'))
    
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set specific logger levels for external libraries
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("jira").setLevel(logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("HEALTHCARE TEST GENERATOR - LOGGING INITIALIZED")
    logger.info(f"Log Level: {level}")
    logger.info(f"Log File: {log_file or 'Console only'}")
    logger.info("="*70)
    
    return logger

# Initialize logging
logger = setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "output/healthcare_test_generator.log")
)

# Project Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "hackathon-11")

def init_genai_client():
    """Initialize the Google Genai client."""
    try:
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
        )
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Genai client: {str(e)}")
        return None

# Document Processing Tools

def list_prd_documents(bucket_name: str):
    """
    List all PRD documents in the GCS input folder.
    
    Method Signature:
        list_prd_documents(bucket_name: str) -> list
    
    Args:
        bucket_name (str): GCS bucket name where PRD documents are stored
    
    Returns:
        list: List of GCS paths to PRD documents (gs://bucket-name/path/to/file.pdf)
        
    Example:
        prd_docs = list_prd_documents("my-healthcare-bucket")
        # Returns: ["gs://hackathon-11/prd1.pdf", ...]
    """
    try:
        logging.info(f"Listing PRD documents in bucket: {bucket_name}")
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # List all PDF files in the bucket
        blobs = bucket.list_blobs(prefix="")
        prd_documents = []
        
        for blob in blobs:
            if blob.name.lower().endswith('.pdf') and not blob.name.endswith('/'):
                gcs_path = f"gs://{bucket_name}/{blob.name}"
                prd_documents.append(gcs_path)
        
        logging.info(f"Found {len(prd_documents)} PRD documents")
        return prd_documents
        
    except Exception as e:
        logging.error(f"Error listing PRD documents: {e}")
        return []

# Broken down into simpler functions for better ADK compatibility

def extract_pdf_text(gcs_pdf_path: str):
    """
    Extract text content from PDF stored in GCS bucket.
    
    Method Signature:
        extract_pdf_text(gcs_pdf_path: str) -> dict
    
    Args:
        gcs_pdf_path (str): GCS path to the PDF document (gs://bucket-name/path/to/document.pdf)
    
    Returns:
        dict: Dictionary containing extraction results with keys:
            - status (str): "success" or "error"
            - gcs_path (str): Original GCS path
            - text_content (str): Extracted text from PDF
            - extraction_timestamp (str): ISO timestamp of extraction
            - error (str): Error message if status is "error"
            
    Example:
        result = extract_pdf_text("gs://my-bucket/hackathon-11/Input/prd.pdf")
        # Returns: {"status": "success", "text_content": "...", ...}
    """
    try:
        logging.info(f"Extracting text from PDF: {gcs_pdf_path}")
        
        # Download and extract text from PDF in GCS
        document_content = _extract_pdf_text_from_gcs(gcs_pdf_path)
        
        return {
            "status": "success",
            "gcs_path": gcs_pdf_path,
            "text_content": document_content,
            "extraction_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error extracting PDF text: {e}")
        return {"error": f"Failed to extract PDF text: {e}"}

def analyze_requirements_from_text(text_content: str, standards: str):
    """
    Analyze text content and extract healthcare requirements using LLM.
    
    Method Signature:
        analyze_requirements_from_text(text_content: str, standards: str) -> dict
    
    Args:
        text_content (str): Extracted text content from PDF document
        standards (str): Comma-separated compliance standards (e.g., "FDA,HIPAA,IEC 62304")
    
    Returns:
        dict: Dictionary containing extracted requirements and metadata with keys:
            - requirements (list): List of requirement dictionaries
            - metadata (dict): Document metadata
            - compliance_standards (list): Applied compliance standards
            - processing_timestamp (str): ISO timestamp
            - error (str): Error message if extraction failed
            
    Example:
        result = analyze_requirements_from_text("Document text...", "FDA,HIPAA")
        # Returns: {"requirements": [...], "metadata": {...}, ...}
    """
    try:
        logging.info(f"Analyzing requirements from text content")
        
        # Convert string to list of compliance standards
        if isinstance(standards, str):
            compliance_standards = [s.strip() for s in standards.split(',') if s.strip()]
        else:
            compliance_standards = standards if standards else ["FDA", "HIPAA", "IEC 62304"]
        
        client = init_genai_client()
        if not client:
            raise RuntimeError("Failed to initialize GenAI client.")
        
        # Create prompt for LLM to extract requirements
        prompt = f"""
        You are a healthcare software requirements analyst. Extract and analyze requirements from the following document text.
        
        Compliance Standards: {', '.join(compliance_standards)}
        
        Document Content:
        {text_content}
        
        Please extract all requirements and return them in the following JSON format:
        {{
            "requirements": [
                {{
                    "id": "REQ-001",
                    "title": "Requirement Title",
                    "description": "Detailed requirement description",
                    "type": "functional|non-functional|security|compliance",
                    "priority": "critical|high|medium|low",
                    "category": "security|compliance|performance|usability|reliability",
                    "acceptance_criteria": ["Criterion 1", "Criterion 2"],
                    "compliance_standards": ["FDA", "HIPAA", "IEC 62304"],
                    "risk_level": "high|medium|low",
                    "page_number": 1,
                    "section": "Section Name"
                }}
            ],
            "metadata": {{
                "document_title": "Document Title",
                "version": "1.0",
                "author": "Author Name",
                "last_modified": "2024-01-15",
                "total_requirements": 0,
                "document_type": "PDF"
            }}
        }}
        
        Focus on healthcare-specific requirements and ensure compliance with the specified standards.
        Include page numbers and section references where possible.
        """
        
        # Generate content using LLM
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        # Parse LLM response
        requirements = json.loads(response.candidates[0].content.parts[0].text)
        
        # Add processing metadata
        requirements["compliance_standards"] = compliance_standards
        requirements["processing_timestamp"] = datetime.now().isoformat()
        
        logging.info(f"Successfully analyzed text and extracted {len(requirements.get('requirements', []))} requirements")
        return requirements
        
    except Exception as e:
        logging.error(f"Error analyzing requirements from text: {e}")
        return {"error": f"Failed to analyze requirements from text: {e}"}

def process_requirements_document(gcs_pdf_path: str, standards: str):
    """
    Complete workflow: Extract PDF text and analyze requirements.
    
    Method Signature:
        process_requirements_document(gcs_pdf_path: str, standards: str) -> dict
    
    Args:
        gcs_pdf_path (str): GCS path to the PDF document (gs://bucket-name/path/to/document.pdf)
        standards (str): Comma-separated compliance standards (e.g., "FDA,HIPAA,IEC 62304")
    
    Returns:
        dict: Dictionary containing extracted requirements and metadata with keys:
            - requirements (list): List of requirement dictionaries
            - metadata (dict): Document metadata including gcs_path
            - compliance_standards (list): Applied compliance standards
            - processing_timestamp (str): ISO timestamp
            - error (str): Error message if processing failed
            
    Example:
        result = process_requirements_document("gs://bucket/prd.pdf", "FDA,HIPAA")
        # Returns: {"requirements": [...], "metadata": {...}, ...}
    """
    try:
        # Step 1: Extract text from PDF
        text_result = extract_pdf_text(gcs_pdf_path)
        if "error" in text_result:
            return text_result
        
        # Step 2: Analyze requirements from text
        requirements_result = analyze_requirements_from_text(text_result["text_content"], standards)
        if "error" in requirements_result:
            return requirements_result
        
        # Add GCS path to metadata
        if "metadata" in requirements_result:
            requirements_result["metadata"]["gcs_path"] = gcs_pdf_path
        
        return requirements_result
        
    except Exception as e:
        logging.error(f"Error in complete requirements processing: {e}")
        return {"error": f"Failed to process requirements document: {e}"}

def _extract_pdf_text_from_gcs(gcs_pdf_path: str) -> str:
    """Extract text content from PDF stored in GCS bucket."""
    try:
        # Parse GCS path
        parsed_url = urlparse(gcs_pdf_path)
        bucket_name = parsed_url.netloc
        blob_path = parsed_url.path.lstrip('/')
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Download PDF content
        pdf_content = blob.download_as_bytes()
        
        # Extract text using PyPDF2
        import PyPDF2
        import io
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text_content = ""
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            text_content += f"\n--- Page {page_num} ---\n"
            text_content += page_text
            text_content += "\n"
        
        logging.info(f"Successfully extracted text from PDF: {gcs_pdf_path}")
        return text_content
        
    except Exception as e:
        logging.error(f"Error extracting PDF text from GCS: {e}")
        return f"[Error extracting PDF text from {gcs_pdf_path}: {e}]"


# Jira Integration Tools

# JIRA EXTRACTION - COMMENTED OUT FOR NOW
# Focus is on PRD document processing from GCS input folder
# This can be uncommented later if Jira integration is needed

# def extract_requirements_from_jira(jira_query: str, project_key: str, compliance_standards: List[str]) -> Dict[str, Any]:
#     """
#     Extract requirements from Jira tickets and epics.
#     
#     Args:
#         jira_query: JQL query to fetch requirements
#         project_key: Jira project key
#         compliance_standards: Applicable compliance standards
#     
#     Returns:
#         Dictionary containing extracted requirements from Jira
#     """
#     try:
#         logging.info(f"Extracting requirements from Jira project: {project_key}")
#         
#         # Extract Jira issues using the Jira API
#         jira_issues = _extract_jira_issues(jira_query, project_key)
#         
#         if "error" in jira_issues:
#             return {"error": f"Failed to extract Jira issues: {jira_issues['error']}"}
#         
#         # Process the raw Jira issues into requirements format
#         raw_jira_issues = jira_issues
#         
#         # Initialize GenAI client for LLM processing
#         client = init_genai_client()
#         if not client:
#             raise RuntimeError("Failed to initialize GenAI client.")
#         
#         # Create prompt for LLM to convert Jira issues to requirements
#         prompt = f"""
#         You are a healthcare software requirements analyst. Convert the following Jira issues into structured requirements.
#         
#         Jira Project: {project_key}
#         JQL Query: {jira_query}
#         Compliance Standards: {', '.join(compliance_standards)}
#         
#         Raw Jira Issues:
#         {json.dumps(raw_jira_issues, indent=2)}
#         
#         Please convert these Jira issues into requirements and return them in the following JSON format:
#         {{
#             "requirements": [
#                 {{
#                     "id": "JIRA-123",
#                     "title": "Requirement Title from Jira Summary",
#                     "description": "Detailed requirement description from Jira Description",
#                     "type": "story|epic|task",
#                     "priority": "critical|high|medium|low",
#                     "category": "security|compliance|performance|usability|reliability",
#                     "acceptance_criteria": ["Criterion 1", "Criterion 2"],
#                     "compliance_standards": ["FDA", "HIPAA", "IEC 62304"],
#                     "risk_level": "high|medium|low",
#                     "jira_key": "MED-123"
#                 }}
#             ],
#             "metadata": {{
#                 "jql_query": "{jira_query}",
#                 "jira_project": "{project_key}",
#                 "total_issues_extracted": 0,
#                 "extraction_timestamp": "{datetime.now().isoformat()}"
#             }}
#         }}
#         
#         Focus on healthcare-specific requirements and ensure compliance with the specified standards.
#         """
#         
#         # Generate content using LLM
#         response = client.models.generate_content(
#             model="gemini-2.5-pro",
#             contents=prompt,
#             config=genai.types.GenerateContentConfig(
#                 temperature=0.1,
#                 response_mime_type="application/json"
#             )
#         )
#         
#         # Parse LLM response
#         jira_requirements = json.loads(response.candidates[0].content.parts[0].text)
#         
#         # Add compliance metadata
#         jira_requirements["compliance_standards"] = compliance_standards
#         jira_requirements["extraction_timestamp"] = datetime.now().isoformat()
#         jira_requirements["jira_project"] = project_key
#         
#         logging.info(f"Successfully extracted {len(jira_requirements.get('requirements', []))} requirements from Jira")
#         return jira_requirements
#         
#     except Exception as e:
#         logging.error(f"Error extracting from Jira: {e}")
#         return {"error": f"Failed to extract from Jira: {e}"}

# def _extract_jira_issues(jql_query: str, project_key: str) -> Dict[str, Any]:
#     """Extract issues from Jira using the Jira API."""
#     try:
#         from jira import JIRA
#         
#         # Initialize Jira client
#         jira_url = os.getenv("JIRA_URL")
#         jira_username = os.getenv("JIRA_USERNAME")
#         jira_token = os.getenv("JIRA_API_TOKEN")
#         
#         if not all([jira_url, jira_username, jira_token]):
#             raise ValueError("Jira credentials not configured")
#         
#         jira_client = JIRA(
#             server=jira_url,
#             basic_auth=(jira_username, jira_token)
#         )
#         
#         # Search for issues using JQL
#         issues = jira_client.search_issues(jql_query, expand='changelog')
#         
#         # Extract issue data
#         extracted_issues = []
#         for issue in issues:
#             issue_data = {
#                 "key": issue.key,
#                 "summary": issue.fields.summary,
#                 "description": issue.fields.description or "",
#                 "issuetype": issue.fields.issuetype.name,
#                 "priority": getattr(issue.fields, 'priority', {}).get('name', 'Medium') if hasattr(issue.fields, 'priority') and issue.fields.priority else 'Medium',
#                 "labels": getattr(issue.fields, 'labels', []) or [],
#                 "status": issue.fields.status.name,
#                 "assignee": getattr(issue.fields, 'assignee', {}).get('displayName', 'Unassigned') if hasattr(issue.fields, 'assignee') and issue.fields.assignee else 'Unassigned',
#                 "created": issue.fields.created,
#                 "updated": issue.fields.updated
#             }
#             extracted_issues.append(issue_data)
#         
#         return {
#             "issues": extracted_issues,
#             "jql_query": jql_query,
#             "project_key": project_key,
#             "total_count": len(extracted_issues)
#         }
#         
#     except Exception as e:
#         logging.error(f"Error extracting Jira issues: {e}")
#         return {
#             "issues": [],
#             "jql_query": jql_query,
#             "project_key": project_key,
#             "error": str(e)
#         }

def export_to_jira(test_cases_json: str, project_key: str, epic_key: str = "", traceability_matrix_json: str = ""):
    """
    Export generated test cases to Jira as tickets.
    
    Method Signature:
        export_to_jira(test_cases_json: str, project_key: str, epic_key: str = "", traceability_matrix_json: str = "") -> dict
    
    Args:
        test_cases_json (str): JSON string of test cases to export
        project_key (str): Target Jira project key (e.g., "MED")
        epic_key (str, optional): Parent epic key for linking test cases (empty string if none)
        traceability_matrix_json (str, optional): JSON string of traceability matrix data (empty string if none)
    
    Returns:
        dict: Dictionary containing export results with keys:
            - exported_cases (list): List of successfully exported test case dictionaries
            - project_key (str): Target Jira project key
            - epic_key (str): Parent epic key (if provided)
            - export_timestamp (str): ISO timestamp
            - total_exported (int): Number of test cases successfully exported
            - traceability_matrix (dict): Traceability matrix (if provided)
            - error (str): Error message if export failed
            
    Example:
        result = export_to_jira('{"test_cases": [...]}', "MED", "MED-123", '{"requirements": [...]}')
        # Returns: {"exported_cases": [...], "total_exported": 15, ...}
    """
    try:
        # Handle file path if provided instead of JSON string
        if isinstance(test_cases_json, str) and os.path.exists(test_cases_json):
            logger.info(f"Reading test cases from file: {test_cases_json}")
            with open(test_cases_json, 'r', encoding='utf-8') as f:
                test_cases_data = json.load(f)
        # Parse inputs
        elif isinstance(test_cases_json, str):
            test_cases_data = json.loads(test_cases_json)
        else:
            test_cases_data = test_cases_json
            
        # Extract test cases array from the data structure
        if isinstance(test_cases_data, dict) and 'test_cases' in test_cases_data:
            test_cases = test_cases_data['test_cases']
        elif isinstance(test_cases_data, list):
            test_cases = test_cases_data
        else:
            test_cases = test_cases_data
            
        # Parse traceability matrix if provided
        traceability_matrix = None
        if traceability_matrix_json and traceability_matrix_json != "":
            if isinstance(traceability_matrix_json, str):
                traceability_matrix = json.loads(traceability_matrix_json)
            else:
                traceability_matrix = traceability_matrix_json
        
        # Handle empty epic_key
        if epic_key == "":
            epic_key = None
            
        logging.info(f"Exporting {len(test_cases)} test cases to Jira project: {project_key}")
        
        from jira import JIRA
        
        # Initialize Jira client
        jira_url = os.getenv("JIRA_URL")
        # Support both JIRA_EMAIL and JIRA_USERNAME for compatibility
        jira_username = os.getenv("JIRA_EMAIL") or os.getenv("JIRA_USERNAME")
        jira_token = os.getenv("JIRA_API_TOKEN")
        
        if not all([jira_url, jira_username, jira_token]):
            raise ValueError("Jira credentials not configured. Please set JIRA_URL, JIRA_EMAIL (or JIRA_USERNAME), and JIRA_API_TOKEN")
        
        jira_client = JIRA(
            server=jira_url,
            basic_auth=(jira_username, jira_token)
        )
        
        # Helper function to format Jira description with proper test steps
        def format_jira_description(test_case):
            """Format test case description with test steps using Jira wiki markup"""
            description = test_case.get("description", "")
            test_steps = test_case.get('test_steps', [])
            prerequisites = test_case.get('prerequisites', [])
            expected_results = test_case.get('expected_results', {})
            test_data = test_case.get('test_data', {})
            pass_criteria = test_case.get('pass_criteria', [])
            fail_criteria = test_case.get('fail_criteria', [])
            
            # Build description using Jira wiki markup
            jira_desc_parts = []
            
            # Add main description
            if description:
                jira_desc_parts.append(description)
                jira_desc_parts.append("")
            
            # Add Preconditions/Prerequisites
            if prerequisites:
                if isinstance(prerequisites, list) and prerequisites:
                    jira_desc_parts.append("h3. Preconditions")
                    jira_desc_parts.append("")
                    for prereq in prerequisites:
                        jira_desc_parts.append(f"* {prereq}")
                    jira_desc_parts.append("")
            
            # Add Test Steps (MOST IMPORTANT!)
            if test_steps and isinstance(test_steps, list):
                jira_desc_parts.append("h3. Test Steps")
                jira_desc_parts.append("")
                for step in test_steps:
                    if isinstance(step, dict):
                        step_num = step.get('step_number', '')
                        action = step.get('action', '')
                        input_data = step.get('input_data', '')
                        expected_result = step.get('expected_result', '')
                        
                        # Format as numbered list in Jira
                        jira_desc_parts.append(f"# {action}")
                        
                        if input_data and input_data.strip():
                            jira_desc_parts.append(f"**Input:** {input_data}")
                        
                        if expected_result and expected_result.strip():
                            jira_desc_parts.append(f"**Expected Result:** {expected_result}")
                        
                        jira_desc_parts.append("")
                    elif isinstance(step, str):
                        jira_desc_parts.append(f"# {step}")
                        jira_desc_parts.append("")
                jira_desc_parts.append("")
            
            # Add Expected Results
            if expected_results:
                if isinstance(expected_results, dict):
                    primary = expected_results.get('primary_result', '')
                    criteria = expected_results.get('verification_criteria', [])
                    
                    if primary or criteria:
                        jira_desc_parts.append("h3. Expected Results")
                        jira_desc_parts.append("")
                        
                        if primary:
                            jira_desc_parts.append(f"{primary}")
                            jira_desc_parts.append("")
                        
                        if criteria and isinstance(criteria, list):
                            jira_desc_parts.append("*Verification Criteria:*")
                            for crit in criteria:
                                jira_desc_parts.append(f"* {crit}")
                            jira_desc_parts.append("")
            
            # Add Test Data
            if test_data and isinstance(test_data, dict):
                required_data = test_data.get('required_data', [])
                test_env = test_data.get('test_environment', '')
                cleanup = test_data.get('data_cleanup', '')
                
                if required_data or test_env or cleanup:
                    jira_desc_parts.append("h3. Test Data")
                    jira_desc_parts.append("")
                    
                    if required_data and isinstance(required_data, list):
                        jira_desc_parts.append("*Required Data:*")
                        for item in required_data:
                            jira_desc_parts.append(f"* {item}")
                        jira_desc_parts.append("")
                    
                    if test_env:
                        jira_desc_parts.append(f"*Test Environment:* {test_env}")
                        jira_desc_parts.append("")
                    
                    if cleanup:
                        jira_desc_parts.append(f"*Data Cleanup:* {cleanup}")
                        jira_desc_parts.append("")
            
            # Add Pass Criteria
            if pass_criteria and isinstance(pass_criteria, list):
                jira_desc_parts.append("h3. Pass Criteria")
                jira_desc_parts.append("")
                for crit in pass_criteria:
                    jira_desc_parts.append(f"* {crit}")
                jira_desc_parts.append("")
            
            # Add Fail Criteria
            if fail_criteria and isinstance(fail_criteria, list):
                jira_desc_parts.append("h3. Fail Criteria")
                jira_desc_parts.append("")
                for crit in fail_criteria:
                    jira_desc_parts.append(f"* {crit}")
                jira_desc_parts.append("")
            
            return "\n".join(jira_desc_parts)
        
        exported_cases = []
        for i, test_case in enumerate(test_cases):
            try:
                # Format description with proper test steps
                formatted_description = format_jira_description(test_case)
                
                # Get priority from metadata if not at top level
                priority = test_case.get("priority")
                if not priority:
                    metadata = test_case.get('metadata', {})
                    if isinstance(metadata, dict):
                        priority = metadata.get('priority', 'Medium')
                    else:
                        priority = 'Medium'
                
                # Create Jira issue
                issue_dict = {
                    'project': {'key': project_key},
                    'summary': test_case.get("title", f"Test Case {i + 1}"),
                    'description': formatted_description,
                    'issuetype': {'name': 'Task'},  # Use 'Task' instead of 'Test' for better compatibility
                    'labels': test_case.get("labels", [])
                }
                
                # Add priority (now extracted from metadata if needed)
                try:
                    issue_dict['priority'] = {'name': priority}
                except:
                    # Skip priority if it causes issues
                    pass
                
                # Add parent link (Epic) if provided
                # Note: Newer Jira instances use 'parent' field instead of 'customfield_10014'
                if epic_key:
                    issue_dict['parent'] = {'key': epic_key}
                
                # Create the issue
                new_issue = jira_client.create_issue(fields=issue_dict)
                
                # Build Jira URL for viewing the issue
                jira_base_url = jira_url.rstrip('/')
                jira_issue_url = f"{jira_base_url}/browse/{new_issue.key}"
                
                exported_cases.append({
                    "key": new_issue.key,
                    "summary": new_issue.fields.summary,
                    "description": new_issue.fields.description,
                    "issue_type": "Test",
                    "priority": test_case.get("priority", "Medium"),
                    "labels": test_case.get("labels", []),
                    "epic_link": epic_key,
                    "status": "To Do",
                    "jira_url": jira_issue_url  # ✅ Added Jira URL
                })
                
            except Exception as e:
                logging.error(f"Error creating Jira ticket for test case {i + 1}: {e}")
                continue
        
        # Build summary URLs
        jira_base_url = jira_url.rstrip('/')
        project_url = f"{jira_base_url}/browse/{project_key}"
        epic_url = f"{jira_base_url}/browse/{epic_key}" if epic_key else None
        
        # Build JQL query to view all exported test cases
        exported_keys = [case["key"] for case in exported_cases]
        if exported_keys:
            jql_query = f"key in ({', '.join(exported_keys)})"
            jql_encoded = jql_query.replace(' ', '+').replace(',', '%2C')
            all_test_cases_url = f"{jira_base_url}/issues/?jql={jql_encoded}"
        else:
            all_test_cases_url = None
        
        result = {
            "exported_cases": exported_cases,
            "project_key": project_key,
            "epic_key": epic_key,
            "export_timestamp": datetime.now().isoformat(),
            "total_exported": len(exported_cases),
            "urls": {
                "project": project_url,
                "epic": epic_url,
                "all_test_cases": all_test_cases_url,  # ✅ Jira search URL for all exported cases
                "jira_base": jira_base_url
            }
        }
        
        if traceability_matrix:
            result["traceability_matrix"] = traceability_matrix
        
        logging.info(f"Successfully exported {len(exported_cases)} test cases to Jira")
        logger.info(f"✓ Jira Project: {project_url}")
        if epic_url:
            logger.info(f"✓ Epic: {epic_url}")
        if all_test_cases_url:
            logger.info(f"✓ View All Test Cases: {all_test_cases_url}")
        return result
        
    except Exception as e:
        logging.error(f"Error exporting to Jira: {e}")
        return {"error": f"Failed to export to Jira: {e}"}


# Google Cloud Storage Tools

def export_test_cases_to_csv(test_cases_json: str, output_filename: str = "") -> dict:
    """
    Export test cases to CSV format in the local output folder.
    
    This function accepts either a JSON string OR a file path to a JSON file.
    If a file path is provided, it reads the test cases from the file.
    
    Args:
        test_cases_json (str): JSON string of test cases OR file path to JSON file (e.g., "output/test_cases_20251102_201538.json")
        output_filename (str): Custom filename (auto-generated if empty)
    
    Returns:
        dict: Export result with local file path
              
    Example:
        # From JSON string:
        result = export_test_cases_to_csv('[{"id": "TC-001", ...}]')
        
        # From file path (recommended after parallel generation):
        result = export_test_cases_to_csv('output/test_cases_20251102_201538.json')
        # Returns: {"success": True, "file_path": "output/test_cases_20250102_143022.csv"}
    """
    try:
        # Handle file path if provided instead of JSON string
        if isinstance(test_cases_json, str) and os.path.exists(test_cases_json):
            logger.info(f"Reading test cases from file: {test_cases_json}")
            with open(test_cases_json, 'r', encoding='utf-8') as f:
                test_cases_data = json.load(f)
        # Parse test cases from JSON string
        elif isinstance(test_cases_json, str):
            test_cases_data = json.loads(test_cases_json)
        else:
            test_cases_data = test_cases_json
        
        # Extract test cases list
        if isinstance(test_cases_data, dict):
            test_cases = test_cases_data.get('test_cases', [])
        elif isinstance(test_cases_data, list):
            test_cases = test_cases_data
        else:
            test_cases = []
            
        logging.info(f"Exporting {len(test_cases)} test cases to CSV in output folder")
        
        # Ensure output folder exists
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_filename:
            output_filename = f"test_cases_{timestamp}.csv"
        
        file_path = os.path.join(output_dir, output_filename)
        
        # Helper function to format test steps as readable text
        def format_test_steps_for_csv(test_steps):
            """Format test steps as readable numbered list"""
            if not test_steps:
                return ''
            
            if not isinstance(test_steps, list):
                return str(test_steps)
            
            formatted_steps = []
            for step in test_steps:
                if isinstance(step, dict):
                    step_num = step.get('step_number', '')
                    action = step.get('action', '')
                    input_data = step.get('input_data', '')
                    expected_result = step.get('expected_result', '')
                    
                    step_text = f"Step {step_num}: {action}"
                    if input_data and input_data.strip():
                        step_text += f" | Input: {input_data}"
                    if expected_result and expected_result.strip():
                        step_text += f" | Expected: {expected_result}"
                    
                    formatted_steps.append(step_text)
                elif isinstance(step, str):
                    formatted_steps.append(step)
                else:
                    formatted_steps.append(str(step))
            
            return '\n'.join(formatted_steps)  # Use newlines for multi-line in CSV
        
        # Helper function to format expected results as readable text
        def format_expected_results_for_csv(expected_results):
            """Format expected results as readable text"""
            if not expected_results:
                return ''
            
            if isinstance(expected_results, dict):
                primary = expected_results.get('primary_result', '')
                criteria = expected_results.get('verification_criteria', [])
                
                result_text = primary if primary else ''
                if criteria and isinstance(criteria, list):
                    if result_text:
                        result_text += '\n'
                    result_text += 'Verification Criteria:\n' + '\n'.join([f"• {c}" for c in criteria])
                
                return result_text
            elif isinstance(expected_results, list):
                return '\n'.join([str(r) for r in expected_results])
            else:
                return str(expected_results)
        
        # Helper function to format test data as readable text
        def format_test_data_for_csv(test_data):
            """Format test data as readable text"""
            if not test_data:
                return ''
            
            if isinstance(test_data, dict):
                required_data = test_data.get('required_data', [])
                test_env = test_data.get('test_environment', '')
                cleanup = test_data.get('data_cleanup', '')
                
                parts = []
                if required_data and isinstance(required_data, list):
                    parts.append("Required Data:")
                    parts.extend([f"• {item}" for item in required_data])
                if test_env:
                    parts.append(f"\nTest Environment: {test_env}")
                if cleanup:
                    parts.append(f"\nData Cleanup: {cleanup}")
                
                return '\n'.join(parts) if parts else str(test_data)
            elif isinstance(test_data, list):
                return '\n'.join([str(item) for item in test_data])
            else:
                return str(test_data)
        
        # Create CSV content using pandas
        csv_data = []
        
        for test_case in test_cases:
            if not isinstance(test_case, dict):
                continue
            
            # Format test steps as readable numbered list
            test_steps = test_case.get('test_steps', [])
            test_steps_str = format_test_steps_for_csv(test_steps)
            
            # Format expected results as readable text
            expected_results = test_case.get('expected_results', {})
            expected_results_str = format_expected_results_for_csv(expected_results)
            
            # Format test data as readable text
            test_data = test_case.get('test_data', {})
            test_data_str = format_test_data_for_csv(test_data)
            
            # Format prerequisites
            prerequisites = test_case.get('prerequisites', [])
            if isinstance(prerequisites, list):
                preconditions_str = '\n'.join([f"• {p}" for p in prerequisites]) if prerequisites else 'N/A'
            else:
                preconditions_str = str(prerequisites) if prerequisites else 'N/A'
            
            # Handle compliance standards
            compliance = test_case.get('compliance_standards', [])
            if isinstance(compliance, list):
                compliance_str = ', '.join(compliance)
            else:
                compliance_str = str(compliance) if compliance else ''
            
            # Handle requirement IDs - check multiple possible locations
            req_ids = test_case.get('requirement_ids', test_case.get('linked_requirements', []))
            
            # If not found at top level, check metadata
            if not req_ids:
                metadata = test_case.get('metadata', {})
                if isinstance(metadata, dict):
                    req_id = metadata.get('requirement_id', '')
                    if req_id:
                        req_ids = [req_id]
            
            # Convert to string
            if isinstance(req_ids, list):
                req_ids_str = ', '.join([str(r) for r in req_ids if r])
            else:
                req_ids_str = str(req_ids) if req_ids else ''
            
            # Get priority and category from metadata if not at top level
            priority = test_case.get('priority', 'N/A')
            if priority == 'N/A':
                metadata = test_case.get('metadata', {})
                if isinstance(metadata, dict):
                    priority = metadata.get('priority', 'N/A')
            
            category = test_case.get('category', test_case.get('type', 'N/A'))
            if category == 'N/A':
                metadata = test_case.get('metadata', {})
                if isinstance(metadata, dict):
                    category = metadata.get('test_type', 'N/A')
            
            risk_level = test_case.get('risk_level', 'N/A')
            if risk_level == 'N/A':
                metadata = test_case.get('metadata', {})
                if isinstance(metadata, dict):
                    risk_level = metadata.get('risk_level', 'N/A')
            
            estimated_time = test_case.get('estimated_time', 'N/A')
            if estimated_time == 'N/A':
                metadata = test_case.get('metadata', {})
                if isinstance(metadata, dict):
                    estimated_time = metadata.get('estimated_duration', 'N/A')
            
            csv_data.append({
                'Test Case ID': test_case.get('id', test_case.get('test_case_id', 'N/A')),
                'Title': test_case.get('title', test_case.get('name', 'N/A')),
                'Priority': priority,
                'Category': category,
                'Preconditions': preconditions_str,
                'Test Steps': test_steps_str,
                'Expected Results': expected_results_str,
                'Test Data': test_data_str,
                'Compliance Standards': compliance_str,
                'Requirement IDs': req_ids_str,
                'Risk Level': risk_level,
                'Estimated Time': estimated_time
            })
        
        # Write to CSV using pandas
        import pandas as pd
        df = pd.DataFrame(csv_data)
        # Export with proper handling of multi-line cells
        df.to_csv(file_path, index=False, encoding='utf-8')
        
        logging.info(f"Successfully exported test cases to CSV: {file_path}")
        
        # Upload to GCS
        logger.info("Uploading test cases CSV to GCS...")
        gcs_filename = f"test_cases/{output_filename}"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                csv_content = f.read()
            
            gcs_result = store_in_gcs(csv_content, gcs_filename, "text/csv")
            gcs_path = gcs_result.get('gcs_path', '')
            logger.info(f"✓ Uploaded to GCS: {gcs_path}")
        except Exception as e:
            logger.warning(f"Failed to upload to GCS: {e}")
            gcs_path = "GCS upload failed"
        
        return {
            "success": True,
            "message": f"Test cases exported successfully",
            "file_path": file_path,
            "gcs_path": gcs_path,
            "total_test_cases": len(csv_data),
            "timestamp": timestamp
        }
        
    except Exception as e:
        logging.error(f"Error exporting test cases to CSV: {e}")
        return {"error": f"Failed to export test cases: {e}"}

def _create_test_cases_csv(test_cases):
    """Create CSV content from test cases using pandas for better stability and data handling."""
    import io
    
    try:
        # Validate input
        if not test_cases or not isinstance(test_cases, list):
            logging.warning("No test cases provided or invalid format")
            return ""
        
        # Prepare data for DataFrame with proper data cleaning
        csv_data = []
        
        for i, test_case in enumerate(test_cases):
            if not isinstance(test_case, dict):
                logging.warning(f"Skipping invalid test case at index {i}: not a dictionary")
                continue
                
            # Handle test steps - can be list or string
            test_steps = test_case.get('test_steps', [])
            if isinstance(test_steps, list):
                test_steps_str = '; '.join([str(step) for step in test_steps])
            else:
                test_steps_str = str(test_steps) if test_steps else ''
            
            # Clean and prepare row data
            row_data = {
                'Test Case ID': str(test_case.get('id', f'TC-{i+1:03d}')),
                'Title': str(test_case.get('title', '')).strip(),
                'Description': str(test_case.get('description', '')).strip(),
                'Test Type': str(test_case.get('test_type', '')).strip(),
                'Priority': str(test_case.get('priority', 'Medium')).strip(),
                'Requirement ID': str(test_case.get('requirement_id', '')).strip(),
                'Test Steps': test_steps_str.strip(),
                'Expected Result': str(test_case.get('expected_result', '')).strip(),
                'Compliance Standard': str(test_case.get('compliance_standard', '')).strip(),
                'Test Data': str(test_case.get('test_data', '')).strip(),
                'Prerequisites': str(test_case.get('prerequisites', '')).strip(),
                'Risk Level': str(test_case.get('risk_level', 'Medium')).strip(),
                'Status': 'Draft',
                'Created Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            csv_data.append(row_data)
        
        if not csv_data:
            logging.warning("No valid test cases found to export")
            return ""
        
        # Create DataFrame with pandas
        df = pd.DataFrame(csv_data)
        
        # Convert to CSV string with proper formatting
        output = io.StringIO()
        df.to_csv(
            output, 
            index=False, 
            encoding='utf-8',
            quoting=1,  # Quote all fields for better compatibility
            lineterminator='\n'  # Ensure consistent line endings
        )
        
        csv_content = output.getvalue()
        logging.info(f"Successfully created CSV with {len(df)} test cases")
        return csv_content
        
    except Exception as e:
        logging.error(f"Error creating CSV with pandas: {e}")
        # Fallback to basic CSV creation if pandas fails
        return _create_basic_csv_fallback(test_cases)

def _create_basic_csv_fallback(test_cases):
    """Fallback CSV creation method if pandas fails."""
    import csv
    import io
    
    try:
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        
        # Write header
        writer.writerow([
            'Test Case ID', 'Title', 'Description', 'Test Type', 'Priority',
            'Requirement ID', 'Test Steps', 'Expected Result', 'Compliance Standard',
            'Test Data', 'Prerequisites', 'Risk Level', 'Status', 'Created Date'
        ])
        
        # Write test cases
        for i, test_case in enumerate(test_cases):
            if isinstance(test_case, dict):
                test_steps = test_case.get('test_steps', [])
                if isinstance(test_steps, list):
                    test_steps_str = '; '.join([str(step) for step in test_steps])
                else:
                    test_steps_str = str(test_steps) if test_steps else ''
                
                writer.writerow([
                    str(test_case.get('id', f'TC-{i+1:03d}')),
                    str(test_case.get('title', '')),
                    str(test_case.get('description', '')),
                    str(test_case.get('test_type', '')),
                    str(test_case.get('priority', 'Medium')),
                    str(test_case.get('requirement_id', '')),
                    test_steps_str,
                    str(test_case.get('expected_result', '')),
                    str(test_case.get('compliance_standard', '')),
                    str(test_case.get('test_data', '')),
                    str(test_case.get('prerequisites', '')),
                    str(test_case.get('risk_level', 'Medium')),
                    'Draft',
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ])
        
        return output.getvalue()
        
    except Exception as e:
        logging.error(f"Fallback CSV creation also failed: {e}")
        return "Error creating CSV file"

def _create_comprehensive_test_cases_csv(test_cases):
    """Create comprehensive CSV content from detailed test cases with all fields."""
    import io
    
    try:
        if not test_cases or not isinstance(test_cases, list):
            logging.warning("No test cases provided for comprehensive CSV")
            return ""
        
        csv_data = []
        
        for i, test_case in enumerate(test_cases):
            if not isinstance(test_case, dict):
                logging.warning(f"Skipping invalid test case at index {i}")
                continue
            
            # Extract metadata
            metadata = test_case.get('metadata', {})
            test_data = test_case.get('test_data', {})
            expected_results = test_case.get('expected_results', {})
            compliance_validation = test_case.get('compliance_validation', {})
            
            # Format test steps
            test_steps = test_case.get('test_steps', [])
            formatted_steps = []
            for step in test_steps:
                if isinstance(step, dict):
                    step_text = f"Step {step.get('step_number', '')}: {step.get('action', '')}"
                    if step.get('input_data'):
                        step_text += f" | Input: {step.get('input_data')}"
                    if step.get('expected_result'):
                        step_text += f" | Expected: {step.get('expected_result')}"
                    formatted_steps.append(step_text)
            
            # Create comprehensive row
            row_data = {
                'Test Case ID': str(test_case.get('test_case_id', f'TC-{i+1:03d}')),
                'Title': str(test_case.get('title', '')).strip(),
                'Description': str(test_case.get('description', '')).strip(),
                'Requirement ID': str(metadata.get('requirement_id', '')),
                'Test Type': str(metadata.get('test_type', '')),
                'Priority': str(metadata.get('priority', 'Medium')),
                'Risk Level': str(metadata.get('risk_level', 'Medium')),
                'Software Class': str(metadata.get('software_class', 'B')),
                'Compliance Standards': ', '.join(metadata.get('compliance_standards', [])),
                'Estimated Duration': str(metadata.get('estimated_duration', '')),
                'Automation Feasible': str(metadata.get('automation_feasible', False)),
                'Test Category': str(metadata.get('test_category', '')),
                'Test Steps': ' | '.join(formatted_steps),
                'Prerequisites': '; '.join(test_case.get('prerequisites', [])),
                'Test Environment': str(test_data.get('test_environment', '')),
                'Required Data': '; '.join(test_data.get('required_data', [])),
                'Data Cleanup': str(test_data.get('data_cleanup', '')),
                'Primary Expected Result': str(expected_results.get('primary_result', '')),
                'Verification Criteria': '; '.join(expected_results.get('verification_criteria', [])),
                'Pass Criteria': '; '.join(test_case.get('pass_criteria', [])),
                'Fail Criteria': '; '.join(test_case.get('fail_criteria', [])),
                'Post Conditions': '; '.join(test_case.get('post_conditions', [])),
                'Regulatory Requirements': '; '.join(compliance_validation.get('regulatory_requirements', [])),
                'Validation Evidence': str(compliance_validation.get('validation_evidence', '')),
                'Status': 'Draft',
                'Created Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            csv_data.append(row_data)
        
        if not csv_data:
            return ""
        
        # Create DataFrame with pandas
        df = pd.DataFrame(csv_data)
        
        # Convert to CSV string
        output = io.StringIO()
        df.to_csv(
            output, 
            index=False, 
            encoding='utf-8',
            quoting=1,
            lineterminator='\n'
        )
        
        csv_content = output.getvalue()
        logging.info(f"Successfully created comprehensive CSV with {len(df)} test cases")
        return csv_content
        
    except Exception as e:
        logging.error(f"Error creating comprehensive CSV: {e}")
        return _create_basic_csv_fallback(test_cases)

def _create_test_cases_pdf(test_cases):
    """Create professional PDF document for test cases using ReportLab."""
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    import io
    
    try:
        if not test_cases:
            return b""
        
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=12,
            textColor=colors.darkgreen
        )
        
        # Build story
        story = []
        
        # Title page
        story.append(Paragraph("Healthcare Test Cases - Detailed Specification", title_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"Total Test Cases: {len(test_cases)}", styles['Normal']))
        story.append(PageBreak())
        
        # Process each test case
        for i, test_case in enumerate(test_cases):
            if i > 0:
                story.append(PageBreak())
            
            # Test case header
            tc_id = test_case.get('test_case_id', f'TC-{i+1:03d}')
            title = test_case.get('title', 'Untitled Test Case')
            
            story.append(Paragraph(f"TEST CASE: {tc_id}", title_style))
            story.append(Paragraph(f"Title: {title}", heading_style))
            
            # Metadata table
            metadata = test_case.get('metadata', {})
            meta_data = [
                ['Priority', metadata.get('priority', 'Medium')],
                ['Type', metadata.get('test_type', 'Functional')],
                ['Risk Level', metadata.get('risk_level', 'Medium')],
                ['Requirement ID', metadata.get('requirement_id', 'N/A')],
                ['Compliance', ', '.join(metadata.get('compliance_standards', []))],
                ['Software Class', metadata.get('software_class', 'B')],
                ['Duration', metadata.get('estimated_duration', 'N/A')]
            ]
            
            meta_table = Table(meta_data, colWidths=[2*inch, 3*inch])
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 12))
            
            # Test objective
            story.append(Paragraph("TEST OBJECTIVE:", heading_style))
            description = test_case.get('description', 'No description provided')
            story.append(Paragraph(description, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Prerequisites
            prerequisites = test_case.get('prerequisites', [])
            if prerequisites:
                story.append(Paragraph("PREREQUISITES:", heading_style))
                for prereq in prerequisites:
                    story.append(Paragraph(f"✓ {prereq}", styles['Normal']))
                story.append(Spacer(1, 12))
            
            # Test steps
            test_steps = test_case.get('test_steps', [])
            if test_steps:
                story.append(Paragraph("TEST STEPS:", heading_style))
                
                steps_data = [['#', 'Action & Expected Result']]
                for step in test_steps:
                    if isinstance(step, dict):
                        step_num = str(step.get('step_number', ''))
                        action = step.get('action', '')
                        expected = step.get('expected_result', '')
                        input_data = step.get('input_data', '')
                        
                        step_text = f"{action}"
                        if input_data:
                            step_text += f"\nInput: {input_data}"
                        if expected:
                            step_text += f"\n→ Expected: {expected}"
                        
                        steps_data.append([step_num, step_text])
                
                steps_table = Table(steps_data, colWidths=[0.5*inch, 4.5*inch])
                steps_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP')
                ]))
                story.append(steps_table)
                story.append(Spacer(1, 12))
            
            # Pass/Fail criteria
            pass_criteria = test_case.get('pass_criteria', [])
            fail_criteria = test_case.get('fail_criteria', [])
            
            if pass_criteria or fail_criteria:
                criteria_data = []
                if pass_criteria:
                    criteria_data.append(['PASS CRITERIA', ''])
                    for criteria in pass_criteria:
                        criteria_data.append(['✓', criteria])
                
                if fail_criteria:
                    if criteria_data:
                        criteria_data.append(['', ''])  # Spacer
                    criteria_data.append(['FAIL CRITERIA', ''])
                    for criteria in fail_criteria:
                        criteria_data.append(['✗', criteria])
                
                criteria_table = Table(criteria_data, colWidths=[0.5*inch, 4.5*inch])
                criteria_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                ]))
                story.append(criteria_table)
                story.append(Spacer(1, 12))
            
            # Compliance validation
            compliance_validation = test_case.get('compliance_validation', {})
            regulatory_reqs = compliance_validation.get('regulatory_requirements', [])
            if regulatory_reqs:
                story.append(Paragraph("COMPLIANCE EVIDENCE:", heading_style))
                for req in regulatory_reqs:
                    story.append(Paragraph(f"• {req} ✓", styles['Normal']))
                story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        logging.info(f"Successfully created test cases PDF with {len(test_cases)} test cases")
        return pdf_content
        
    except Exception as e:
        logging.error(f"Error creating test cases PDF: {e}")
        return b""

def _format_test_cases_with_llm_prompt(test_cases):
    """Format test cases using the TEST_CASE_PDF_PROMPT for consistent formatting."""
    try:
        # Import the prompt
        try:
            from .prompts import TEST_CASE_PDF_PROMPT
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location("prompts", os.path.join(os.path.dirname(__file__), "prompts.py"))
            prompts_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(prompts_module)
            TEST_CASE_PDF_PROMPT = prompts_module.TEST_CASE_PDF_PROMPT
        
        # Initialize GenAI client
        client = init_genai_client()
        if not client:
            return "Error: Could not initialize GenAI client for PDF formatting"
        
        # Create prompt with test cases
        prompt = f"""
        {TEST_CASE_PDF_PROMPT}
        
        Format the following test cases according to the template above:
        
        {json.dumps(test_cases, indent=2)}
        
        Return ONLY the formatted text content for the PDF, following the exact template format.
        Use proper spacing, alignment, and the specified symbols (✓, ✗, ├──, └──, etc.).
        """
        
        # Generate formatted content
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="text/plain"
            )
        )
        
        return response.candidates[0].content.parts[0].text
        
    except Exception as e:
        logging.error(f"Error formatting test cases with LLM: {e}")
        return f"Error formatting test cases: {e}"

def _create_test_cases_pdf_with_prompt_format(test_cases):
    """Create PDF using TEST_CASE_PDF_PROMPT format with LLM formatting."""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    import io
    
    try:
        if not test_cases:
            return b""
        
        # Get LLM-formatted content using TEST_CASE_PDF_PROMPT
        formatted_content = _format_test_cases_with_llm_prompt(test_cases)
        
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1*cm, leftMargin=1*cm, topMargin=1.5*cm, bottomMargin=1*cm)
        
        # Simple styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, spaceAfter=20, textColor=colors.darkblue, alignment=TA_CENTER)
        normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=10, spaceAfter=6, alignment=TA_LEFT)
        
        # Build story with formatted content
        story = []
        story.append(Paragraph("Healthcare Test Cases - TEST_CASE_PDF_PROMPT Format", title_style))
        story.append(Spacer(1, 20))
        
        # Convert formatted content to paragraphs
        for line in formatted_content.split('\n'):
            if line.strip():
                story.append(Paragraph(line, normal_style))
            else:
                story.append(Spacer(1, 6))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        logging.info(f"Successfully created TEST_CASE_PDF_PROMPT formatted PDF with {len(test_cases)} test cases")
        return pdf_content
        
    except Exception as e:
        logging.error(f"Error creating TEST_CASE_PDF_PROMPT formatted PDF: {e}")
        return b""

def _create_test_cases_pdf_improved(test_cases):
    """Create professional PDF document for test cases using TEST_CASE_PDF_PROMPT format."""
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    import io
    
    try:
        if not test_cases:
            return b""
        
        # First, use LLM to format test cases according to TEST_CASE_PDF_PROMPT
        formatted_content = _format_test_cases_with_llm_prompt(test_cases)
        
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            rightMargin=1*cm, 
            leftMargin=1*cm, 
            topMargin=1.5*cm, 
            bottomMargin=1*cm
        )
        
        # Enhanced styles
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.darkblue,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        )
        
        subheading_style = ParagraphStyle(
            'SubHeading',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=8,
            textColor=colors.black,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'BodyText',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        )
        
        # Build story using LLM-formatted content
        story = []
        
        # Title page
        story.append(Paragraph("Healthcare Test Cases", title_style))
        story.append(Paragraph("Generated using TEST_CASE_PDF_PROMPT Format", heading_style))
        story.append(Spacer(1, 30))
        
        # Add the LLM-formatted content as the main body
        # Convert the formatted text to PDF paragraphs
        content_lines = formatted_content.split('\n')
        for line in content_lines:
            if line.strip():
                # Detect different types of content and apply appropriate styling
                if line.startswith('TEST CASE:'):
                    story.append(Spacer(1, 20))
                    story.append(Paragraph(line, title_style))
                elif line.startswith('Title:') or line.startswith('Priority:'):
                    story.append(Paragraph(line, heading_style))
                elif line.startswith('REQUIREMENT TRACEABILITY:') or line.startswith('TEST OBJECTIVE:') or \
                     line.startswith('PREREQUISITES:') or line.startswith('TEST STEPS:') or \
                     line.startswith('PASS CRITERIA:') or line.startswith('FAIL CRITERIA:') or \
                     line.startswith('COMPLIANCE EVIDENCE:'):
                    story.append(Spacer(1, 10))
                    story.append(Paragraph(f"<b>{line}</b>", subheading_style))
                else:
                    # Regular content
                    story.append(Paragraph(line, body_style))
            else:
                story.append(Spacer(1, 6))
        
        # Add a simple summary at the end
        story.append(Spacer(1, 30))
        story.append(Paragraph(f"<b>Document Summary</b>", heading_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
        story.append(Paragraph(f"Total Test Cases: {len(test_cases)}", body_style))
        story.append(Paragraph(f"Format: TEST_CASE_PDF_PROMPT Compliant", body_style))
        story.append(PageBreak())
        
        # Process each test case
        for i, test_case in enumerate(test_cases):
            # Keep test case together on same page when possible
            test_case_content = []
            
            # Test case header
            tc_id = test_case.get('test_case_id', f'TC-{i+1:03d}')
            title = test_case.get('title', 'Untitled Test Case')
            
            test_case_content.append(Paragraph(f"TEST CASE: {tc_id}", title_style))
            test_case_content.append(Paragraph(title, heading_style))
            
            # Metadata table with improved layout
            metadata = test_case.get('metadata', {})
            meta_data = [
                ['Field', 'Value'],
                ['Priority', metadata.get('priority', 'Medium')],
                ['Test Type', metadata.get('test_type', 'Functional')],
                ['Risk Level', metadata.get('risk_level', 'Medium')],
                ['Requirement ID', metadata.get('requirement_id', 'N/A')],
                ['Compliance Standards', ', '.join(metadata.get('compliance_standards', []))],
                ['Software Class', metadata.get('software_class', 'B')],
                ['Estimated Duration', metadata.get('estimated_duration', 'N/A')],
                ['Automation Feasible', str(metadata.get('automation_feasible', False))]
            ]
            
            meta_table = Table(meta_data, colWidths=[4*cm, 8*cm])
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            test_case_content.append(meta_table)
            test_case_content.append(Spacer(1, 15))
            
            # Test objective
            test_case_content.append(Paragraph("TEST OBJECTIVE", subheading_style))
            description = test_case.get('description', 'No description provided')
            test_case_content.append(Paragraph(description, body_style))
            test_case_content.append(Spacer(1, 10))
            
            # Prerequisites
            prerequisites = test_case.get('prerequisites', [])
            if prerequisites:
                test_case_content.append(Paragraph("PREREQUISITES", subheading_style))
                for j, prereq in enumerate(prerequisites, 1):
                    test_case_content.append(Paragraph(f"{j}. {prereq}", body_style))
                test_case_content.append(Spacer(1, 10))
            
            # Test steps with improved formatting
            test_steps = test_case.get('test_steps', [])
            if test_steps:
                test_case_content.append(Paragraph("TEST STEPS", subheading_style))
                
                for step in test_steps:
                    if isinstance(step, dict):
                        step_num = step.get('step_number', '')
                        action = step.get('action', '')
                        input_data = step.get('input_data', '')
                        expected = step.get('expected_result', '')
                        
                        # Step header
                        test_case_content.append(Paragraph(f"<b>Step {step_num}:</b> {action}", body_style))
                        
                        if input_data and input_data.strip():
                            test_case_content.append(Paragraph(f"<b>Input:</b> {input_data}", body_style))
                        
                        if expected and expected.strip():
                            test_case_content.append(Paragraph(f"<b>Expected Result:</b> {expected}", body_style))
                        
                        test_case_content.append(Spacer(1, 8))
            
            # Pass/Fail criteria
            pass_criteria = test_case.get('pass_criteria', [])
            fail_criteria = test_case.get('fail_criteria', [])
            
            if pass_criteria:
                test_case_content.append(Paragraph("PASS CRITERIA", subheading_style))
                for j, criteria in enumerate(pass_criteria, 1):
                    test_case_content.append(Paragraph(f"✓ {criteria}", body_style))
                test_case_content.append(Spacer(1, 8))
            
            if fail_criteria:
                test_case_content.append(Paragraph("FAIL CRITERIA", subheading_style))
                for j, criteria in enumerate(fail_criteria, 1):
                    test_case_content.append(Paragraph(f"✗ {criteria}", body_style))
                test_case_content.append(Spacer(1, 8))
            
            # Compliance validation
            compliance_validation = test_case.get('compliance_validation', {})
            regulatory_reqs = compliance_validation.get('regulatory_requirements', [])
            if regulatory_reqs:
                test_case_content.append(Paragraph("COMPLIANCE EVIDENCE", subheading_style))
                for req in regulatory_reqs:
                    test_case_content.append(Paragraph(f"• {req}", body_style))
                test_case_content.append(Spacer(1, 10))
            
            # Add test case content to story
            story.extend(test_case_content)
            
            # Add page break between test cases (except for the last one)
            if i < len(test_cases) - 1:
                story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
        
        # Get PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        logging.info(f"Successfully created improved test cases PDF with {len(test_cases)} test cases")
        return pdf_content
        
    except Exception as e:
        logging.error(f"Error creating improved test cases PDF: {e}")
        return b""

def _create_traceability_matrix_from_test_cases(requirements, test_cases):
    """Create traceability matrix from requirements and test cases."""
    try:
        traceability_matrix = {
            "matrix_info": {
                "generated_date": datetime.now().isoformat(),
                "total_requirements": len(requirements),
                "total_test_cases": len(test_cases),
                "coverage_percentage": 0.0
            },
            "traceability_links": [],
            "coverage_summary": {
                "covered_requirements": 0,
                "uncovered_requirements": 0,
                "requirements_with_multiple_tests": 0
            }
        }
        
        # Create mapping of requirements to test cases
        req_to_tests = {}
        for req in requirements:
            req_id = req.get("id", "")
            req_to_tests[req_id] = {
                "requirement": req,
                "test_cases": []
            }
        
        # Map test cases to requirements
        for tc in test_cases:
            req_id = tc.get("metadata", {}).get("requirement_id") or tc.get("requirement_id")
            if req_id and req_id in req_to_tests:
                req_to_tests[req_id]["test_cases"].append(tc)
        
        # Build traceability links
        covered_count = 0
        multiple_tests_count = 0
        
        for req_id, data in req_to_tests.items():
            test_case_ids = [tc.get("test_case_id", "") for tc in data["test_cases"]]
            
            link = {
                "requirement_id": req_id,
                "requirement_title": data["requirement"].get("title", ""),
                "requirement_type": data["requirement"].get("type", ""),
                "requirement_priority": data["requirement"].get("priority", ""),
                "test_case_ids": test_case_ids,
                "test_case_count": len(test_case_ids),
                "coverage_status": "covered" if test_case_ids else "uncovered"
            }
            
            traceability_matrix["traceability_links"].append(link)
            
            if test_case_ids:
                covered_count += 1
                if len(test_case_ids) > 1:
                    multiple_tests_count += 1
        
        # Update summary
        traceability_matrix["coverage_summary"]["covered_requirements"] = covered_count
        traceability_matrix["coverage_summary"]["uncovered_requirements"] = len(requirements) - covered_count
        traceability_matrix["coverage_summary"]["requirements_with_multiple_tests"] = multiple_tests_count
        traceability_matrix["matrix_info"]["coverage_percentage"] = (covered_count / len(requirements) * 100) if requirements else 0.0
        
        return traceability_matrix
        
    except Exception as e:
        logging.error(f"Error creating traceability matrix: {e}")
        return {"error": str(e)}

def _create_traceability_matrix_pdf_improved(traceability_matrix):
    """Create professional PDF document for traceability matrix with improved formatting."""
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    import io
    
    try:
        if not traceability_matrix or "error" in traceability_matrix:
            return b""
        
        # Create PDF buffer - use landscape for better table layout
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=landscape(A4), 
            rightMargin=1*cm, 
            leftMargin=1*cm, 
            topMargin=1.5*cm, 
            bottomMargin=1*cm
        )
        
        # Enhanced styles
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.darkblue,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        )
        
        # Build story
        story = []
        
        # Title page
        story.append(Paragraph("Requirements Traceability Matrix", title_style))
        story.append(Paragraph("Healthcare Test Case Coverage Report", heading_style))
        story.append(Spacer(1, 30))
        
        # Summary information
        matrix_info = traceability_matrix.get("matrix_info", {})
        coverage_summary = traceability_matrix.get("coverage_summary", {})
        
        summary_data = [
            ['Traceability Summary', ''],
            ['Generated Date', matrix_info.get("generated_date", "N/A")[:19]],
            ['Total Requirements', str(matrix_info.get("total_requirements", 0))],
            ['Total Test Cases', str(matrix_info.get("total_test_cases", 0))],
            ['Covered Requirements', str(coverage_summary.get("covered_requirements", 0))],
            ['Uncovered Requirements', str(coverage_summary.get("uncovered_requirements", 0))],
            ['Coverage Percentage', f"{matrix_info.get('coverage_percentage', 0):.1f}%"],
            ['Requirements with Multiple Tests', str(coverage_summary.get("requirements_with_multiple_tests", 0))]
        ]
        
        summary_table = Table(summary_data, colWidths=[6*cm, 4*cm])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (0, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 30))
        
        # Traceability matrix table
        story.append(Paragraph("Detailed Traceability Matrix", heading_style))
        
        # Prepare table data
        table_data = [
            ['Requirement ID', 'Title', 'Type', 'Priority', 'Test Cases', 'Coverage']
        ]
        
        traceability_links = traceability_matrix.get("traceability_links", [])
        for link in traceability_links:
            test_case_list = ', '.join(link.get("test_case_ids", []))
            if len(test_case_list) > 50:  # Truncate long lists
                test_case_list = test_case_list[:47] + "..."
            
            coverage_status = link.get("coverage_status", "uncovered")
            coverage_icon = "✓" if coverage_status == "covered" else "✗"
            
            table_data.append([
                link.get("requirement_id", ""),
                link.get("requirement_title", "")[:40] + ("..." if len(link.get("requirement_title", "")) > 40 else ""),
                link.get("requirement_type", ""),
                link.get("requirement_priority", ""),
                test_case_list,
                f"{coverage_icon} ({link.get('test_case_count', 0)})"
            ])
        
        # Create table with appropriate column widths
        matrix_table = Table(table_data, colWidths=[3*cm, 6*cm, 2.5*cm, 2*cm, 8*cm, 2.5*cm])
        matrix_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            
            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            
            # Grid and alternating colors
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            
            # Coverage column alignment
            ('ALIGN', (-1, 1), (-1, -1), 'CENTER'),
        ]))
        
        story.append(matrix_table)
        
        # Build PDF
        doc.build(story)
        
        # Get PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        logging.info(f"Successfully created traceability matrix PDF")
        return pdf_content
        
    except Exception as e:
        logging.error(f"Error creating traceability matrix PDF: {e}")
        return b""

# USER-CONTROLLED TEST CASE GENERATION WORKFLOW

def list_extracted_requirements(requirements_json: str):
    """
    List extracted requirements for user selection and review.
    
    This function parses and displays requirements in a user-friendly format,
    allowing users to see what requirements are available and select which ones
    to generate test cases for.
    
    Args:
        requirements_json (str): JSON string of extracted requirements
    
    Returns:
        dict: Dictionary containing formatted requirements list and summary
    """
    try:
        logging.info("Listing extracted requirements for user selection")
        
        # Parse requirements
        if isinstance(requirements_json, str):
            requirements_data = json.loads(requirements_json)
        else:
            requirements_data = requirements_json
            
        # Extract requirements list
        if isinstance(requirements_data, dict) and "requirements" in requirements_data:
            requirements = requirements_data["requirements"]
        elif isinstance(requirements_data, list):
            requirements = requirements_data
        else:
            requirements = [requirements_data]
        
        # Format requirements for display
        formatted_requirements = []
        for i, req in enumerate(requirements):
            formatted_req = {
                "index": i + 1,
                "id": req.get("id", f"REQ-{i+1:03d}"),
                "title": req.get("title", "Untitled Requirement"),
                "type": req.get("type", "Unknown"),
                "priority": req.get("priority", "Medium"),
                "description": req.get("description", "No description")[:200] + ("..." if len(req.get("description", "")) > 200 else ""),
                "compliance_standards": req.get("compliance_standards", []),
                "risk_level": req.get("risk_level", "Medium")
            }
            formatted_requirements.append(formatted_req)
        
        # Create summary
        summary = {
            "total_requirements": len(requirements),
            "by_type": {},
            "by_priority": {},
            "by_risk_level": {}
        }
        
        for req in requirements:
            req_type = req.get("type", "Unknown")
            priority = req.get("priority", "Medium")
            risk_level = req.get("risk_level", "Medium")
            
            summary["by_type"][req_type] = summary["by_type"].get(req_type, 0) + 1
            summary["by_priority"][priority] = summary["by_priority"].get(priority, 0) + 1
            summary["by_risk_level"][risk_level] = summary["by_risk_level"].get(risk_level, 0) + 1
        
        result = {
            "requirements": formatted_requirements,
            "summary": summary,
            "message": f"Found {len(requirements)} requirements. Select which ones to generate test cases for.",
            "selection_instructions": "Use generate_test_cases_for_requirements() with specific requirement IDs to generate test cases."
        }
        
        logging.info(f"Successfully listed {len(requirements)} requirements for user selection")
        return result
        
    except Exception as e:
        logging.error(f"Error listing requirements: {e}")
        return {"error": f"Failed to list requirements: {e}"}

def _clean_requirements_json(requirements_json: str):
    """Simple function to clean and validate requirements JSON."""
    try:
        # Parse and re-serialize to clean up any formatting issues
        if isinstance(requirements_json, str):
            data = json.loads(requirements_json)
        else:
            data = requirements_json
        
        # Ensure proper structure
        if isinstance(data, dict) and "requirements" in data:
            requirements = data["requirements"]
        elif isinstance(data, list):
            requirements = data
        else:
            requirements = [data]
        
        # Basic cleanup for each requirement
        cleaned_requirements = []
        for req in requirements:
            if isinstance(req, dict):
                # Ensure basic required fields exist
                cleaned_req = {
                    "id": req.get("id", "REQ-001"),
                    "title": req.get("title", "Requirement"),
                    "type": req.get("type", "functional"),
                    "priority": req.get("priority", "medium"),
                    "description": req.get("description", "Requirement description"),
                    "risk_level": req.get("risk_level", "medium"),
                    "compliance_standards": req.get("compliance_standards", []),
                    "acceptance_criteria": req.get("acceptance_criteria", [])
                }
                cleaned_requirements.append(cleaned_req)
        
        return {"requirements": cleaned_requirements}
        
    except Exception as e:
        logging.error(f"Error cleaning requirements JSON: {e}")
        return {"error": f"Failed to clean requirements JSON: {e}"}

def generate_test_cases_batch(requirements_json: str, batch_requirement_ids: str, batch_number: int, test_types: str = "functional,security,compliance", standards: str = "FDA,HIPAA,IEC 62304", risk_level: str = "medium") -> dict:
    """
    Generate test cases for a batch of requirements (designed for parallel execution).
    
    This function is used by ParallelAgent to generate test cases concurrently for different requirement batches.
    Each parallel agent processes one batch independently.
    
    Args:
        requirements_json (str): JSON string of all requirements
        batch_requirement_ids (str): Comma-separated requirement IDs for this batch (e.g., "MED-110,MED-111,MED-112")
        batch_number (int): Batch number (for logging and identification)
        test_types (str): Comma-separated test types
        standards (str): Comma-separated compliance standards
        risk_level (str): Risk level for test cases
    
    Returns:
        dict: Dictionary containing generated test cases for this batch with keys:
            - batch_number (int): Batch identifier
            - batch_requirement_ids (list): Requirement IDs processed in this batch
            - test_cases (list): Generated test cases for this batch
            - batch_status (str): "success" or "error"
            - error (str): Error message if generation failed
    """
    try:
        logger.info("="*70)
        logger.info(f"📦 BATCH {batch_number}: Generating Test Cases")
        logger.info(f"   Requirement IDs: {batch_requirement_ids}")
        logger.info("="*70)
        
        # Use existing function but return batch-specific format
        result = generate_test_cases_for_requirements(
            requirements_json=requirements_json,
            selected_requirement_ids=batch_requirement_ids,
            test_types=test_types,
            standards=standards,
            risk_level=risk_level
        )
        
        if "error" in result:
            logger.error(f"✗ Batch {batch_number} failed: {result['error']}")
            return {
                "batch_number": batch_number,
                "batch_requirement_ids": [req_id.strip() for req_id in batch_requirement_ids.split(',')],
                "test_cases": [],
                "batch_status": "error",
                "error": result["error"]
            }
        
        # Extract test cases from result
        test_cases = result.get("test_cases", [])
        logger.info(f"✓ Batch {batch_number} complete: Generated {len(test_cases)} test cases")
        logger.info("="*70)
        
        return {
            "batch_number": batch_number,
            "batch_requirement_ids": [req_id.strip() for req_id in batch_requirement_ids.split(',')],
            "test_cases": test_cases,
            "batch_status": "success",
            "requirements_covered": result.get("requirements_covered", len(test_cases)),
            "generation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"✗ Batch {batch_number} exception: {e}")
        logger.exception("Full traceback:")
        return {
            "batch_number": batch_number,
            "batch_requirement_ids": [req_id.strip() for req_id in batch_requirement_ids.split(',')],
            "test_cases": [],
            "batch_status": "error",
            "error": str(e)
        }

def split_requirements_into_batches(requirements_json: str, batch_size: int = 10) -> dict:
    """
    Split requirements into batches for parallel processing.
    
    This function prepares requirements for parallel test case generation by
    splitting them into manageable batches.
    
    Args:
        requirements_json (str): JSON string of all requirements
        batch_size (int): Number of requirements per batch (default: 10)
    
    Returns:
        dict: Dictionary with batches and metadata:
            - total_requirements (int): Total number of requirements
            - batch_size (int): Number of requirements per batch
            - total_batches (int): Total number of batches created
            - batches (list): List of batch dictionaries with:
                - batch_number (int): Batch identifier (1-based)
                - requirement_ids (list): Requirement IDs in this batch
                - requirement_ids_str (str): Comma-separated IDs for function calls
    """
    try:
        logger.info("="*70)
        logger.info(f"📊 SPLITTING REQUIREMENTS INTO BATCHES")
        logger.info(f"   Batch Size: {batch_size}")
        logger.info("="*70)
        
        # Parse requirements
        if isinstance(requirements_json, str):
            requirements_data = json.loads(requirements_json)
        else:
            requirements_data = requirements_json
        
        # Extract requirements list
        if isinstance(requirements_data, dict) and "requirements" in requirements_data:
            all_requirements = requirements_data["requirements"]
        elif isinstance(requirements_data, list):
            all_requirements = requirements_data
        else:
            all_requirements = [requirements_data]
        
        total_requirements = len(all_requirements)
        logger.info(f"✓ Found {total_requirements} total requirements")
        
        # Split into batches
        batches = []
        total_batches = (total_requirements + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_requirements)
            batch_requirements = all_requirements[start_idx:end_idx]
            
            # Extract requirement IDs
            requirement_ids = []
            for req in batch_requirements:
                req_id = req.get("id") or req.get("key") or f"REQ-{start_idx + len(requirement_ids) + 1}"
                requirement_ids.append(req_id)
            
            batches.append({
                "batch_number": batch_num + 1,
                "requirement_ids": requirement_ids,
                "requirement_ids_str": ",".join(requirement_ids),
                "requirement_count": len(requirement_ids)
            })
            
            logger.info(f"   Batch {batch_num + 1}/{total_batches}: {len(requirement_ids)} requirements ({requirement_ids[0]} to {requirement_ids[-1]})")
        
        logger.info("="*70)
        logger.info(f"✓ BATCH SPLITTING COMPLETE")
        logger.info(f"   Total Requirements: {total_requirements}")
        logger.info(f"   Batch Size: {batch_size}")
        logger.info(f"   Total Batches: {total_batches}")
        logger.info("="*70)
        
        return {
            "total_requirements": total_requirements,
            "batch_size": batch_size,
            "total_batches": total_batches,
            "batches": batches,
            "requirements_json": requirements_json  # Return original for use in parallel agents
        }
        
    except Exception as e:
        logger.error(f"✗ Failed to split requirements: {e}")
        logger.exception("Full traceback:")
        return {"error": f"Failed to split requirements into batches: {e}"}

def generate_test_cases_parallel_batches(requirements_json: str, batch_size: int = 10, test_types: str = "functional,security,compliance", standards: str = "FDA,HIPAA,IEC 62304", risk_level: str = "medium") -> dict:
    """
    Generate test cases using parallel batch processing (PRODUCTION-GRADE).
    
    This function orchestrates the complete parallel test case generation workflow:
    1. Splits requirements into batches
    2. Processes batches in parallel (concurrent execution)
    3. Consolidates all results
    
    This is the main function to use after test plan approval for efficient,
    scalable test case generation.
    
    Args:
        requirements_json (str): JSON string of all requirements
        batch_size (int): Number of requirements per batch (default: 10)
        test_types (str): Comma-separated test types
        standards (str): Comma-separated compliance standards
        risk_level (str): Risk level for test cases
    
    Returns:
        dict: Consolidated test cases from all parallel batches
    """
    try:
        logger.info("="*70)
        logger.info("⚡ PARALLEL TEST CASE GENERATION (BATCH PROCESSING)")
        logger.info(f"   Batch Size: {batch_size}")
        logger.info("="*70)
        
        # Step 1: Split requirements into batches
        split_result = split_requirements_into_batches(requirements_json, batch_size)
        if "error" in split_result:
            return split_result
        
        batches = split_result["batches"]
        total_batches = split_result["total_batches"]
        requirements_json = split_result["requirements_json"]
        
        logger.info(f"✓ Split into {total_batches} batches for parallel processing")
        
        # Step 2: Process batches in parallel using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        batch_results = []
        failed_batches = []
        
        # Process all batches in parallel (up to total_batches concurrent)
        with ThreadPoolExecutor(max_workers=min(total_batches, 5)) as executor:  # Limit to 5 concurrent to avoid overwhelming
            # Submit all batch processing tasks
            future_to_batch = {
                executor.submit(
                    generate_test_cases_batch,
                    requirements_json,
                    batch["requirement_ids_str"],
                    batch["batch_number"],
                    test_types,
                    standards,
                    risk_level
                ): batch["batch_number"]
                for batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    batch_result = future.result()
                    batch_results.append(batch_result)
                    
                    if batch_result.get("batch_status") == "success":
                        logger.info(f"✓ Batch {batch_num} completed successfully")
                    else:
                        logger.error(f"✗ Batch {batch_num} failed: {batch_result.get('error', 'Unknown')}")
                        failed_batches.append(batch_num)
                except Exception as e:
                    logger.error(f"✗ Batch {batch_num} exception: {e}")
                    failed_batches.append(batch_num)
        
        # Step 3: Consolidate all batch results
        logger.info("="*70)
        logger.info(f"📊 PARALLEL PROCESSING COMPLETE")
        logger.info(f"   Successful Batches: {len(batch_results) - len(failed_batches)}/{total_batches}")
        if failed_batches:
            logger.warning(f"   ⚠ Failed Batches: {failed_batches}")
        logger.info("="*70)
        
        consolidated = consolidate_parallel_test_cases(batch_results)
        
        # Add parallel processing metadata
        consolidated["parallel_processing_info"] = {
            "batch_size": batch_size,
            "total_batches": total_batches,
            "successful_batches": len(batch_results) - len(failed_batches),
            "failed_batches": len(failed_batches),
            "failed_batch_numbers": failed_batches,
            "processing_method": "parallel_threadpool"
        }
        
        logger.info("="*70)
        logger.info("✅ PARALLEL TEST CASE GENERATION COMPLETE")
        logger.info(f"   Total Test Cases: {consolidated.get('total_test_cases', 0)}")
        logger.info(f"   Requirements Covered: {consolidated.get('requirements_covered', 0)}")
        logger.info(f"   Success Rate: {consolidated.get('coverage_summary', {}).get('success_rate', 'N/A')}")
        logger.info("="*70)
        
        # Save test cases to file to avoid large JSON in function calls
        # This prevents malformed function calls when passing to validate_test_case_quality
        test_cases = consolidated.get("test_cases", [])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_cases_file = os.path.join("output", f"test_cases_{timestamp}.json")
        
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        
        # Save full test cases to file
        with open(test_cases_file, 'w', encoding='utf-8') as f:
            json.dump({"test_cases": test_cases}, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Test cases saved to: {test_cases_file}")
        logger.info(f"   File size: {os.path.getsize(test_cases_file)} bytes")
        
        # Return consolidated results with FILE REFERENCE instead of full test cases
        # This prevents malformed function calls - agent should read from file if needed
        result = {
            "test_cases_file": test_cases_file,  # File path instead of full JSON
            "total_test_cases": consolidated.get("total_test_cases", 0),
            "batches_processed": consolidated.get("batches_processed", 0),
            "batches_failed": consolidated.get("batches_failed", 0),
            "requirements_covered": consolidated.get("requirements_covered", 0),
            "coverage_summary": consolidated.get("coverage_summary", {}),
            "parallel_processing_info": consolidated.get("parallel_processing_info", {}),
            "consolidated_timestamp": consolidated.get("consolidated_timestamp"),
            "message": consolidated.get("message", ""),
            # Include minimal requirement metadata (not full JSON) for traceability
            "requirement_summary": {
                "total_requirements": split_result.get("total_requirements", 0),
                "requirement_ids": [req_id for batch in batches for req_id in batch.get("requirement_ids", [])][:20]  # First 20 IDs only
            },
            # Include minimal test case summary (first 3 test cases only) for reference
            "test_cases_sample": test_cases[:3] if len(test_cases) > 3 else test_cases,
            "note": f"Full test cases saved to {test_cases_file}. Use validate_test_case_quality_from_file() to validate, or read file for test_cases_json parameter."
        }
        
        return result
        
    except Exception as e:
        logger.error("="*70)
        logger.error("✗ PARALLEL TEST CASE GENERATION FAILED")
        logger.error(f"   Error: {str(e)}")
        logger.error("="*70)
        logger.exception("Full traceback:")
        return {"error": f"Failed to generate test cases in parallel batches: {e}"}

def consolidate_parallel_test_cases(batch_results: list) -> dict:
    """
    Consolidate test cases from multiple parallel batch results.
    
    This function merges test cases generated by parallel agents into a single
    consolidated result set.
    
    Args:
        batch_results (list): List of batch result dictionaries from parallel execution
    
    Returns:
        dict: Consolidated test cases with:
            - test_cases (list): All test cases from all batches
            - total_test_cases (int): Total count
            - batches_processed (int): Number of batches successfully processed
            - batches_failed (int): Number of batches that failed
            - coverage_summary (dict): Coverage statistics
            - consolidated_timestamp (str): ISO timestamp
    """
    try:
        logger.info("="*70)
        logger.info("🔗 CONSOLIDATING PARALLEL TEST CASE RESULTS")
        logger.info(f"   Batches to consolidate: {len(batch_results)}")
        logger.info("="*70)
        
        all_test_cases = []
        successful_batches = 0
        failed_batches = 0
        total_requirements_covered = 0
        failed_requirement_ids = []
        
        for batch_result in batch_results:
            batch_num = batch_result.get("batch_number", "unknown")
            
            if batch_result.get("batch_status") == "success":
                test_cases = batch_result.get("test_cases", [])
                all_test_cases.extend(test_cases)
                successful_batches += 1
                total_requirements_covered += batch_result.get("requirements_covered", 0)
                logger.info(f"✓ Batch {batch_num}: {len(test_cases)} test cases")
            else:
                failed_batches += 1
                failed_reqs = batch_result.get("batch_requirement_ids", [])
                failed_requirement_ids.extend(failed_reqs)
                error_msg = batch_result.get("error", "Unknown error")
                logger.error(f"✗ Batch {batch_num}: Failed - {error_msg}")
                logger.error(f"   Failed requirements: {', '.join(failed_reqs)}")
        
        # Calculate coverage statistics
        unique_requirement_ids = set()
        for tc in all_test_cases:
            req_id = tc.get("metadata", {}).get("requirement_id") or tc.get("requirement_id")
            if req_id:
                unique_requirement_ids.add(req_id)
        
        logger.info("="*70)
        logger.info("✓ CONSOLIDATION COMPLETE")
        logger.info(f"   Total Test Cases: {len(all_test_cases)}")
        logger.info(f"   Successful Batches: {successful_batches}")
        logger.info(f"   Failed Batches: {failed_batches}")
        logger.info(f"   Unique Requirements Covered: {len(unique_requirement_ids)}")
        if failed_requirement_ids:
            logger.warning(f"   ⚠ Failed Requirements: {len(failed_requirement_ids)}")
        logger.info("="*70)
        
        return {
            "test_cases": all_test_cases,
            "total_test_cases": len(all_test_cases),
            "batches_processed": successful_batches,
            "batches_failed": failed_batches,
            "total_batches": len(batch_results),
            "requirements_covered": len(unique_requirement_ids),
            "total_requirements_attempted": total_requirements_covered,
            "failed_requirement_ids": failed_requirement_ids,
            "coverage_summary": {
                "total_test_cases": len(all_test_cases),
                "unique_requirements": len(unique_requirement_ids),
                "success_rate": f"{(successful_batches / len(batch_results) * 100):.1f}%" if batch_results else "0%"
            },
            "consolidated_timestamp": datetime.now().isoformat(),
            "message": f"Successfully consolidated {len(all_test_cases)} test cases from {successful_batches}/{len(batch_results)} batches"
        }
        
    except Exception as e:
        logger.error(f"✗ Failed to consolidate test cases: {e}")
        logger.exception("Full traceback:")
        return {"error": f"Failed to consolidate parallel test case results: {e}"}

def generate_test_cases_for_requirements(requirements_json: str, selected_requirement_ids: str, test_types: str = "functional,security,compliance", standards: str = "FDA,HIPAA,IEC 62304", risk_level: str = "medium"):
    """
    Generate test cases for specific selected requirements only.
    
    This function generates test cases for only the requirements the user has selected,
    providing faster execution and more control over the process.
    
    Args:
        requirements_json (str): JSON string of all requirements
        selected_requirement_ids (str): Comma-separated list of requirement IDs to process (e.g., "REQ-001,REQ-003,REQ-005")
        test_types (str): Comma-separated test types
        standards (str): Comma-separated compliance standards
        risk_level (str): Risk level for test cases
    
    Returns:
        dict: Dictionary containing generated test cases for selected requirements
    """
    try:
        logging.info(f"Generating test cases for selected requirements")
        
        # Clean requirements JSON first
        cleaned_data = _clean_requirements_json(requirements_json)
        if "error" in cleaned_data:
            return cleaned_data
        
        requirements_data = cleaned_data
            
        # Extract requirements list
        if isinstance(requirements_data, dict) and "requirements" in requirements_data:
            all_requirements = requirements_data["requirements"]
        elif isinstance(requirements_data, list):
            all_requirements = requirements_data
        else:
            all_requirements = [requirements_data]
        
        # Parse selected requirement IDs
        selected_ids = [req_id.strip() for req_id in selected_requirement_ids.split(',') if req_id.strip()]
        
        # Filter requirements to only selected ones
        selected_requirements = [req for req in all_requirements if req.get("id", "") in selected_ids]
        
        if not selected_requirements:
            return {"error": f"No requirements found for IDs: {selected_requirement_ids}"}
        
        logging.info(f"Processing {len(selected_requirements)} selected requirements: {selected_ids}")
        
        test_type_list = [t.strip() for t in test_types.split(',') if t.strip()]
        standards_list = [s.strip() for s in standards.split(',') if s.strip()]
        
        client = init_genai_client()
        if not client:
            raise RuntimeError("Failed to initialize GenAI client.")
        
        # Use the detailed test case prompt
        try:
            from .prompts import DETAILED_TEST_CASE_PROMPT
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location("prompts", os.path.join(os.path.dirname(__file__), "prompts.py"))
            prompts_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(prompts_module)
            DETAILED_TEST_CASE_PROMPT = prompts_module.DETAILED_TEST_CASE_PROMPT
        
        # Create focused prompt for selected requirements with strict JSON format enforcement
        prompt = f"""
        You are a healthcare software testing expert. Generate comprehensive test cases for healthcare software requirements.
        
        CRITICAL: You MUST return ONLY valid JSON in the exact format specified below. No additional text, explanations, or comments outside the JSON structure.
        
        FOCUSED GENERATION: Generate test cases for the {len(selected_requirements)} SELECTED requirements below.
        Each requirement MUST have 2-3 comprehensive test cases generated.
        
        SELECTED REQUIREMENTS TO PROCESS:
        {json.dumps(selected_requirements, indent=2)}
        
        GENERATION PARAMETERS:
        - Test Types: {', '.join(test_type_list)}
        - Compliance Standards: {', '.join(standards_list)}
        - Risk Level: {risk_level}
        - Software Class: B (Medical Device Software)
        
        MANDATORY REQUIREMENTS:
        1. Generate 2-3 test cases for EACH of the {len(selected_requirements)} selected requirements
        2. Include detailed test steps with step_number, action, input_data, expected_result
        3. Ensure requirement IDs match exactly ({', '.join(selected_ids)})
        4. Focus on healthcare-specific scenarios and regulatory compliance
        5. RETURN ONLY THE JSON OBJECT - NO OTHER TEXT
        
        REQUIRED JSON OUTPUT FORMAT (return exactly this structure):
        {{
            "test_cases": [
                {{
                    "test_case_id": "TC-REQ-XXX-01",
                    "title": "Test case title",
                    "description": "Detailed description",
                    "metadata": {{
                        "requirement_id": "REQ-XXX",
                        "test_type": "functional|security|compliance",
                        "priority": "critical|high|medium|low",
                        "compliance_standards": ["FDA", "HIPAA"],
                        "risk_level": "{risk_level}",
                        "software_class": "B",
                        "estimated_duration": "30 minutes",
                        "automation_feasible": true,
                        "test_category": "positive"
                    }},
                    "test_steps": [
                        {{
                            "step_number": 1,
                            "action": "Specific action to perform",
                            "input_data": "Input data or parameters",
                            "expected_result": "Expected outcome"
                        }}
                    ],
                    "prerequisites": ["Prerequisite 1", "Prerequisite 2"],
                    "test_data": {{
                        "required_data": ["Data item 1"],
                        "test_environment": "Environment description",
                        "data_cleanup": "Cleanup steps"
                    }},
                    "expected_results": {{
                        "primary_result": "Main expected outcome",
                        "verification_criteria": ["Criteria 1", "Criteria 2"]
                    }},
                    "pass_criteria": ["Pass condition 1", "Pass condition 2"],
                    "fail_criteria": ["Fail condition 1", "Fail condition 2"],
                    "post_conditions": ["Post condition 1"],
                    "compliance_validation": {{
                        "regulatory_requirements": ["Regulation clause 1"],
                        "validation_evidence": "Evidence description"
                    }}
                }}
            ],
            "summary": {{
                "total_test_cases": 0,
                "requirements_processed": {len(selected_requirements)},
                "selected_requirement_ids": {selected_ids},
                "by_requirement": {{}},
                "by_type": {{}},
                "by_priority": {{}}
            }}
        }}
        
        CRITICAL: Return ONLY the JSON object above with your generated test cases. Do not include any text before or after the JSON.
        """
        
        # Generate content using LLM
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )
        
        # Parse LLM response with robust error handling
        response_text = response.candidates[0].content.parts[0].text.strip()
        
        # Clean up any potential formatting issues
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            # First try basic JSON parsing
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error in generate_test_cases_for_requirements: {e}")
            logging.error(f"Response text: {response_text[:500]}...")
            
            # Try to fix malformed JSON using our simple correction tool
            logging.info("Attempting to fix malformed JSON using simple correction...")
            corrected_result = fix_malformed_function_output(response_text, "test_cases")
            
            if "error" not in corrected_result:
                result = corrected_result
                logging.info("Successfully corrected malformed JSON")
            else:
                return {"error": f"Invalid JSON response from LLM and could not auto-correct: {e}"}
        
        # Basic structure validation as fallback
        if not isinstance(result, dict) or "test_cases" not in result:
            return {"error": "Invalid response structure: missing test_cases"}
        
        if not isinstance(result["test_cases"], list):
            return {"error": "Invalid response structure: test_cases must be an array"}
        
        # Add metadata
        result["generation_timestamp"] = datetime.now().isoformat()
        result["parameters"] = {
            "selected_requirements": selected_ids,
            "requirements_count": len(selected_requirements),
            "test_types": test_type_list,
            "standards": standards_list,
            "risk_level": risk_level
        }
        
        test_cases = result.get("test_cases", [])
        logging.info(f"Successfully generated {len(test_cases)} test cases for {len(selected_requirements)} selected requirements")
        
        return result
        
    except Exception as e:
        logging.error(f"Error generating test cases for selected requirements: {e}")
        return {"error": f"Failed to generate test cases: {e}"}

def create_test_case_reports(test_cases_json: str, report_name: str = ""):
    """
    Create PDF and CSV reports from generated test cases.
    
    Args:
        test_cases_json (str): JSON string of test cases
        report_name (str): Optional custom name for the reports
    
    Returns:
        dict: Dictionary containing paths to created report files
    """
    try:
        logging.info("Creating test case reports (PDF and CSV)")
        
        # Parse test cases
        if isinstance(test_cases_json, str):
            test_cases_data = json.loads(test_cases_json)
        else:
            test_cases_data = test_cases_json
        
        test_cases = test_cases_data.get("test_cases", []) if isinstance(test_cases_data, dict) else test_cases_data
        
        if not test_cases:
            return {"error": "No test cases provided for report generation"}
        
        # Generate timestamp and filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = report_name if report_name else "test_cases"
        
        local_files = []
        
        # Generate CSV report
        csv_content = _create_comprehensive_test_cases_csv(test_cases)
        csv_filename = f"output/{base_name}_{timestamp}.csv"
        with open(csv_filename, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_content)
        local_files.append(csv_filename)
        
        # Generate PDF report using TEST_CASE_PDF_PROMPT format
        pdf_content = _create_test_cases_pdf_with_prompt_format(test_cases)
        pdf_filename = f"output/{base_name}_{timestamp}.pdf"
        with open(pdf_filename, 'wb') as f:
            f.write(pdf_content)
        local_files.append(pdf_filename)
        
        result = {
            "report_files": local_files,
            "test_cases_count": len(test_cases),
            "generation_timestamp": datetime.now().isoformat(),
            "message": f"Successfully created reports for {len(test_cases)} test cases"
        }
        
        logging.info(f"Successfully created test case reports: {', '.join(local_files)}")
        return result
        
    except Exception as e:
        logging.error(f"Error creating test case reports: {e}")
        return {"error": f"Failed to create reports: {e}"}

def create_traceability_report(requirements_json: str = "", test_cases_json: str = "", report_name: str = ""):
    """
    Create traceability matrix report linking requirements to test cases.
    
    Args:
        requirements_json (str): JSON string of requirements
        test_cases_json (str): JSON string of test cases
        report_name (str): Optional custom name for the report
    
    Returns:
        dict: Dictionary containing path to created traceability report
    """
    try:
        logging.info("Creating traceability matrix report")
        
        # Parse inputs
        if isinstance(requirements_json, str):
            requirements_data = json.loads(requirements_json)
        else:
            requirements_data = requirements_json
            
        if isinstance(test_cases_json, str):
            test_cases_data = json.loads(test_cases_json)
        else:
            test_cases_data = test_cases_json
        
        # Extract requirements and test cases
        if isinstance(requirements_data, dict) and "requirements" in requirements_data:
            requirements = requirements_data["requirements"]
        elif isinstance(requirements_data, list):
            requirements = requirements_data
        else:
            requirements = [requirements_data]
            
        test_cases = test_cases_data.get("test_cases", []) if isinstance(test_cases_data, dict) else test_cases_data
        
        # Generate traceability matrix
        traceability_matrix = _create_traceability_matrix_from_test_cases(requirements, test_cases)
        
        # Generate timestamp and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = report_name if report_name else "traceability_matrix"
        
        local_files = []
        
        # Save traceability matrix as JSON
        json_filename = f"output/{base_name}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(traceability_matrix, f, indent=2, ensure_ascii=False)
        local_files.append(json_filename)
        
        # Generate PDF report
        pdf_content = _create_traceability_matrix_pdf_improved(traceability_matrix)
        pdf_filename = f"output/{base_name}_{timestamp}.pdf"
        with open(pdf_filename, 'wb') as f:
            f.write(pdf_content)
        local_files.append(pdf_filename)
        
        result = {
            "traceability_files": local_files,
            "traceability_matrix": traceability_matrix,
            "generation_timestamp": datetime.now().isoformat(),
            "message": f"Successfully created traceability report covering {len(requirements)} requirements and {len(test_cases)} test cases"
        }
        
        logging.info(f"Successfully created traceability report: {', '.join(local_files)}")
        return result
        
    except Exception as e:
        logging.error(f"Error creating traceability report: {e}")
        return {"error": f"Failed to create traceability report: {e}"}

# ============================================================================
# JIRA PROJECT ANALYSIS TOOLS (NEW FOR MULTI-AGENT SYSTEM)
# ============================================================================

def analyze_jira_project_structure(project_key: str) -> dict:
    """
    Analyze Jira project structure to identify epics, stories, and requirements.
    
    Method Signature:
        analyze_jira_project_structure(project_key: str) -> dict
    
    Args:
        project_key (str): The Jira project key to analyze (e.g., "MED")
    
    Returns:
        dict: Dictionary containing project analysis with keys:
            - project_info (dict): Basic project information
            - epics (list): List of epics with their details
            - stories (list): List of stories grouped by epic
            - requirements_summary (dict): Summary of requirements by type
            - risk_assessment (dict): Risk analysis for the project
            - recommendations (list): Recommendations for test case generation
            - error (str): Error message if analysis failed
            
    Example:
        result = analyze_jira_project_structure("MED")
        # Returns: {"project_info": {...}, "epics": [...], "stories": [...], ...}
    """
    try:
        logger.info("="*70)
        logger.info(f"🔍 STARTING: Jira Project Analysis")
        logger.info(f"   Project Key: {project_key}")
        logger.info("="*70)
        
        # Import Jira client
        from jira import JIRA
        
        # Initialize Jira client
        logger.debug("Checking Jira credentials from environment variables")
        jira_url = os.getenv("JIRA_URL")
        # Support both JIRA_EMAIL and JIRA_USERNAME for compatibility
        jira_username = os.getenv("JIRA_EMAIL") or os.getenv("JIRA_USERNAME")
        jira_token = os.getenv("JIRA_API_TOKEN")
        
        if not all([jira_url, jira_username, jira_token]):
            error_msg = "Jira credentials not configured. Please set JIRA_URL, JIRA_EMAIL (or JIRA_USERNAME), and JIRA_API_TOKEN"
            logger.error(error_msg)
            logger.error(f"   JIRA_URL: {'✓ Set' if jira_url else '✗ Missing'}")
            logger.error(f"   JIRA_EMAIL/USERNAME: {'✓ Set' if jira_username else '✗ Missing'}")
            logger.error(f"   JIRA_API_TOKEN: {'✓ Set' if jira_token else '✗ Missing'}")
            raise ValueError(error_msg)
        
        logger.info(f"✓ Jira credentials found")
        logger.debug(f"   URL: {jira_url}")
        logger.debug(f"   Username: {jira_username}")
        logger.debug(f"   Token: {'*' * 10} (hidden)")
        
        logger.info("Connecting to Jira...")
        jira_client = JIRA(
            server=jira_url,
            basic_auth=(jira_username, jira_token)
        )
        logger.info("✓ Successfully connected to Jira")
        
        # Get project information
        logger.info("Fetching project information...")
        project = jira_client.project(project_key)
        project_info = {
            "key": project.key,
            "name": project.name,
            "description": getattr(project, 'description', 'No description'),
            "lead": getattr(project.lead, 'displayName', 'Unknown'),
            "project_type": getattr(project, 'projectTypeKey', 'Unknown')
        }
        logger.info(f"✓ Project: {project_info['name']} ({project_info['key']})")
        logger.debug(f"   Lead: {project_info['lead']}, Type: {project_info['project_type']}")
        
        # Get all epics in the project
        logger.info("Searching for epics...")
        epics_jql = f'project = {project_key} AND issuetype = "Epic"'
        epics_issues = jira_client.search_issues(epics_jql, maxResults=False)
        logger.info(f"✓ Found {len(epics_issues)} epics")
        
        epics = []
        stories = []
        requirements_by_type = {
            "functional": 0,
            "security": 0,
            "compliance": 0,
            "ui_ux": 0,
            "performance": 0,
            "integration": 0,
            "other": 0
        }
        
        logger.debug("Processing epics and stories...")
        for epic in epics_issues:
            # Get stories for this epic
            stories_jql = f'project = {project_key} AND "Epic Link" = {epic.key}'
            epic_stories = jira_client.search_issues(stories_jql, maxResults=False)
            logger.debug(f"   Processing {epic.key}: {len(epic_stories)} stories")
            
            # Analyze epic content for categorization
            epic_description = getattr(epic.fields, 'description', '') or ''
            epic_summary = getattr(epic.fields, 'summary', '') or ''
            epic_content = f"{epic_summary} {epic_description}".lower()
            
            # Determine epic category and risk level
            category = "functional"
            risk_level = "medium"
            
            if any(keyword in epic_content for keyword in ["security", "authentication", "authorization", "encryption"]):
                category = "security"
                risk_level = "high"
            elif any(keyword in epic_content for keyword in ["compliance", "hipaa", "fda", "audit", "regulatory"]):
                category = "compliance"
                risk_level = "high"
            elif any(keyword in epic_content for keyword in ["ui", "ux", "interface", "screen", "dashboard"]):
                category = "ui_ux"
                risk_level = "medium"
            elif any(keyword in epic_content for keyword in ["performance", "speed", "load", "response"]):
                category = "performance"
                risk_level = "medium"
            elif any(keyword in epic_content for keyword in ["integration", "api", "interface", "external"]):
                category = "integration"
                risk_level = "high"
            
            # Count requirements by type
            requirements_by_type[category] += len(epic_stories)
            
            epic_data = {
                "key": epic.key,
                "summary": epic_summary,
                "description": epic_description,
                "status": getattr(epic.fields.status, 'name', 'Unknown'),
                "priority": getattr(epic.fields.priority, 'name', 'Medium') if hasattr(epic.fields, 'priority') and epic.fields.priority else 'Medium',
                "story_count": len(epic_stories),
                "category": category,
                "risk_level": risk_level,
                "assignee": getattr(epic.fields.assignee, 'displayName', 'Unassigned') if hasattr(epic.fields, 'assignee') and epic.fields.assignee else 'Unassigned'
            }
            epics.append(epic_data)
            
            # Add stories for this epic
            for story in epic_stories:
                story_data = {
                    "key": story.key,
                    "summary": getattr(story.fields, 'summary', ''),
                    "description": getattr(story.fields, 'description', '') or '',
                    "status": getattr(story.fields.status, 'name', 'Unknown'),
                    "priority": getattr(story.fields.priority, 'name', 'Medium') if hasattr(story.fields, 'priority') and story.fields.priority else 'Medium',
                    "epic_key": epic.key,
                    "epic_summary": epic_summary,
                    "assignee": getattr(story.fields.assignee, 'displayName', 'Unassigned') if hasattr(story.fields, 'assignee') and story.fields.assignee else 'Unassigned'
                }
                stories.append(story_data)
        
        # Generate risk assessment
        total_stories = len(stories)
        high_risk_epics = len([e for e in epics if e["risk_level"] == "high"])
        
        risk_assessment = {
            "total_epics": len(epics),
            "total_stories": total_stories,
            "high_risk_epics": high_risk_epics,
            "risk_score": min(100, (high_risk_epics / len(epics) * 100) if epics else 0),
            "complexity_score": min(100, (total_stories / 10) * 10),  # Rough complexity based on story count
            "recommendations": []
        }
        
        # Generate recommendations
        recommendations = []
        if high_risk_epics > 0:
            recommendations.append(f"Prioritize testing for {high_risk_epics} high-risk epics (security, compliance, integration)")
        
        if requirements_by_type["compliance"] > 0:
            recommendations.append("Include regulatory compliance validation in test cases")
        
        if requirements_by_type["security"] > 0:
            recommendations.append("Implement comprehensive security testing scenarios")
        
        if total_stories > 50:
            recommendations.append("Consider phased test case generation due to large project size")
        
        recommendations.append("Ensure traceability between requirements and test cases")
        recommendations.append("Include both positive and negative test scenarios")
        
        # Summary logging
        logger.info("="*70)
        logger.info("✓ ANALYSIS COMPLETE")
        logger.info(f"   Project: {project_info['name']} ({project_key})")
        logger.info(f"   Epics: {len(epics)}")
        logger.info(f"   Stories: {total_stories}")
        logger.info(f"   Risk Score: {risk_assessment['risk_score']:.1f}%")
        logger.info(f"   High Risk Epics: {risk_assessment['high_risk_epics']}")
        logger.info("="*70)
        
        return {
            "project_info": project_info,
            "epics": epics,
            "stories": stories,
            "requirements_summary": requirements_by_type,
            "risk_assessment": risk_assessment,
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat(),
            "message": f"Successfully analyzed project {project_key} with {len(epics)} epics and {total_stories} stories"
        }
        
    except Exception as e:
        logger.error("="*70)
        logger.error("✗ ANALYSIS FAILED")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   Project Key: {project_key}")
        logger.error("="*70)
        logger.exception("Full traceback:")
        return {"error": f"Failed to analyze Jira project structure: {e}"}

def get_epic_requirements_summary(project_key: str, epic_keys: str = "", max_stories_per_epic: int = 20, include_full_details: bool = False) -> dict:
    """
    Get requirements summary for specific epics (OPTIMIZED FOR LARGE EPICS).
    
    This function now processes epics in a way that avoids malformed function calls
    by limiting data size and providing options for summary vs. full details.
    
    Method Signature:
        get_epic_requirements_summary(project_key: str, epic_keys: str = "", max_stories_per_epic: int = 20, include_full_details: bool = False) -> dict
    
    Args:
        project_key (str): The Jira project key (e.g., "MED")
        epic_keys (str, optional): Comma-separated epic keys to analyze (empty for all epics)
        max_stories_per_epic (int): Maximum stories to include per epic (default: 20). 
                                   If epic has more, returns summary only.
        include_full_details (bool): If False, returns summary without full descriptions (default: False)
    
    Returns:
        dict: Dictionary containing epic requirements summary with keys:
            - epic_summaries (list): Summary for each epic (may be truncated if large)
            - total_requirements (int): Total number of requirements found
            - requirements_by_category (dict): Requirements grouped by category (summary counts)
            - testing_recommendations (list): Specific testing recommendations
            - processing_info (dict): Information about processing (warnings, truncations)
            - error (str): Error message if analysis failed
            
    Example:
        result = get_epic_requirements_summary("MED", "MED-107", max_stories_per_epic=15, include_full_details=False)
    """
    try:
        logger.info("="*70)
        logger.info(f"📊 STARTING: Epic Requirements Summary (OPTIMIZED)")
        logger.info(f"   Project Key: {project_key}")
        logger.info(f"   Epic Keys: {epic_keys or 'All epics'}")
        logger.info(f"   Max Stories per Epic: {max_stories_per_epic}")
        logger.info(f"   Include Full Details: {include_full_details}")
        logger.info("="*70)
        
        # Import Jira client
        from jira import JIRA
        
        # Initialize Jira client
        logger.debug("Initializing Jira client...")
        jira_url = os.getenv("JIRA_URL")
        jira_username = os.getenv("JIRA_EMAIL") or os.getenv("JIRA_USERNAME")
        jira_token = os.getenv("JIRA_API_TOKEN")
        
        if not all([jira_url, jira_username, jira_token]):
            raise ValueError("Jira credentials not configured. Please set JIRA_URL, JIRA_EMAIL (or JIRA_USERNAME), and JIRA_API_TOKEN")
        
        jira_client = JIRA(server=jira_url, basic_auth=(jira_username, jira_token))
        logger.info("✓ Connected to Jira")
        
        # Determine which epics to analyze
        if epic_keys and epic_keys.strip():
            epic_key_list = [key.strip() for key in epic_keys.split(",")]
            epics_jql = f'project = {project_key} AND issuetype = "Epic" AND key in ({",".join(epic_key_list)})'
            logger.info(f"Processing {len(epic_key_list)} specific epic(s): {', '.join(epic_key_list)}")
        else:
            epics_jql = f'project = {project_key} AND issuetype = "Epic"'
            logger.info("Processing all epics in project")
        
        logger.info("Fetching epics from Jira...")
        epics_issues = jira_client.search_issues(epics_jql, maxResults=False)
        logger.info(f"✓ Found {len(epics_issues)} epic(s)")
        
        epic_summaries = []
        total_requirements = 0
        requirements_by_category = {
            "functional": [],
            "security": [],
            "compliance": [],
            "ui_ux": [],
            "performance": [],
            "integration": [],
            "other": []
        }
        processing_info = {
            "warnings": [],
            "truncated_epics": [],
            "total_stories_processed": 0,
            "total_stories_found": 0
        }
        
        # Process each epic
        for idx, epic in enumerate(epics_issues, 1):
            logger.info(f"[{idx}/{len(epics_issues)}] Processing epic {epic.key}...")
            
            # Get stories for this epic
            stories_jql = f'project = {project_key} AND "Epic Link" = {epic.key}'
            epic_stories = jira_client.search_issues(stories_jql, maxResults=False)
            processing_info["total_stories_found"] += len(epic_stories)
            
            logger.info(f"   Found {len(epic_stories)} stories in {epic.key}")
            
            # Note: With parallel processing, we process ALL stories (no truncation needed)
            # Only mark as "large" for informational purposes
            is_large_epic = len(epic_stories) > max_stories_per_epic
            if is_large_epic:
                logger.info(f"   ℹ Epic {epic.key} has {len(epic_stories)} stories (large epic - using parallel processing)")
            
            # Analyze epic content
            epic_description = getattr(epic.fields, 'description', '') or ''
            epic_summary_text = getattr(epic.fields, 'summary', '') or ''
            
            # Use parallel processing for ALL stories (PRODUCTION-GRADE)
            logger.info(f"   ⚡ Using parallel processing for {len(epic_stories)} stories...")
            
            # Process epic stories in parallel (batch size 5 for optimal performance)
            parallel_result = process_epic_stories_parallel(
                project_key=project_key,
                epic_key=epic.key,
                batch_size=5  # Process 5 stories in parallel per batch
            )
            
            if "error" in parallel_result:
                logger.error(f"   ✗ Parallel processing failed for {epic.key}: {parallel_result['error']}")
                # Fallback to sequential processing if parallel fails
                logger.warning(f"   Falling back to sequential processing...")
                # In fallback mode, still process all stories (don't truncate)
                stories_to_process = epic_stories
                story_requirements = []
                category_counts = {}
                
                for story in stories_to_process:
                    story_summary = getattr(story.fields, 'summary', '') or ''
                    story_description = getattr(story.fields, 'description', '') or ''
                    
                    if not include_full_details and len(story_description) > 200:
                        story_description = story_description[:200] + "... [truncated]"
                    
                    story_content = f"{story_summary} {story_description}".lower()
                    category = "functional"
                    if any(kw in story_content for kw in ["security", "authentication", "authorization"]):
                        category = "security"
                    elif any(kw in story_content for kw in ["compliance", "hipaa", "fda"]):
                        category = "compliance"
                    elif any(kw in story_content for kw in ["ui", "ux", "interface"]):
                        category = "ui_ux"
                    elif any(kw in story_content for kw in ["performance", "speed", "load"]):
                        category = "performance"
                    elif any(kw in story_content for kw in ["integration", "api", "external"]):
                        category = "integration"
                    
                    requirement = {
                        "key": story.key,  # Jira story key (e.g., "MED-110")
                        "id": story.key,   # Same as key for compatibility with test case generation
                        "summary": story_summary,
                        "category": category,
                        "priority": getattr(story.fields.priority, 'name', 'Medium') if hasattr(story.fields, 'priority') and story.fields.priority else 'Medium',
                        "status": getattr(story.fields.status, 'name', 'Unknown')
                    }
                    if include_full_details:
                        requirement["description"] = story_description
                    story_requirements.append(requirement)
                    category_counts[category] = category_counts.get(category, 0) + 1
                    total_requirements += 1
                    processing_info["total_stories_processed"] += 1
                    requirements_by_category[category].append({
                        "key": story.key,
                        "summary": story_summary[:100] + "..." if len(story_summary) > 100 else story_summary
                    })
            else:
                # Extract requirements from parallel processing result
                story_requirements = parallel_result.get("requirements", [])
                category_counts = parallel_result.get("category_counts", {})
                processing_info["total_stories_processed"] += parallel_result.get("processed_stories", 0)
                total_requirements += len(story_requirements)
                
                # Map parallel processing results to requirements_by_category format
                for req in story_requirements:
                    category = req.get("category", "functional")
                    requirements_by_category[category].append({
                        "key": req["key"],
                        "summary": req["summary"][:100] + "..." if len(req.get("summary", "")) > 100 else req.get("summary", "")
                    })
                
                # Truncate descriptions if not including full details
                if not include_full_details:
                    for req in story_requirements:
                        desc = req.get("description", "")
                        if desc and len(desc) > 200:
                            req["description"] = desc[:200] + "... [truncated]"
                        elif "description" not in req:
                            req["description"] = ""
            
            logger.info(f"   ✓ Processed {len(story_requirements)} stories from {epic.key} (parallel processing)")
            
            # Create epic summary (all stories processed via parallel processing)
            epic_summary = {
                "epic_key": epic.key,
                "epic_summary": epic_summary_text,
                "story_count": len(epic_stories),
                "stories_processed": len(story_requirements),
                "processing_method": "parallel",
                "is_large_epic": is_large_epic,
                "categories_covered": list(set([req["category"] for req in story_requirements])),
                "category_counts": category_counts,
                "priority_distribution": {
                    "high": len([req for req in story_requirements if req.get("priority") == "High" or req.get("priority") == "Critical"]),
                    "medium": len([req for req in story_requirements if req.get("priority") == "Medium"]),
                    "low": len([req for req in story_requirements if req.get("priority") == "Low"])
                },
                "requirements": story_requirements  # All requirements included (parallel processing handles all)
            }
            
            # Add parallel processing info if available
            if "error" not in parallel_result:
                epic_summary["parallel_processing_info"] = {
                    "batch_size": 5,
                    "total_stories": parallel_result.get("total_stories", len(story_requirements)),
                    "processed_stories": parallel_result.get("processed_stories", len(story_requirements)),
                    "failed_stories": parallel_result.get("failed_stories", 0)
                }
            
            epic_summaries.append(epic_summary)
        
        # Generate testing recommendations
        testing_recommendations = []
        
        if requirements_by_category["security"]:
            testing_recommendations.append(f"Security Testing: {len(requirements_by_category['security'])} security requirements need comprehensive testing including penetration testing, access control validation, and encryption verification")
        
        if requirements_by_category["compliance"]:
            testing_recommendations.append(f"Compliance Testing: {len(requirements_by_category['compliance'])} compliance requirements need regulatory validation testing for HIPAA, FDA, and other healthcare standards")
        
        if requirements_by_category["integration"]:
            testing_recommendations.append(f"Integration Testing: {len(requirements_by_category['integration'])} integration requirements need end-to-end testing with external systems and APIs")
        
        if requirements_by_category["ui_ux"]:
            testing_recommendations.append(f"UI/UX Testing: {len(requirements_by_category['ui_ux'])} interface requirements need usability testing, accessibility validation, and cross-browser compatibility testing")
        
        if requirements_by_category["performance"]:
            testing_recommendations.append(f"Performance Testing: {len(requirements_by_category['performance'])} performance requirements need load testing, stress testing, and response time validation")
        
        testing_recommendations.append("Implement risk-based testing approach prioritizing patient safety and regulatory compliance")
        testing_recommendations.append("Ensure comprehensive traceability matrix linking test cases to requirements")
        testing_recommendations.append("Include both positive and negative test scenarios for each requirement")
        
        # Note: All epics processed via parallel processing (no truncation needed)
        logger.info(f"✓ All epics processed using parallel processing (batch size: 5)")
        
        # Summary logging
        logger.info("="*70)
        logger.info("✓ SUMMARY COMPLETE (PARALLEL PROCESSING)")
        logger.info(f"   Epics Processed: {len(epic_summaries)}")
        logger.info(f"   Stories Found: {processing_info['total_stories_found']}")
        logger.info(f"   Stories Processed: {processing_info['total_stories_processed']} (via parallel batches)")
        logger.info(f"   Total Requirements: {total_requirements}")
        logger.info(f"   Processing Method: Parallel (batch size: 5)")
        logger.info("="*70)
        
        # Optimize return - only include category counts, not full lists
        return {
            "epic_summaries": epic_summaries,
            "total_requirements": total_requirements,
            "requirements_by_category": {k: len(v) for k, v in requirements_by_category.items()},
            "testing_recommendations": testing_recommendations,
            "processing_info": processing_info,
            "analysis_timestamp": datetime.now().isoformat(),
            "message": f"Successfully analyzed {len(epic_summaries)} epic(s). Found {processing_info['total_stories_found']} stories, processed {processing_info['total_stories_processed']}."
        }
        
    except Exception as e:
        logger.error("="*70)
        logger.error("✗ SUMMARY FAILED")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   Project Key: {project_key}")
        logger.error(f"   Epic Keys: {epic_keys}")
        logger.error("="*70)
        logger.exception("Full traceback:")
        return {"error": f"Failed to get epic requirements summary: {e}"}

def process_single_user_story(project_key: str, story_key: str) -> dict:
    """
    Process a single user story to extract structured requirements.
    
    This is designed to be called in parallel for efficient processing.
    
    Args:
        project_key (str): The Jira project key
        story_key (str): The user story key (e.g., "MED-110")
    
    Returns:
        dict: Structured requirement data for the story
    """
    try:
        logger.debug(f"Processing story: {story_key}")
        
        from jira import JIRA
        
        jira_url = os.getenv("JIRA_URL")
        jira_username = os.getenv("JIRA_EMAIL") or os.getenv("JIRA_USERNAME")
        jira_token = os.getenv("JIRA_API_TOKEN")
        
        if not all([jira_url, jira_username, jira_token]):
            raise ValueError("Jira credentials not configured")
        
        jira_client = JIRA(server=jira_url, basic_auth=(jira_username, jira_token))
        
        # Get story details
        story = jira_client.issue(story_key)
        
        story_summary = getattr(story.fields, 'summary', '') or ''
        story_description = getattr(story.fields, 'description', '') or ''
        story_content = f"{story_summary} {story_description}".lower()
        
        # Categorize story
        category = "functional"  # default
        if any(keyword in story_content for keyword in ["security", "authentication", "authorization", "encryption", "access control"]):
            category = "security"
        elif any(keyword in story_content for keyword in ["compliance", "hipaa", "fda", "audit", "regulatory", "gdpr"]):
            category = "compliance"
        elif any(keyword in story_content for keyword in ["ui", "ux", "interface", "screen", "dashboard", "form", "button"]):
            category = "ui_ux"
        elif any(keyword in story_content for keyword in ["performance", "speed", "load", "response", "timeout"]):
            category = "performance"
        elif any(keyword in story_content for keyword in ["integration", "api", "interface", "external", "third-party"]):
            category = "integration"
        
        # Determine risk level
        risk_level = "medium"
        priority = getattr(story.fields.priority, 'name', 'Medium') if hasattr(story.fields, 'priority') and story.fields.priority else 'Medium'
        if priority == "Critical" or "critical" in story_content or "patient safety" in story_content:
            risk_level = "high"
        elif priority == "Low" or "low priority" in story_content:
            risk_level = "low"
        
        requirement = {
            "key": story.key,  # Jira story key (e.g., "MED-110")
            "id": story.key,   # Same as key for compatibility with test case generation
            "summary": story_summary,
            "description": story_description,
            "category": category,
            "priority": priority,
            "status": getattr(story.fields.status, 'name', 'Unknown'),
            "risk_level": risk_level,
            "assignee": getattr(story.fields.assignee, 'displayName', 'Unassigned') if hasattr(story.fields, 'assignee') and story.fields.assignee else 'Unassigned',
            "epic_key": getattr(story.fields, 'customfield_10014', '') or '',  # Epic Link custom field
            "processing_timestamp": datetime.now().isoformat()
        }
        
        logger.debug(f"✓ Processed story {story_key}: {category}, {risk_level} risk")
        
        return {
            "success": True,
            "requirement": requirement,
            "story_key": story_key
        }
        
    except Exception as e:
        logger.error(f"✗ Failed to process story {story_key}: {e}")
        return {
            "success": False,
            "story_key": story_key,
            "error": str(e)
        }

def process_epic_stories_parallel(project_key: str, epic_key: str, batch_size: int = 5) -> dict:
    """
    Process ALL user stories in an epic using parallel processing (PRODUCTION-GRADE).
    
    This function processes stories in parallel batches using concurrent execution,
    designed for production use with proper error handling and logging.
    
    Args:
        project_key (str): The Jira project key (e.g., "MED")
        epic_key (str): The epic key to process (e.g., "MED-107")
        batch_size (int): Number of stories to process in parallel (default: 5)
    
    Returns:
        dict: Complete processed requirements for all stories in the epic
    """
    try:
        logger.info("="*70)
        logger.info(f"⚡ PARALLEL PROCESSING: Epic User Stories")
        logger.info(f"   Project: {project_key}")
        logger.info(f"   Epic: {epic_key}")
        logger.info(f"   Parallel Batch Size: {batch_size}")
        logger.info("="*70)
        
        from jira import JIRA
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        jira_url = os.getenv("JIRA_URL")
        jira_username = os.getenv("JIRA_EMAIL") or os.getenv("JIRA_USERNAME")
        jira_token = os.getenv("JIRA_API_TOKEN")
        
        if not all([jira_url, jira_username, jira_token]):
            raise ValueError("Jira credentials not configured")
        
        jira_client = JIRA(server=jira_url, basic_auth=(jira_username, jira_token))
        logger.info("✓ Connected to Jira")
        
        # Get all stories for epic
        stories_jql = f'project = {project_key} AND "Epic Link" = {epic_key}'
        all_stories = jira_client.search_issues(stories_jql, maxResults=False)
        total_stories = len(all_stories)
        
        logger.info(f"✓ Found {total_stories} stories in {epic_key}")
        
        if total_stories == 0:
            return {
                "epic_key": epic_key,
                "total_stories": 0,
                "processed_stories": 0,
                "requirements": [],
                "message": f"No stories found in epic {epic_key}"
            }
        
        # Extract story keys
        story_keys = [story.key for story in all_stories]
        logger.info(f"Processing {total_stories} stories in parallel batches of {batch_size}")
        
        # Process stories in parallel batches
        all_requirements = []
        failed_stories = []
        category_counts = {}
        
        # Process in batches to avoid overwhelming the system
        total_batches = (total_stories + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_stories)
            batch_stories = story_keys[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_num + 1}/{total_batches}: {len(batch_stories)} stories")
            
            # Process batch in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                # Submit all stories in batch
                future_to_story = {
                    executor.submit(process_single_user_story, project_key, story_key): story_key
                    for story_key in batch_stories
                }
                
                # Collect results as they complete
                batch_results = []
                for future in as_completed(future_to_story):
                    story_key = future_to_story[future]
                    try:
                        result = future.result()
                        if result.get("success"):
                            all_requirements.append(result["requirement"])
                            category = result["requirement"]["category"]
                            category_counts[category] = category_counts.get(category, 0) + 1
                            batch_results.append(f"✓ {story_key}")
                        else:
                            failed_stories.append({
                                "story_key": story_key,
                                "error": result.get("error", "Unknown error")
                            })
                            batch_results.append(f"✗ {story_key}: {result.get('error', 'Failed')}")
                    except Exception as e:
                        logger.error(f"Exception processing {story_key}: {e}")
                        failed_stories.append({
                            "story_key": story_key,
                            "error": str(e)
                        })
                        batch_results.append(f"✗ {story_key}: {str(e)}")
                
                # Log batch completion
                success_count = len([r for r in batch_results if r.startswith("✓")])
                logger.info(f"   Batch {batch_num + 1} complete: {success_count}/{len(batch_stories)} successful")
        
        # Calculate statistics
        high_risk_count = len([r for r in all_requirements if r.get("risk_level") == "high"])
        medium_risk_count = len([r for r in all_requirements if r.get("risk_level") == "medium"])
        low_risk_count = len([r for r in all_requirements if r.get("risk_level") == "low"])
        
        logger.info("="*70)
        logger.info("✓ PARALLEL PROCESSING COMPLETE")
        logger.info(f"   Epic: {epic_key}")
        logger.info(f"   Total Stories: {total_stories}")
        logger.info(f"   Successfully Processed: {len(all_requirements)}")
        logger.info(f"   Failed: {len(failed_stories)}")
        logger.info(f"   Categories: {category_counts}")
        logger.info(f"   Risk Distribution: High={high_risk_count}, Medium={medium_risk_count}, Low={low_risk_count}")
        if failed_stories:
            logger.warning(f"   ⚠ Failed Stories: {[s['story_key'] for s in failed_stories]}")
        logger.info("="*70)
        
        return {
            "epic_key": epic_key,
            "total_stories": total_stories,
            "processed_stories": len(all_requirements),
            "failed_stories": len(failed_stories),
            "requirements": all_requirements,
            "category_counts": category_counts,
            "risk_distribution": {
                "high": high_risk_count,
                "medium": medium_risk_count,
                "low": low_risk_count
            },
            "failed_stories_details": failed_stories,
            "processing_info": {
                "batch_size": batch_size,
                "total_batches": total_batches,
                "parallel_processing": True,
                "processing_timestamp": datetime.now().isoformat()
            },
            "message": f"Successfully processed {len(all_requirements)}/{total_stories} stories from epic {epic_key} using parallel processing"
        }
        
    except Exception as e:
        logger.error("="*70)
        logger.error("✗ PARALLEL PROCESSING FAILED")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   Epic: {epic_key}")
        logger.error("="*70)
        logger.exception("Full traceback:")
        return {"error": f"Failed to process epic stories in parallel: {e}"}

def get_single_epic_stories(project_key: str, epic_key: str, batch_size: int = 20, batch_number: int = 1) -> dict:
    """
    Get stories for a single epic in batches (for large epics).
    
    Use this function when an epic has too many stories and get_epic_requirements_summary() 
    truncates the results.
    
    Args:
        project_key (str): The Jira project key (e.g., "MED")
        epic_key (str): The epic key to get stories for (e.g., "MED-107")
        batch_size (int): Number of stories per batch (default: 20)
        batch_number (int): Which batch to retrieve (1-based, default: 1)
    
    Returns:
        dict: Stories for the requested batch with pagination info
    """
    try:
        logger.info("="*70)
        logger.info(f"📖 Getting Stories for Epic (BATCH {batch_number})")
        logger.info(f"   Project: {project_key}")
        logger.info(f"   Epic: {epic_key}")
        logger.info(f"   Batch Size: {batch_size}")
        logger.info("="*70)
        
        from jira import JIRA
        
        jira_url = os.getenv("JIRA_URL")
        jira_username = os.getenv("JIRA_EMAIL") or os.getenv("JIRA_USERNAME")
        jira_token = os.getenv("JIRA_API_TOKEN")
        
        if not all([jira_url, jira_username, jira_token]):
            raise ValueError("Jira credentials not configured")
        
        jira_client = JIRA(server=jira_url, basic_auth=(jira_username, jira_token))
        logger.info("✓ Connected to Jira")
        
        # Get all stories for epic
        stories_jql = f'project = {project_key} AND "Epic Link" = {epic_key}'
        all_stories = jira_client.search_issues(stories_jql, maxResults=False)
        total_stories = len(all_stories)
        
        logger.info(f"✓ Found {total_stories} total stories in {epic_key}")
        
        # Calculate pagination
        start_idx = (batch_number - 1) * batch_size
        end_idx = start_idx + batch_size
        total_batches = (total_stories + batch_size - 1) // batch_size  # Ceiling division
        
        if start_idx >= total_stories:
            return {
                "error": f"Batch {batch_number} is out of range. Total batches: {total_batches}"
            }
        
        stories_batch = all_stories[start_idx:end_idx]
        logger.info(f"Retrieving batch {batch_number}/{total_batches}: stories {start_idx+1}-{min(end_idx, total_stories)}")
        
        # Process stories in batch
        batch_requirements = []
        for story in stories_batch:
            story_summary = getattr(story.fields, 'summary', '') or ''
            story_description = getattr(story.fields, 'description', '') or ''
            
            batch_requirements.append({
                "key": story.key,
                "summary": story_summary,
                "description": story_description,
                "priority": getattr(story.fields.priority, 'name', 'Medium') if hasattr(story.fields, 'priority') and story.fields.priority else 'Medium',
                "status": getattr(story.fields.status, 'name', 'Unknown')
            })
        
        logger.info(f"✓ Processed {len(batch_requirements)} stories in batch {batch_number}")
        
        return {
            "epic_key": epic_key,
            "batch_number": batch_number,
            "total_batches": total_batches,
            "total_stories": total_stories,
            "batch_size": len(batch_requirements),
            "stories": batch_requirements,
            "pagination": {
                "current_batch": batch_number,
                "total_batches": total_batches,
                "has_next": batch_number < total_batches,
                "has_previous": batch_number > 1
            },
            "message": f"Retrieved batch {batch_number} of {total_batches} for epic {epic_key}"
        }
        
    except Exception as e:
        logger.error(f"Error getting epic stories: {e}")
        logger.exception("Full traceback:")
        return {"error": f"Failed to get epic stories: {e}"}

# ============================================================================
# TEST PLAN GENERATION TOOLS (NEW FOR MULTI-AGENT SYSTEM)
# ============================================================================

def generate_comprehensive_test_plan(project_key: str, selected_epics: str, requirements_summary: str = "") -> dict:
    """
    Generate comprehensive test plan for selected epics with healthcare-specific considerations.
    
    Method Signature:
        generate_comprehensive_test_plan(project_key: str, selected_epics: str, requirements_summary: str = "") -> dict
    
    Args:
        project_key (str): The Jira project key (e.g., "MED")
        selected_epics (str): Comma-separated epic keys to include in test plan
        requirements_summary (str, optional): JSON string of requirements summary for context
    
    Returns:
        dict: Dictionary containing comprehensive test plan with keys:
            - test_plan_id (str): Unique test plan identifier
            - title (str): Test plan title
            - scope (str): Testing scope and objectives
            - strategy (dict): Test strategy by category
            - environment_requirements (list): Test environment specifications
            - entry_exit_criteria (dict): Entry and exit criteria
            - risk_assessment (dict): Risk analysis and mitigation
            - resource_requirements (dict): Required resources and skills
            - timeline (dict): Testing timeline and milestones
            - compliance_approach (dict): Regulatory compliance testing approach
            - deliverables (list): Expected test deliverables
            - error (str): Error message if generation failed
            
    Example:
        result = generate_comprehensive_test_plan("MED", "MED-1,MED-2", '{"total_requirements": 25}')
        # Returns: {"test_plan_id": "TP-MED-001", "title": "CarePlus EHR Test Plan", ...}
    """
    try:
        logging.info(f"Generating comprehensive test plan for project: {project_key}")
        
        # Parse selected epics
        epic_keys = [key.strip() for key in selected_epics.split(",") if key.strip()]
        
        # Parse requirements summary if provided
        req_summary = {}
        if requirements_summary:
            try:
                req_summary = json.loads(requirements_summary)
            except json.JSONDecodeError:
                logging.warning("Could not parse requirements summary JSON")
        
        # Generate unique test plan ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_plan_id = f"TP-{project_key}-{timestamp}"
        
        # Initialize GenAI client
        client = init_genai_client()
        
        # Create test plan generation prompt
        test_plan_prompt = f"""
        Generate a comprehensive healthcare software test plan for the following:
        
        PROJECT: {project_key}
        SELECTED EPICS: {', '.join(epic_keys)}
        REQUIREMENTS SUMMARY: {json.dumps(req_summary, indent=2) if req_summary else 'Not provided'}
        
        Create a detailed test plan that includes:
        
        1. TEST PLAN OVERVIEW
        - Title and identifier
        - Scope and objectives
        - Testing approach summary
        
        2. TEST STRATEGY BY CATEGORY
        - Functional Testing Strategy
        - Security Testing Strategy
        - Compliance Testing Strategy (HIPAA, FDA 21 CFR Part 11, IEC 62304)
        - Integration Testing Strategy
        - Performance Testing Strategy
        - Usability/Accessibility Testing Strategy
        - Negative Testing Strategy
        
        3. HEALTHCARE-SPECIFIC CONSIDERATIONS
        - Patient safety validation approach
        - PHI (Protected Health Information) handling
        - Audit trail verification
        - Clinical workflow validation
        - Medical device integration testing
        - Regulatory compliance checkpoints
        
        4. TEST ENVIRONMENT REQUIREMENTS
        - Hardware specifications
        - Software requirements
        - Test data requirements (synthetic patient data)
        - Security configurations
        - Integration endpoints
        
        5. ENTRY AND EXIT CRITERIA
        - Entry criteria for test execution
        - Exit criteria for test completion
        - Suspension and resumption criteria
        - Defect management approach
        
        6. RISK ASSESSMENT AND MITIGATION
        - Identified risks and impact
        - Risk mitigation strategies
        - Contingency plans
        - Risk monitoring approach
        
        7. RESOURCE REQUIREMENTS
        - Team structure and roles
        - Required skills and expertise
        - Training requirements
        - External dependencies
        
        8. TIMELINE AND MILESTONES
        - Test planning phase
        - Test design and development
        - Test execution phases
        - Reporting and closure
        
        9. COMPLIANCE AND REGULATORY APPROACH
        - FDA validation requirements
        - HIPAA compliance testing
        - IEC 62304 medical device standards
        - Audit preparation and documentation
        
        10. DELIVERABLES AND ARTIFACTS
        - Test cases and scripts
        - Test execution reports
        - Defect reports
        - Compliance documentation
        - Traceability matrices
        
        Format the response as a structured JSON object with clear sections and actionable details.
        Focus on healthcare industry best practices and regulatory requirements.
        """
        
        # Generate test plan using GenAI
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=test_plan_prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=8192
            )
        )
        
        # Parse the response
        test_plan_content = response.text
        
        # Try to extract JSON from the response
        try:
            # Look for JSON content in the response
            json_start = test_plan_content.find('{')
            json_end = test_plan_content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                test_plan_json = json.loads(test_plan_content[json_start:json_end])
            else:
                # If no JSON found, create structured response
                test_plan_json = {
                    "content": test_plan_content,
                    "format": "text"
                }
        except json.JSONDecodeError:
            # If JSON parsing fails, return as text content
            test_plan_json = {
                "content": test_plan_content,
                "format": "text"
            }
        
        # Create comprehensive test plan structure
        test_plan = {
            "test_plan_id": test_plan_id,
            "title": f"Healthcare Test Plan - {project_key}",
            "project_key": project_key,
            "selected_epics": epic_keys,
            "scope": f"Comprehensive testing for {len(epic_keys)} selected epics in healthcare project {project_key}",
            "generated_content": test_plan_json,
            "creation_timestamp": datetime.now().isoformat(),
            "status": "draft",
            "version": "1.0",
            "author": "Healthcare Test Generator AI",
            "requirements_coverage": req_summary.get("total_requirements", 0) if req_summary else 0
        }
        
        # Store test plan in GCS for reference
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(GCS_BUCKET)
            
            # Create timestamped folder
            folder_name = f"test_plans/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            blob_name = f"{folder_name}/{test_plan_id}.json"
            
            blob = bucket.blob(blob_name)
            blob.upload_from_string(
                json.dumps(test_plan, indent=2),
                content_type='application/json'
            )
            
            test_plan["gcs_location"] = f"gs://{GCS_BUCKET}/{blob_name}"
            logging.info(f"Test plan stored in GCS: {test_plan['gcs_location']}")
            
        except Exception as e:
            logging.warning(f"Could not store test plan in GCS: {e}")
        
        logging.info(f"Successfully generated test plan {test_plan_id} for {len(epic_keys)} epics")
        
        return {
            **test_plan,
            "message": f"Successfully generated comprehensive test plan {test_plan_id} for {len(epic_keys)} epics"
        }
        
    except Exception as e:
        logging.error(f"Error generating comprehensive test plan: {e}")
        return {"error": f"Failed to generate comprehensive test plan: {e}"}

def quality_review_with_self_correction(test_cases_file: str, validation_result: Optional[dict] = None, max_iterations: int = 3) -> dict:
    """
    Quality review workflow with Human-in-the-Loop approval and automatic self-correction.
    
    This function:
    1. Shows validation results to user for approval
    2. If quality is poor (< 75%) or user rejects, automatically improves test cases
    3. Re-validates improved test cases
    4. Repeats until quality is acceptable OR user approves OR max iterations reached
    5. Returns approved test cases file path
    
    Args:
        test_cases_file (str): Path to test cases JSON file
        validation_result (dict, optional): Pre-computed validation result (if None, validates first)
        max_iterations (int): Maximum self-correction attempts (default: 3)
    
    Returns:
        dict: Final quality review result with approved test cases file path
    """
    try:
        logger.info("="*70)
        logger.info("QUALITY REVIEW WITH SELF-CORRECTION WORKFLOW")
        logger.info(f"   Test Cases File: {test_cases_file}")
        logger.info("="*70)
        
        current_file = test_cases_file
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"QUALITY REVIEW ITERATION {iteration}/{max_iterations}")
            logger.info(f"{'='*70}")
            
            # Step 1: Validate (if not provided)
            if validation_result is None or iteration > 1:
                logger.info("")
                logger.info("STEP 1: VALIDATION")
                logger.info("-" * 70)
                logger.info(f"   Validating test cases from: {current_file}")
                
                validation_result = validate_test_case_quality_from_file(current_file)
                
                if "error" in validation_result:
                    logger.error(f"✗ Validation failed: {validation_result['error']}")
                    return {
                        "success": False,
                        "error": f"Validation failed: {validation_result['error']}",
                        "iteration": iteration
                    }
                logger.info("✓ Validation complete")
            
            quality_score = validation_result.get("quality_score", 0)
            passed_validation = validation_result.get("passed_validation", False)
            issues_found = validation_result.get("issues_found", [])
            recommendations = validation_result.get("recommendations", [])
            
            logger.info("")
            logger.info("VALIDATION RESULTS:")
            logger.info(f"   Quality Score: {quality_score}%")
            logger.info(f"   Passed Validation: {passed_validation}")
            logger.info(f"   Issues Found: {len(issues_found)}")
            if issues_found:
                logger.info("   Top Issues:")
                for i, issue in enumerate(issues_found[:5], 1):
                    logger.info(f"      {i}. {issue}")
            logger.info("")
            
            # Step 2: Determine if approval is needed or auto-improve
            logger.info("="*70)
            logger.info("STEP 2: QUALITY REVIEW DECISION")
            logger.info("="*70)
            
            # Auto-approve if quality is good (>= 80%) and validation passed
            auto_approve_threshold = 80.0
            logger.info(f"Auto-approve threshold: {auto_approve_threshold}%")
            logger.info(f"Current quality score: {quality_score}%")
            logger.info(f"Validation passed: {passed_validation}")
            logger.info(f"Issues count: {len(issues_found)}")
            logger.info("")
            
            if quality_score >= auto_approve_threshold and passed_validation and len(issues_found) < 5:
                logger.info("="*70)
                logger.info("✅ AUTO-APPROVED")
                logger.info(f"   Reason: Quality Score {quality_score}% >= {auto_approve_threshold}%")
                logger.info(f"   Validation Passed: {passed_validation}")
                logger.info(f"   Issues: {len(issues_found)} < 5")
                logger.info("="*70)
                
                return {
                    "success": True,
                    "approved": True,
                    "auto_approved": True,
                    "test_cases_file": current_file,
                    "quality_score": quality_score,
                    "validation_result": validation_result,
                    "iterations": iteration,
                    "message": f"Test cases auto-approved (quality {quality_score:.1f}% >= {auto_approve_threshold}%)"
                }
            
            # Step 3: Self-correction needed (quality poor - below 80%)
            needs_improvement = (
                quality_score < auto_approve_threshold or  # Below auto-approve threshold
                not passed_validation or
                len(issues_found) > 5  # Too many issues
            )
            
            logger.info("="*70)
            logger.info("STEP 3: IMPROVEMENT DECISION")
            logger.info("="*70)
            logger.info(f"   Needs Improvement: {needs_improvement}")
            if needs_improvement:
                reasons = []
                if quality_score < auto_approve_threshold:
                    reasons.append(f"Quality score {quality_score}% < {auto_approve_threshold}%")
                if not passed_validation:
                    reasons.append("Validation not passed")
                if len(issues_found) > 5:
                    reasons.append(f"Too many issues ({len(issues_found)} > 5)")
                logger.info("   Reasons:")
                for reason in reasons:
                    logger.info(f"      - {reason}")
            logger.info("")
            
            if needs_improvement and iteration < max_iterations:
                logger.info("="*70)
                logger.info("⚠️ SELF-CORRECTION STARTING")
                logger.info(f"   Iteration: {iteration}/{max_iterations}")
                logger.info("="*70)
                
                # Determine improvement focus from validation results
                logger.info("Determining improvement focus areas...")
                improvement_focus = []
                
                completeness_score = validation_result.get("validation_results", {}).get("completeness", {}).get("score", 0)
                clarity_score = validation_result.get("validation_results", {}).get("clarity", {}).get("score", 0)
                compliance_score = validation_result.get("validation_results", {}).get("healthcare_compliance", {}).get("score", 0)
                traceability_score = validation_result.get("validation_results", {}).get("traceability", {}).get("score", 0)
                
                logger.info(f"   Completeness: {completeness_score}%")
                logger.info(f"   Clarity: {clarity_score}%")
                logger.info(f"   Compliance: {compliance_score}%")
                logger.info(f"   Traceability: {traceability_score}%")
                logger.info("")
                
                if completeness_score < 80:
                    improvement_focus.append("completeness")
                    logger.info(f"   + Adding focus: completeness (score {completeness_score}% < 80%)")
                if clarity_score < 80:
                    improvement_focus.append("clarity")
                    logger.info(f"   + Adding focus: clarity (score {clarity_score}% < 80%)")
                if compliance_score < 70:
                    improvement_focus.append("compliance")
                    logger.info(f"   + Adding focus: compliance (score {compliance_score}% < 70%)")
                if traceability_score < 80:
                    improvement_focus.append("traceability")
                    logger.info(f"   + Adding focus: traceability (score {traceability_score}% < 80%)")
                
                if not improvement_focus:
                    improvement_focus = ["completeness", "clarity", "compliance", "traceability"]
                    logger.info("   No specific focus areas identified, using all areas")
                
                logger.info(f"✓ Improvement focus: {', '.join(improvement_focus)}")
                logger.info("")
                
                # Read test cases from current file
                logger.info(f"Reading test cases from: {current_file}")
                with open(current_file, 'r', encoding='utf-8') as f:
                    test_cases_data = json.load(f)
                logger.info(f"✓ Loaded {len(test_cases_data.get('test_cases', []))} test cases")
                logger.info("")
                
                # Prepare feedback for improvement
                improvement_feedback = {
                    "quality_score": quality_score,
                    "issues_found": issues_found,
                    "recommendations": recommendations,
                    "user_comments": "",  # No user comments in auto mode
                    "focus_areas": improvement_focus
                }
                
                # Improve test cases
                logger.info("="*70)
                logger.info("CALLING IMPROVE_TEST_CASES_WITH_FEEDBACK")
                logger.info(f"   Focus: {', '.join(improvement_focus)}")
                logger.info("="*70)
                logger.info("")
                
                improved_result = improve_test_cases_with_feedback(
                    test_cases_json=json.dumps(test_cases_data),
                    feedback_json=json.dumps(improvement_feedback),
                    improvement_focus=",".join(improvement_focus)
                )
                
                logger.info("")
                logger.info("="*70)
                logger.info("IMPROVE_TEST_CASES_WITH_FEEDBACK COMPLETED")
                logger.info("="*70)
                
                if "error" in improved_result:
                    logger.error("")
                    logger.error("="*70)
                    logger.error("✗ SELF-CORRECTION FAILED")
                    logger.error(f"   Error: {improved_result['error']}")
                    logger.error("="*70)
                    return {
                        "success": False,
                        "error": f"Self-correction failed: {improved_result['error']}",
                        "iteration": iteration,
                        "current_quality_score": quality_score
                    }
                
                logger.info("✓ Self-correction succeeded")
                logger.info("")
                
                # Extract improved test cases from result
                improved_test_cases = improved_result.get("improved_test_cases", improved_result.get("test_cases", []))
                if not improved_test_cases and isinstance(improved_result, dict):
                    # Try to find test cases in nested structure
                    if "test_cases" in improved_result:
                        improved_test_cases = improved_result["test_cases"]
                    elif isinstance(improved_result, list):
                        improved_test_cases = improved_result
                
                if not improved_test_cases:
                    logger.error("No improved test cases found in improvement result")
                    return {
                        "success": False,
                        "error": "Self-correction did not return improved test cases",
                        "iteration": iteration,
                        "current_quality_score": quality_score
                    }
                
                # Save improved test cases to new file in correct format
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                improved_file = os.path.join("output", f"test_cases_improved_iter{iteration}_{timestamp}.json")
                os.makedirs("output", exist_ok=True)
                
                # Save in same format as original (with test_cases key)
                improved_data = {
                    "test_cases": improved_test_cases,
                    "improvement_metadata": {
                        "iteration": iteration,
                        "original_file": test_cases_file,
                        "improvements_made": improved_result.get("improvements_made", []),
                        "improvement_timestamp": timestamp
                    }
                }
                
                with open(improved_file, 'w', encoding='utf-8') as f:
                    json.dump(improved_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✓ Improved test cases saved to: {improved_file}")
                
                # Update current file and reset validation result for next iteration
                current_file = improved_file
                validation_result = None  # Re-validate in next iteration
                
            else:
                # Max iterations reached or quality not improving
                logger.warning("="*70)
                logger.warning(f"⚠️ MAX ITERATIONS REACHED OR QUALITY NOT IMPROVING")
                logger.warning(f"   Final Quality Score: {quality_score}%")
                logger.warning("="*70)
                
                return {
                    "success": False,
                    "approved": False,
                    "test_cases_file": current_file,
                    "quality_score": quality_score,
                    "validation_result": validation_result,
                    "iterations": iteration,
                    "max_iterations_reached": iteration >= max_iterations,
                    "message": f"Quality review incomplete after {iteration} iteration(s). Quality score: {quality_score}%"
                }
        
        # Should not reach here, but return current state
        return {
            "success": False,
            "approved": False,
            "test_cases_file": current_file,
            "quality_score": quality_score,
            "iterations": iteration,
            "message": "Quality review workflow completed without approval"
        }
        
    except Exception as e:
        logger.error(f"✗ Quality review workflow failed: {e}")
        logger.exception("Full traceback:")
        return {
            "success": False,
            "error": f"Quality review workflow failed: {e}",
            "test_cases_file": test_cases_file
        }

def export_after_validation(test_cases_file: str, project_key: str, epic_key: str = "") -> dict:
    """
    Automatically export test cases after validation completes.
    
    This function orchestrates all export steps that should happen after quality validation:
    1. Generate test cases preview (for user review)
    2. Export test cases to CSV
    3. Export traceability matrix to CSV
    4. Export test cases to Jira
    5. Return formatted summary with Jira links
    
    Args:
        test_cases_file (str): Path to test cases JSON file (from generate_test_cases_parallel_batches)
        project_key (str): Jira project key (e.g., "MED")
        epic_key (str, optional): Epic key for linking test cases in Jira
    
    Returns:
        dict: Summary of all export operations with previews and Jira links
    """
    try:
        logger.info("="*70)
        logger.info("AUTOMATIC EXPORT AFTER VALIDATION")
        logger.info(f"   Test Cases File: {test_cases_file}")
        logger.info(f"   Project Key: {project_key}")
        logger.info("="*70)
        
        results = {
            "test_cases_file": test_cases_file,
            "project_key": project_key,
            "epic_key": epic_key,
            "exports": {},
            "previews": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 0: Generate test cases preview (for user review)
        logger.info("Step 0: Generating test cases preview...")
        preview_result = generate_test_cases_preview(test_cases_file, max_preview=10)
        results["previews"]["test_cases"] = preview_result
        
        # Step 1: Export test cases to CSV
        logger.info("Step 1: Exporting test cases to CSV...")
        csv_result = export_test_cases_to_csv(test_cases_file)
        results["exports"]["csv"] = csv_result
        
        # Step 2: Export traceability matrix to CSV
        logger.info("Step 2: Exporting traceability matrix to CSV...")
        matrix_result = export_traceability_matrix_to_csv(test_cases_file)
        results["exports"]["traceability_matrix"] = matrix_result
        
        # Step 3: Export to Jira
        logger.info("Step 3: Exporting test cases to Jira...")
        jira_result = export_to_jira(test_cases_file, project_key, epic_key)
        results["exports"]["jira"] = jira_result
        
        logger.info("="*70)
        logger.info("✅ ALL EXPORTS COMPLETE")
        logger.info(f"   CSV: {csv_result.get('file_path', 'N/A')}")
        logger.info(f"   Traceability Matrix: {matrix_result.get('file_path', 'N/A')}")
        logger.info(f"   Jira Exported: {jira_result.get('total_exported', 0)} test cases")
        
        # Add formatted summary for user display
        results["summary"] = {
            "test_cases_generated": preview_result.get("summary", {}).get("total_test_cases", 0),
            "test_cases_exported": jira_result.get("total_exported", 0),
            "csv_file": csv_result.get("file_path", "N/A"),
            "traceability_file": matrix_result.get("file_path", "N/A"),
            "jira_urls": jira_result.get("urls", {}),
            "jira_test_cases": [
                {
                    "key": case.get("key"),
                    "title": case.get("summary"),
                    "url": case.get("jira_url")
                }
                for case in jira_result.get("exported_cases", [])[:20]  # First 20 for display
            ]
        }
        
        logger.info("="*70)
        
        results["success"] = True
        results["message"] = "All exports completed successfully"
        return results
        
    except Exception as e:
        logger.error(f"✗ Export after validation failed: {e}")
        logger.exception("Full traceback:")
        return {
            "success": False,
            "error": f"Failed to export after validation: {e}",
            "test_cases_file": test_cases_file,
            "project_key": project_key
        }

def validate_single_test_case_batch_with_llm(test_cases_batch: list, batch_number: int, total_batches: int) -> dict:
    """
    Validate a single batch of test cases using LLM for intelligent quality assessment.
    
    This function uses LLM to provide smarter, context-aware validation instead of rule-based checks.
    
    Args:
        test_cases_batch (list): List of test cases to validate
        batch_number (int): Current batch number
        total_batches (int): Total number of batches
    
    Returns:
        dict: Validation results for this batch
    """
    try:
        logger.info(f"Validating batch {batch_number}/{total_batches}: {len(test_cases_batch)} test cases")
        
        # Initialize GenAI client
        client = init_genai_client()
        if not client:
            raise RuntimeError("Failed to initialize GenAI client.")
        
        # Create LLM validation prompt
        prompt = f"""
        You are a senior healthcare software testing quality assurance expert. Validate the following test cases for quality, completeness, and healthcare compliance.

        TEST CASES TO VALIDATE (Batch {batch_number}/{total_batches}):
        {json.dumps(test_cases_batch, indent=2)}

        VALIDATION CRITERIA:

        1. COMPLETENESS (Check each test case has):
           - Test case ID
           - Clear title
           - Detailed description
           - Complete preconditions
           - Step-by-step test steps with inputs
           - Expected results for each step
           - Test data requirements
           - Compliance standards
           - Requirement traceability

        2. CLARITY (Assess):
           - Are test steps unambiguous?
           - Are expected results specific and measurable?
           - Is terminology consistent and professional?
           - Can a tester execute this without confusion?

        3. HEALTHCARE COMPLIANCE:
           - Patient safety considerations
           - Data privacy (HIPAA) validation
           - FDA 21 CFR Part 11 requirements
           - Audit trail verification
           - Security testing (authentication, authorization)
           - Error handling and recovery

        4. TRACEABILITY:
           - Linked to requirements (requirement_ids present)
           - Category and priority defined
           - Risk level assigned

        5. COVERAGE:
           - Positive test scenarios
           - Negative test scenarios (error cases)
           - Boundary conditions
           - Edge cases

        6. STRUCTURE:
           - Professional formatting
           - Logical test flow
           - Appropriate test data
           - Postconditions defined

        OUTPUT FORMAT (JSON):
        {{
            "batch_number": {batch_number},
            "test_cases_validated": {len(test_cases_batch)},
            "validation_results": [
                {{
                    "test_case_id": "TC-xxx",
                    "overall_score": 85.0,
                    "completeness_score": 90.0,
                    "clarity_score": 85.0,
                    "compliance_score": 80.0,
                    "traceability_score": 90.0,
                    "coverage_score": 85.0,
                    "structure_score": 85.0,
                    "passed": true,
                    "issues": [
                        "Specific issue 1",
                        "Specific issue 2"
                    ],
                    "recommendations": [
                        "Specific recommendation 1"
                    ]
                }}
            ],
            "batch_summary": {{
                "average_score": 85.0,
                "passed_count": 8,
                "failed_count": 2,
                "critical_issues_count": 1,
                "top_issues": [
                    "Most common issue 1",
                    "Most common issue 2"
                ],
                "recommendations": [
                    "Batch-level recommendation 1"
                ]
            }}
        }}

        IMPORTANT: 
        - Be specific about issues (not just "missing field" but "Test step 3 lacks specific input data")
        - Scores should be 0-100
        - A test case passes if overall_score >= 75
        - Focus on healthcare-specific quality issues
        """
        
        # Generate validation using LLM
        response = client.generate_content(prompt)
        
        if not response or not response.text:
            raise RuntimeError("Empty response from GenAI service")
        
        # Parse response
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        
        logger.info(f"✓ Batch {batch_number} validated: Avg score {result.get('batch_summary', {}).get('average_score', 0)}%")
        
        return {
            "batch_status": "success",
            "batch_number": batch_number,
            "validation_result": result
        }
        
    except Exception as e:
        logger.error(f"✗ Batch {batch_number} validation failed: {e}")
        return {
            "batch_status": "error",
            "batch_number": batch_number,
            "error": str(e)
        }

def validate_test_case_quality_from_file(test_cases_file: str, requirements_json: str = "", batch_size: int = 10) -> dict:
    """
    Validate test case quality by reading from file and processing in parallel batches using LLM.
    
    This function:
    1. Reads test cases from file
    2. Splits into batches (avoids large JSON in function calls)
    3. Validates each batch in parallel using LLM for intelligent assessment
    4. Consolidates results
    
    Args:
        test_cases_file (str): Path to JSON file containing test cases
        requirements_json (str, optional): Optional - minimal requirements summary
        batch_size (int): Number of test cases per batch (default: 10)
    
    Returns:
        dict: Consolidated validation results with quality scores and issues
    """
    try:
        logger.info("="*70)
        logger.info("⚡ PARALLEL BATCH VALIDATION WITH LLM")
        logger.info(f"   File: {test_cases_file}")
        logger.info(f"   Batch Size: {batch_size}")
        logger.info("="*70)
        
        if not os.path.exists(test_cases_file):
            return {"error": f"Test cases file not found: {test_cases_file}"}
        
        # Read test cases from file
        with open(test_cases_file, 'r', encoding='utf-8') as f:
            test_cases_data = json.load(f)
        
        # Extract test cases array
        test_cases = test_cases_data.get('test_cases', [])
        if not test_cases:
            return {"error": "No test cases found in file"}
        
        total_test_cases = len(test_cases)
        logger.info(f"✓ Loaded {total_test_cases} test cases from file")
        
        # Split into batches
        batches = []
        for i in range(0, total_test_cases, batch_size):
            batch = test_cases[i:i + batch_size]
            batches.append({
                "batch_number": len(batches) + 1,
                "test_cases": batch
            })
        
        total_batches = len(batches)
        logger.info(f"✓ Split into {total_batches} batches for parallel validation")
        
        # Validate batches in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        batch_results = []
        failed_batches = []
        
        with ThreadPoolExecutor(max_workers=min(total_batches, 5)) as executor:
            future_to_batch = {
                executor.submit(
                    validate_single_test_case_batch_with_llm,
                    batch["test_cases"],
                    batch["batch_number"],
                    total_batches
                ): batch["batch_number"]
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    batch_result = future.result()
                    batch_results.append(batch_result)
                    
                    if batch_result.get("batch_status") == "success":
                        logger.info(f"✓ Batch {batch_num} validation complete")
                    else:
                        logger.error(f"✗ Batch {batch_num} validation failed")
                        failed_batches.append(batch_num)
                except Exception as e:
                    logger.error(f"✗ Batch {batch_num} exception: {e}")
                    failed_batches.append(batch_num)
        
        logger.info("="*70)
        logger.info(f"📊 PARALLEL VALIDATION COMPLETE")
        logger.info(f"   Successful Batches: {len(batch_results) - len(failed_batches)}/{total_batches}")
        logger.info("="*70)
        
        # Consolidate results
        all_validation_results = []
        all_issues = []
        all_recommendations = []
        total_score_sum = 0
        passed_count = 0
        failed_count = 0
        
        for batch_result in batch_results:
            if batch_result.get("batch_status") == "success":
                validation_data = batch_result.get("validation_result", {})
                test_case_results = validation_data.get("validation_results", [])
                
                for tc_result in test_case_results:
                    all_validation_results.append(tc_result)
                    total_score_sum += tc_result.get("overall_score", 0)
                    if tc_result.get("passed", False):
                        passed_count += 1
                    else:
                        failed_count += 1
                    
                    all_issues.extend(tc_result.get("issues", []))
                    all_recommendations.extend(tc_result.get("recommendations", []))
        
        # Calculate overall metrics
        total_validated = len(all_validation_results)
        overall_quality_score = (total_score_sum / total_validated) if total_validated > 0 else 0
        passed_validation = (passed_count / total_validated >= 0.8) if total_validated > 0 else False
        
        # Calculate category averages
        completeness_avg = sum([tc.get("completeness_score", 0) for tc in all_validation_results]) / total_validated if total_validated > 0 else 0
        clarity_avg = sum([tc.get("clarity_score", 0) for tc in all_validation_results]) / total_validated if total_validated > 0 else 0
        compliance_avg = sum([tc.get("compliance_score", 0) for tc in all_validation_results]) / total_validated if total_validated > 0 else 0
        traceability_avg = sum([tc.get("traceability_score", 0) for tc in all_validation_results]) / total_validated if total_validated > 0 else 0
        coverage_avg = sum([tc.get("coverage_score", 0) for tc in all_validation_results]) / total_validated if total_validated > 0 else 0
        structure_avg = sum([tc.get("structure_score", 0) for tc in all_validation_results]) / total_validated if total_validated > 0 else 0
        
        # Get top issues (most common)
        from collections import Counter
        issue_counts = Counter(all_issues)
        top_issues = [issue for issue, count in issue_counts.most_common(10)]
        
        result = {
            "validation_id": f"QV-LLM-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "total_test_cases": total_test_cases,
            "quality_score": round(overall_quality_score, 1),
            "passed_validation": passed_validation,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "validation_results": {
                "completeness": {"score": round(completeness_avg, 1)},
                "clarity": {"score": round(clarity_avg, 1)},
                "healthcare_compliance": {"score": round(compliance_avg, 1)},
                "traceability": {"score": round(traceability_avg, 1)},
                "test_coverage": {"score": round(coverage_avg, 1)},
                "structure": {"score": round(structure_avg, 1)}
            },
            "issues_found": top_issues,
            "recommendations": list(set(all_recommendations))[:10],  # Top 10 unique recommendations
            "test_case_details": all_validation_results,
            "parallel_processing_info": {
                "batch_size": batch_size,
                "total_batches": total_batches,
                "successful_batches": len(batch_results) - len(failed_batches),
                "failed_batches": len(failed_batches),
                "validation_method": "llm_parallel_batch"
            },
            "validation_timestamp": datetime.now().isoformat(),
            "message": f"Validated {total_test_cases} test cases using LLM (parallel batch processing). Quality score: {overall_quality_score:.1f}%"
        }
        
        logger.info("="*70)
        logger.info("✅ VALIDATION COMPLETE")
        logger.info(f"   Quality Score: {overall_quality_score:.1f}%")
        logger.info(f"   Passed: {passed_count}/{total_test_cases}")
        logger.info(f"   Issues: {len(top_issues)}")
        logger.info("="*70)
        
        return result
        
    except Exception as e:
        logger.error(f"✗ Failed to validate test cases from file: {e}")
        logger.exception("Full traceback:")
        return {"error": f"Failed to validate test cases from file: {e}"}

def validate_test_case_quality(test_cases_json: str, requirements_json: str = "") -> dict:
    """
    Validate the quality and completeness of generated test cases.
    
    ⚠️ IMPORTANT FOR AGENT: 
    - If test cases come from generate_test_cases_parallel_batches(), use validate_test_case_quality_from_file() instead
    - This function accepts file paths OR JSON strings
    - If passing JSON string, keep it small (< 50KB) to avoid malformed function calls
    - The requirements_json parameter should be omitted or very small (< 500 chars)
    
    Method Signature:
        validate_test_case_quality(test_cases_json: str, requirements_json: str = "") -> dict
    
    Args:
        test_cases_json (str): JSON string of test cases OR file path to JSON file
        requirements_json (str, optional): Optional - minimal requirements summary for traceability.
                                         Omit this parameter entirely to avoid errors.
                                         If omitted, traceability uses requirement IDs from test cases.
    
    Returns:
        dict: Dictionary containing quality validation results with keys:
            - validation_id (str): Unique validation identifier
            - total_test_cases (int): Total number of test cases validated
            - quality_score (float): Overall quality score (0-100)
            - validation_results (dict): Detailed validation results by category
            - issues_found (list): List of quality issues identified
            - recommendations (list): Recommendations for improvement
            - compliance_check (dict): Healthcare compliance validation results
            - traceability_check (dict): Requirements traceability validation
            - passed_validation (bool): Whether test cases passed quality check
            - error (str): Error message if validation failed
            
    Example:
        result = validate_test_case_quality('{"test_cases": [...]}', '{"requirements": [...]}')
        # Returns: {"validation_id": "QV-001", "quality_score": 85.5, "passed_validation": true, ...}
    """
    try:
        logging.info("Validating test case quality and completeness")
        
        # Handle file path if provided instead of JSON string
        if isinstance(test_cases_json, str) and os.path.exists(test_cases_json):
            logger.info(f"Reading test cases from file: {test_cases_json}")
            with open(test_cases_json, 'r', encoding='utf-8') as f:
                test_cases_data = json.load(f)
        # Handle large JSON strings by checking length
        elif isinstance(test_cases_json, str) and len(test_cases_json) > 50000:  # ~50KB threshold
            logger.warning(f"Test cases JSON is very large ({len(test_cases_json)} chars). Attempting to parse and validate...")
            try:
                test_cases_data = json.loads(test_cases_json)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse large test_cases_json: {e}")
                return {"error": f"Failed to parse test_cases_json. Consider using validate_test_case_quality_from_file() instead."}
        # Normal parsing
        elif isinstance(test_cases_json, str):
            test_cases_data = json.loads(test_cases_json)
        else:
            test_cases_data = test_cases_json
        
        # Extract test cases array
        if isinstance(test_cases_data, dict) and 'test_cases' in test_cases_data:
            test_cases = test_cases_data['test_cases']
        elif isinstance(test_cases_data, list):
            test_cases = test_cases_data
        else:
            test_cases = [test_cases_data]
        
        # Parse requirements if provided (but limit size to avoid malformed calls)
        requirements = []
        if requirements_json:
            try:
                # If requirements_json is too large, extract only requirement IDs for traceability
                if len(str(requirements_json)) > 500:
                    logging.info("Requirements JSON is large, extracting only requirement IDs for traceability check")
                    try:
                        if isinstance(requirements_json, str):
                            req_data = json.loads(requirements_json)
                        else:
                            req_data = requirements_json
                        
                        if isinstance(req_data, dict) and 'requirements' in req_data:
                            req_list = req_data['requirements']
                        elif isinstance(req_data, list):
                            req_list = req_data
                        else:
                            req_list = []
                        
                        # Extract only minimal info: id, key, summary
                        requirements = [
                            {
                                "id": req.get("id") or req.get("key", ""),
                                "key": req.get("key") or req.get("id", ""),
                                "summary": str(req.get("summary", ""))[:100]  # Truncate summary
                            }
                            for req in req_list[:50]  # Limit to first 50
                        ]
                        logging.info(f"Extracted {len(requirements)} requirement summaries for traceability")
                    except Exception as e:
                        logging.warning(f"Could not parse requirements JSON, using test case requirement IDs only: {e}")
                        requirements = []  # Will use requirement IDs from test cases instead
                else:
                    # Small enough to parse normally
                    if isinstance(requirements_json, str):
                        req_data = json.loads(requirements_json)
                    else:
                        req_data = requirements_json
                
                if isinstance(req_data, dict) and 'requirements' in req_data:
                    requirements = req_data['requirements']
                elif isinstance(req_data, list):
                    requirements = req_data
            except json.JSONDecodeError as e:
                logging.warning(f"Could not parse requirements JSON for traceability check: {e}. Using test case requirement IDs only.")
                requirements = []  # Will use requirement IDs from test cases instead
        
        # Generate validation ID
        validation_id = f"QV-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize validation results
        validation_results = {
            "completeness": {"score": 0, "details": []},
            "clarity": {"score": 0, "details": []},
            "healthcare_compliance": {"score": 0, "details": []},
            "traceability": {"score": 0, "details": []},
            "test_coverage": {"score": 0, "details": []},
            "structure": {"score": 0, "details": []}
        }
        
        issues_found = []
        recommendations = []
        
        # Validate each test case
        total_score = 0
        max_possible_score = 0
        
        for i, test_case in enumerate(test_cases):
            case_score = 0
            case_max_score = 60  # 10 points per validation category
            
            # 1. COMPLETENESS CHECK (10 points)
            completeness_score = 0
            required_fields = ["test_case_id", "title", "description", "test_steps", "expected_results"]
            
            for field in required_fields:
                if field in test_case and test_case[field] and str(test_case[field]).strip():
                    completeness_score += 2
                else:
                    issues_found.append(f"Test case {i+1}: Missing or empty required field '{field}'")
            
            validation_results["completeness"]["score"] += completeness_score
            validation_results["completeness"]["details"].append(f"Test case {i+1}: {completeness_score}/10 completeness")
            case_score += completeness_score
            
            # 2. CLARITY CHECK (10 points)
            clarity_score = 0
            
            # Check title clarity
            title = test_case.get("title", "")
            if len(title) > 10 and len(title) < 100:
                clarity_score += 2
            else:
                issues_found.append(f"Test case {i+1}: Title should be 10-100 characters long")
            
            # Check description clarity
            description = test_case.get("description", "")
            if len(description) > 20:
                clarity_score += 2
            else:
                issues_found.append(f"Test case {i+1}: Description should be more detailed")
            
            # Check test steps clarity
            test_steps = test_case.get("test_steps", [])
            if isinstance(test_steps, list) and len(test_steps) >= 3:
                clarity_score += 3
            elif isinstance(test_steps, str) and len(test_steps.split('\n')) >= 3:
                clarity_score += 3
            else:
                issues_found.append(f"Test case {i+1}: Should have at least 3 detailed test steps")
            
            # Check expected results clarity
            expected_results = test_case.get("expected_results", "")
            if len(expected_results) > 20:
                clarity_score += 3
            else:
                issues_found.append(f"Test case {i+1}: Expected results should be more detailed")
            
            validation_results["clarity"]["score"] += clarity_score
            validation_results["clarity"]["details"].append(f"Test case {i+1}: {clarity_score}/10 clarity")
            case_score += clarity_score
            
            # 3. HEALTHCARE COMPLIANCE CHECK (10 points)
            compliance_score = 0
            case_content = json.dumps(test_case).lower()
            
            # Check for patient safety considerations
            if any(keyword in case_content for keyword in ["patient", "safety", "clinical", "medical"]):
                compliance_score += 2
            
            # Check for privacy/security considerations
            if any(keyword in case_content for keyword in ["privacy", "security", "phi", "hipaa", "access"]):
                compliance_score += 2
            
            # Check for audit trail considerations
            if any(keyword in case_content for keyword in ["audit", "log", "trace", "record"]):
                compliance_score += 2
            
            # Check for compliance validation steps
            if any(keyword in case_content for keyword in ["compliance", "regulatory", "fda", "validation"]):
                compliance_score += 2
            
            # Check for error handling
            if any(keyword in case_content for keyword in ["error", "exception", "failure", "invalid"]):
                compliance_score += 2
            
            validation_results["healthcare_compliance"]["score"] += compliance_score
            validation_results["healthcare_compliance"]["details"].append(f"Test case {i+1}: {compliance_score}/10 compliance")
            case_score += compliance_score
            
            # 4. TRACEABILITY CHECK (10 points)
            traceability_score = 0
            
            # Check for requirement ID
            if "requirement_id" in test_case or "req_id" in test_case:
                traceability_score += 5
            else:
                issues_found.append(f"Test case {i+1}: Missing requirement traceability")
            
            # Check for category/priority
            if "category" in test_case or "priority" in test_case:
                traceability_score += 3
            
            # Check for risk level
            if "risk_level" in test_case or "severity" in test_case:
                traceability_score += 2
            
            validation_results["traceability"]["score"] += traceability_score
            validation_results["traceability"]["details"].append(f"Test case {i+1}: {traceability_score}/10 traceability")
            case_score += traceability_score
            
            # 5. TEST COVERAGE CHECK (10 points)
            coverage_score = 0
            
            # Check for positive test scenario
            if "positive" in case_content or "valid" in case_content or "success" in case_content:
                coverage_score += 3
            
            # Check for negative test scenario
            if "negative" in case_content or "invalid" in case_content or "error" in case_content:
                coverage_score += 3
            
            # Check for boundary conditions
            if any(keyword in case_content for keyword in ["boundary", "limit", "edge", "minimum", "maximum"]):
                coverage_score += 2
            
            # Check for integration aspects
            if any(keyword in case_content for keyword in ["integration", "interface", "api", "external"]):
                coverage_score += 2
            
            validation_results["test_coverage"]["score"] += coverage_score
            validation_results["test_coverage"]["details"].append(f"Test case {i+1}: {coverage_score}/10 coverage")
            case_score += coverage_score
            
            # 6. STRUCTURE CHECK (10 points)
            structure_score = 0
            
            # Check for preconditions
            if "preconditions" in test_case or "setup" in test_case:
                structure_score += 2
            
            # Check for postconditions
            if "postconditions" in test_case or "cleanup" in test_case:
                structure_score += 2
            
            # Check for test data
            if "test_data" in test_case or "input" in test_case:
                structure_score += 2
            
            # Check for execution notes
            if "notes" in test_case or "comments" in test_case:
                structure_score += 2
            
            # Check for automation potential
            if "automation" in test_case or "automated" in case_content:
                structure_score += 2
            
            validation_results["structure"]["score"] += structure_score
            validation_results["structure"]["details"].append(f"Test case {i+1}: {structure_score}/10 structure")
            case_score += structure_score
            
            total_score += case_score
            max_possible_score += case_max_score
        
        # Calculate overall quality score
        quality_score = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
        
        # Normalize category scores
        num_test_cases = len(test_cases)
        for category in validation_results:
            validation_results[category]["score"] = (validation_results[category]["score"] / (num_test_cases * 10) * 100) if num_test_cases > 0 else 0
        
        # Generate recommendations
        if quality_score < 70:
            recommendations.append("Test cases need significant improvement before approval")
        elif quality_score < 85:
            recommendations.append("Test cases are good but could benefit from minor improvements")
        else:
            recommendations.append("Test cases meet high quality standards")
        
        if validation_results["completeness"]["score"] < 80:
            recommendations.append("Ensure all required fields are complete and detailed")
        
        if validation_results["healthcare_compliance"]["score"] < 70:
            recommendations.append("Add more healthcare-specific validation steps and compliance checks")
        
        if validation_results["traceability"]["score"] < 80:
            recommendations.append("Improve requirement traceability by adding requirement IDs and categories")
        
        # Compliance check summary
        compliance_check = {
            "patient_safety_coverage": validation_results["healthcare_compliance"]["score"] >= 70,
            "regulatory_compliance": validation_results["healthcare_compliance"]["score"] >= 80,
            "audit_trail_validation": any("audit" in detail.lower() for detail in validation_results["healthcare_compliance"]["details"]),
            "security_considerations": any("security" in detail.lower() for detail in validation_results["healthcare_compliance"]["details"])
        }
        
        # Traceability check summary
        traceability_check = {
            "requirements_linked": validation_results["traceability"]["score"] >= 70,
            "categories_defined": validation_results["traceability"]["score"] >= 60,
            "risk_levels_assigned": validation_results["traceability"]["score"] >= 80,
            "coverage_complete": len(requirements) == 0 or len(test_cases) >= len(requirements) * 0.8
        }
        
        # Determine if validation passed
        passed_validation = (
            quality_score >= 75 and
            len([issue for issue in issues_found if "missing" in issue.lower()]) == 0 and
            validation_results["healthcare_compliance"]["score"] >= 60
        )
        
        logging.info(f"Quality validation completed: {quality_score:.1f}% score, {len(issues_found)} issues found")
        
        return {
            "validation_id": validation_id,
            "total_test_cases": len(test_cases),
            "quality_score": round(quality_score, 1),
            "validation_results": validation_results,
            "issues_found": issues_found,
            "recommendations": recommendations,
            "compliance_check": compliance_check,
            "traceability_check": traceability_check,
            "passed_validation": passed_validation,
            "validation_timestamp": datetime.now().isoformat(),
            "message": f"Quality validation completed for {len(test_cases)} test cases with {quality_score:.1f}% quality score"
        }
        
    except Exception as e:
        logging.error(f"Error validating test case quality: {e}")
        return {"error": f"Failed to validate test case quality: {e}"}

# LEGACY FUNCTION (kept for backward compatibility but not recommended)
def generate_comprehensive_test_cases(requirements_json: str, test_types: str = "functional,security,compliance", standards: str = "FDA,HIPAA,IEC 62304", risk_level: str = "medium"):
    """
    SINGLE RELIABLE FUNCTION for generating comprehensive healthcare test cases.
    
    This function ensures ALL requirements are processed and generates test cases for each one.
    
    Method Signature:
        generate_comprehensive_test_cases(requirements_json: str, test_types: str, standards: str, risk_level: str) -> dict
    
    Args:
        requirements_json (str): JSON string of requirements to generate test cases for
        test_types (str): Comma-separated test types (e.g., "functional,security,compliance")
        standards (str): Comma-separated compliance standards (e.g., "FDA,HIPAA,IEC 62304")
        risk_level (str): Risk level ("high", "medium", "low")
    
    Returns:
        dict: Dictionary containing comprehensive test cases with detailed steps and local file paths
    """
    try:
        logging.info(f"Generating comprehensive test cases with detailed format")
        
        # Parse inputs
        if isinstance(requirements_json, str):
            requirements_data = json.loads(requirements_json)
        else:
            requirements_data = requirements_json
            
        # Extract requirements list
        if isinstance(requirements_data, dict) and "requirements" in requirements_data:
            requirements = requirements_data["requirements"]
        elif isinstance(requirements_data, list):
            requirements = requirements_data
        else:
            requirements = [requirements_data]
            
        test_type_list = [t.strip() for t in test_types.split(',') if t.strip()]
        standards_list = [s.strip() for s in standards.split(',') if s.strip()]
        
        client = init_genai_client()
        if not client:
            raise RuntimeError("Failed to initialize GenAI client.")
        
        # Use the detailed test case prompt from prompts.py
        try:
            from .prompts import DETAILED_TEST_CASE_PROMPT
        except ImportError:
            # Fallback for direct execution
            import importlib.util
            spec = importlib.util.spec_from_file_location("prompts", os.path.join(os.path.dirname(__file__), "prompts.py"))
            prompts_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(prompts_module)
            DETAILED_TEST_CASE_PROMPT = prompts_module.DETAILED_TEST_CASE_PROMPT
        
        # Create comprehensive prompt that explicitly instructs to process ALL requirements
        prompt = f"""
        {DETAILED_TEST_CASE_PROMPT}
        
        CRITICAL INSTRUCTION: Generate test cases for ALL {len(requirements)} requirements provided below. 
        Do not skip any requirement. Each requirement MUST have at least 2-3 test cases generated.
        
        REQUIREMENTS TO PROCESS (ALL {len(requirements)} MUST BE COVERED):
        {json.dumps(requirements, indent=2)}
        
        GENERATION PARAMETERS:
        - Test Types: {', '.join(test_type_list)}
        - Compliance Standards: {', '.join(standards_list)}
        - Risk Level: {risk_level}
        - Software Class: B (Medical Device Software)
        
        MANDATORY REQUIREMENTS:
        1. Generate test cases for EVERY requirement listed above
        2. Each requirement should have 2-4 test cases covering different scenarios
        3. Include detailed test steps, compliance validation, and audit evidence requirements
        4. Focus on healthcare-specific scenarios and regulatory compliance validation
        5. Ensure requirement IDs match exactly (REQ-001, REQ-002, etc.)
        
        Return the result in this JSON format:
        {{
            "test_cases": [
                // Array of comprehensive test cases following the detailed format
                // MUST include test cases for ALL {len(requirements)} requirements
            ],
            "summary": {{
                "total_test_cases": 0,
                "requirements_covered": {len(requirements)},
                "by_requirement": {{}},
                "by_type": {{}},
                "by_priority": {{}},
                "by_compliance": {{}}
            }}
        }}
        """
        
        # Generate content using LLM
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )
        
        # Parse LLM response
        result = json.loads(response.candidates[0].content.parts[0].text)
        
        # Validate that all requirements are covered
        test_cases = result.get("test_cases", [])
        covered_requirements = set()
        for tc in test_cases:
            req_id = tc.get("metadata", {}).get("requirement_id") or tc.get("requirement_id")
            if req_id:
                covered_requirements.add(req_id)
        
        expected_requirements = set([req.get("id", "") for req in requirements])
        missing_requirements = expected_requirements - covered_requirements
        
        if missing_requirements:
            logging.warning(f"Missing test cases for requirements: {missing_requirements}")
            # Add this info to the result
            result["validation"] = {
                "expected_requirements": len(expected_requirements),
                "covered_requirements": len(covered_requirements),
                "missing_requirements": list(missing_requirements)
            }
        
        # Add metadata
        result["generation_timestamp"] = datetime.now().isoformat()
        result["parameters"] = {
            "requirements_count": len(requirements),
            "test_types": test_type_list,
            "standards": standards_list,
            "risk_level": risk_level
        }
        
        # Save to local output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_files = []
        
        # Save JSON test cases locally
        json_filename = f"output/comprehensive_test_cases_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        local_files.append(json_filename)
        
        # Generate and save CSV locally
        csv_content = _create_comprehensive_test_cases_csv(result.get("test_cases", []))
        csv_filename = f"output/comprehensive_test_cases_{timestamp}.csv"
        with open(csv_filename, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_content)
        local_files.append(csv_filename)
        
        # Generate and save PDF test cases locally (with improved formatting)
        pdf_content = _create_test_cases_pdf_improved(result.get("test_cases", []))
        pdf_filename = f"output/test_cases_detailed_{timestamp}.pdf"
        with open(pdf_filename, 'wb') as f:
            f.write(pdf_content)
        local_files.append(pdf_filename)
        
        # Generate traceability matrix
        traceability_matrix = _create_traceability_matrix_from_test_cases(requirements, result.get("test_cases", []))
        
        # Save traceability matrix as JSON
        traceability_json_filename = f"output/traceability_matrix_{timestamp}.json"
        with open(traceability_json_filename, 'w', encoding='utf-8') as f:
            json.dump(traceability_matrix, f, indent=2, ensure_ascii=False)
        local_files.append(traceability_json_filename)
        
        # Generate and save traceability matrix PDF (with improved formatting)
        traceability_pdf_content = _create_traceability_matrix_pdf_improved(traceability_matrix)
        traceability_pdf_filename = f"output/traceability_matrix_{timestamp}.pdf"
        with open(traceability_pdf_filename, 'wb') as f:
            f.write(traceability_pdf_content)
        local_files.append(traceability_pdf_filename)
        
        # Add traceability matrix to result
        result["traceability_matrix"] = traceability_matrix
        result["local_files"] = local_files
        
        test_cases = result.get("test_cases", [])
        logging.info(f"Successfully generated {len(test_cases)} comprehensive test cases")
        logging.info(f"Covered {len(covered_requirements)} out of {len(expected_requirements)} requirements")
        logging.info(f"Local files saved: {', '.join(local_files)}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error generating comprehensive test cases: {e}")
        return {"error": f"Failed to generate comprehensive test cases: {e}"}

# Agentic Orchestration and Status Tools

def generate_test_cases_simple(requirements_count: str, test_types: str = "functional,security,compliance", standards: str = "FDA,HIPAA,IEC 62304", risk_level: str = "medium"):
    """
    Simplified test case generation function that works around JSON parsing issues.
    
    Method Signature:
        generate_test_cases_simple(requirements_count: str, test_types: str, standards: str, risk_level: str) -> dict
    
    Args:
        requirements_count (str): Number of requirements to generate test cases for (e.g., "17")
        test_types (str): Comma-separated test types (e.g., "functional,security,compliance")
        standards (str): Comma-separated compliance standards (e.g., "FDA,HIPAA,IEC 62304")
        risk_level (str): Risk level ("high", "medium", "low")
    
    Returns:
        dict: Dictionary containing generated test cases and metadata
        
    Example:
        result = generate_test_cases_simple("17", "functional,security", "FDA,HIPAA", "high")
        # Returns: {"test_cases": [...], "total_count": 45, ...}
    """
    try:
        logging.info(f"Generating test cases for {requirements_count} requirements with risk level: {risk_level}")
        
        # Parse parameters
        req_count = int(requirements_count) if requirements_count.isdigit() else 10
        test_type_list = [t.strip() for t in test_types.split(',') if t.strip()]
        standards_list = [s.strip() for s in standards.split(',') if s.strip()]
        
        client = init_genai_client()
        if not client:
            raise RuntimeError("Failed to initialize GenAI client.")
        
        # Create simplified prompt for test case generation
        prompt = f"""
        You are a healthcare software testing expert. Generate comprehensive test cases for a healthcare EHR system.
        
        Requirements:
        - Number of requirements to cover: {req_count}
        - Test types needed: {', '.join(test_type_list)}
        - Compliance standards: {', '.join(standards_list)}
        - Risk level: {risk_level}
        
        Generate test cases in the following JSON format:
        {{
            "test_cases": [
                {{
                    "id": "TC-001",
                    "title": "Test Case Title",
                    "description": "Detailed test description",
                    "test_type": "functional|security|compliance|performance",
                    "priority": "critical|high|medium|low",
                    "requirement_id": "REQ-001",
                    "test_steps": ["Step 1", "Step 2", "Step 3"],
                    "expected_result": "Expected outcome",
                    "compliance_standard": "FDA|HIPAA|IEC 62304",
                    "test_data": "Test data requirements",
                    "prerequisites": "Prerequisites for test",
                    "risk_level": "{risk_level}"
                }}
            ],
            "summary": {{
                "total_test_cases": 0,
                "by_type": {{}},
                "by_priority": {{}},
                "by_compliance": {{}}
            }}
        }}
        
        Focus on healthcare-specific scenarios including:
        - Patient data management and security
        - Clinical workflows and decision support
        - Regulatory compliance (FDA, HIPAA, IEC 62304)
        - Data encryption and access controls
        - Audit trails and electronic signatures
        - System performance and reliability
        
        Generate {req_count * 3} comprehensive test cases covering all specified types and compliance standards.
        """
        
        # Generate content using LLM
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )
        
        # Parse LLM response
        result = json.loads(response.candidates[0].content.parts[0].text)
        
        # Add metadata
        result["generation_timestamp"] = datetime.now().isoformat()
        result["parameters"] = {
            "requirements_count": req_count,
            "test_types": test_type_list,
            "standards": standards_list,
            "risk_level": risk_level
        }
        
        test_cases = result.get("test_cases", [])
        logging.info(f"Successfully generated {len(test_cases)} test cases")
        return result
        
    except Exception as e:
        logging.error(f"Error generating test cases: {e}")
        return {"error": f"Failed to generate test cases: {e}"}

def create_traceability_matrix_simple(requirements_count: str, test_cases_count: str, standards: str = "FDA,HIPAA,IEC 62304"):
    """
    Simplified traceability matrix creation that works around JSON parsing issues.
    
    Method Signature:
        create_traceability_matrix_simple(requirements_count: str, test_cases_count: str, standards: str) -> dict
    
    Args:
        requirements_count (str): Number of requirements (e.g., "17")
        test_cases_count (str): Number of test cases (e.g., "45")
        standards (str): Comma-separated compliance standards
    
    Returns:
        dict: Dictionary containing traceability matrix
        
    Example:
        result = create_traceability_matrix_simple("17", "45", "FDA,HIPAA")
        # Returns: {"requirements": [...], "traceability_links": [...], ...}
    """
    try:
        logging.info(f"Creating traceability matrix for {requirements_count} requirements and {test_cases_count} test cases")
        
        # Parse parameters
        req_count = int(requirements_count) if requirements_count.isdigit() else 10
        tc_count = int(test_cases_count) if test_cases_count.isdigit() else 30
        standards_list = [s.strip() for s in standards.split(',') if s.strip()]
        
        client = init_genai_client()
        if not client:
            raise RuntimeError("Failed to initialize GenAI client.")
        
        # Create simplified prompt for traceability matrix
        prompt = f"""
        You are a healthcare software quality assurance expert. Create a comprehensive traceability matrix for a healthcare EHR system.
        
        System Overview:
        - Total Requirements: {req_count}
        - Total Test Cases: {tc_count}
        - Compliance Standards: {', '.join(standards_list)}
        
        Create a traceability matrix in the following JSON format:
        {{
            "requirements": [
                {{
                    "id": "REQ-001",
                    "title": "Requirement Title",
                    "type": "functional|non-functional|security|compliance",
                    "priority": "critical|high|medium|low",
                    "category": "security|compliance|performance|usability|reliability",
                    "linked_test_cases": ["TC-001", "TC-002"],
                    "coverage_status": "covered|partial|not_covered",
                    "compliance_standards": ["FDA", "HIPAA"]
                }}
            ],
            "test_cases": [
                {{
                    "id": "TC-001",
                    "title": "Test Case Title",
                    "type": "functional|security|compliance|performance",
                    "priority": "critical|high|medium|low",
                    "requirement_id": "REQ-001",
                    "compliance_standard": "FDA|HIPAA|IEC 62304",
                    "coverage_type": "positive|negative|edge_case"
                }}
            ],
            "traceability_links": [
                {{
                    "requirement_id": "REQ-001",
                    "test_case_id": "TC-001",
                    "link_type": "covers|validates|tests",
                    "compliance_standard": "FDA",
                    "coverage_percentage": 100
                }}
            ],
            "coverage_analysis": {{
                "total_requirements": {req_count},
                "covered_requirements": 0,
                "uncovered_requirements": 0,
                "coverage_percentage": 0.0,
                "requirements_without_tests": [],
                "test_coverage_by_type": {{
                    "functional": 0,
                    "security": 0,
                    "compliance": 0,
                    "performance": 0
                }}
            }},
            "compliance_mapping": {{
                "FDA": {{
                    "requirements": [],
                    "test_cases": [],
                    "coverage_percentage": 0.0
                }}
            }}
        }}
        
        Focus on healthcare-specific traceability requirements including:
        - Patient data security and privacy (HIPAA)
        - Medical device software compliance (IEC 62304)
        - FDA electronic records compliance (21 CFR Part 11)
        - Quality management systems (ISO 13485)
        - Information security (ISO 27001)
        
        Generate {req_count} requirements and {tc_count} test cases with comprehensive traceability links.
        """
        
        # Generate content using LLM
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        # Parse LLM response
        matrix = json.loads(response.candidates[0].content.parts[0].text)
        
        # Add matrix metadata
        matrix["creation_timestamp"] = datetime.now().isoformat()
        matrix["total_requirements"] = req_count
        matrix["total_test_cases"] = tc_count
        
        logging.info(f"Successfully created traceability matrix with {len(matrix.get('traceability_links', []))} links")
        return matrix
        
    except Exception as e:
        logging.error(f"Error creating traceability matrix: {e}")
        return {"error": f"Failed to create traceability matrix: {e}"}

def get_workflow_status(bucket_name: str, output_folder: str = ""):
    """
    Get the current status of workflow processing for a specific bucket and output folder.
    
    Method Signature:
        get_workflow_status(bucket_name: str, output_folder: str = "") -> dict
    
    Args:
        bucket_name (str): GCS bucket name to check
        output_folder (str, optional): Specific output folder to check (latest if empty)
    
    Returns:
        dict: Dictionary containing workflow status with keys:
            - bucket_name (str): GCS bucket name
            - output_folder (str): Output folder being checked
            - files_found (list): List of files found in the output folder
            - csv_exists (bool): Whether CSV file exists
            - pdf_exists (bool): Whether PDF file exists
            - artifacts_exist (bool): Whether artifacts JSON exists
            - total_files (int): Total number of files in output folder
            - last_modified (str): Last modification timestamp
            - status (str): Overall status (complete/partial/empty)
            - error (str): Error message if status check failed
            
    Example:
        status = get_workflow_status("hackathon-11", "output_20250921_153045")
        # Returns: {"status": "complete", "csv_exists": True, "pdf_exists": True, ...}
    """
    try:
        logging.info(f"Checking workflow status for bucket: {bucket_name}")
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # If no specific output folder, find the latest one
        if not output_folder:
            blobs = bucket.list_blobs(prefix="output_")
            output_folders = set()
            for blob in blobs:
                if "/" in blob.name:
                    folder = blob.name.split("/")[0]
                    if folder.startswith("output_"):
                        output_folders.add(folder)
            
            if not output_folders:
                return {
                    "bucket_name": bucket_name,
                    "output_folder": "",
                    "status": "empty",
                    "files_found": [],
                    "csv_exists": False,
                    "pdf_exists": False,
                    "artifacts_exist": False,
                    "total_files": 0,
                    "message": "No output folders found"
                }
            
            # Get the latest folder (highest timestamp)
            output_folder = sorted(output_folders)[-1]
        
        # List files in the output folder
        prefix = f"{output_folder}/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        files_found = []
        csv_exists = False
        pdf_exists = False
        artifacts_exist = False
        last_modified = None
        
        for blob in blobs:
            if blob.name != prefix:  # Skip the folder itself
                filename = blob.name.replace(prefix, "")
                files_found.append(filename)
                
                if filename.endswith('.csv'):
                    csv_exists = True
                elif filename.endswith('.pdf'):
                    pdf_exists = True
                elif filename.endswith('.json'):
                    artifacts_exist = True
                
                # Track latest modification time
                if last_modified is None or blob.time_created > last_modified:
                    last_modified = blob.time_created
        
        # Determine overall status
        if csv_exists and pdf_exists and artifacts_exist:
            status = "complete"
        elif csv_exists or pdf_exists or artifacts_exist:
            status = "partial"
        else:
            status = "empty"
        
        result = {
            "bucket_name": bucket_name,
            "output_folder": output_folder,
            "files_found": files_found,
            "csv_exists": csv_exists,
            "pdf_exists": pdf_exists,
            "artifacts_exist": artifacts_exist,
            "total_files": len(files_found),
            "last_modified": last_modified.isoformat() if last_modified else None,
            "status": status
        }
        
        logging.info(f"Workflow status check completed: {status} ({len(files_found)} files found)")
        return result
        
    except Exception as e:
        logging.error(f"Error checking workflow status: {e}")
        return {"error": f"Failed to check workflow status: {e}"}

def fix_malformed_adk_call(function_name: str, param1: str = "", param2: str = "", param3: str = "") -> dict:
    """
    Simple function to handle malformed ADK function calls with complex JSON parameters.
    
    This function provides a simple interface when the ADK fails to parse complex JSON in function calls.
    
    Args:
        function_name (str): Name of the function that failed ("create_traceability_report", "create_test_case_reports", etc.)
        param1 (str): First parameter (usually requirements_json or test_cases_json)
        param2 (str): Second parameter (usually test_cases_json or report_name)
        param3 (str): Third parameter (usually report_name)
    
    Returns:
        dict: Result from the corrected function call
    """
    try:
        logging.info(f"Fixing malformed ADK call for function: {function_name}")
        
        if function_name == "create_traceability_report":
            # Call the function with the provided parameters
            return create_traceability_report(
                requirements_json=param1,
                test_cases_json=param2,
                report_name=param3 if param3 else "fixed_traceability_report"
            )
        
        elif function_name == "create_test_case_reports":
            # Call the function with the provided parameters
            return create_test_case_reports(
                test_cases_json=param1,
                report_name=param2 if param2 else "fixed_test_case_reports"
            )
        
        elif function_name == "generate_test_cases_for_requirements":
            # For this function, param1=requirements_json, param2=selected_ids, param3=additional_params
            return generate_test_cases_for_requirements(
                requirements_json=param1,
                selected_requirement_ids=param2 if param2 else "REQ-001",
                test_types="functional,security,compliance",
                standards="FDA,HIPAA,IEC 62304",
                risk_level="high"
            )
        
        else:
            return {"error": f"Unknown function name: {function_name}"}
            
    except Exception as e:
        logging.error(f"Error fixing malformed ADK call: {e}")
        return {"error": f"Failed to fix malformed call for {function_name}: {e}"}


# ===== USER STORY AND MANUAL INPUT PROCESSING =====

def process_user_stories(user_stories_text: str, project_context: str = "", test_types: str = "functional,security,compliance", standards: str = "FDA,HIPAA,IEC 62304") -> dict:
    """
    Process user stories and convert them into structured requirements for test case generation.
    
    This addresses Shantanu's requirement for handling user stories as input alongside PRD documents.
    
    Args:
        user_stories_text (str): Raw text containing user stories (can be multiple stories)
        project_context (str): Additional context about the healthcare project/system
        test_types (str): Types of tests to focus on (comma-separated)
        standards (str): Compliance standards to consider (comma-separated)
    
    Returns:
        dict: Structured requirements extracted from user stories
    """
    try:
        logging.info("Processing user stories for healthcare test case generation")
        
        # Initialize GenAI client
        client = init_genai_client()
        if not client:
            raise RuntimeError("Failed to initialize GenAI client.")
        
        # Create prompt to convert user stories to structured requirements
        prompt = f"""
        You are a healthcare software business analyst expert. Convert the following user stories into structured requirements suitable for comprehensive healthcare test case generation.

        USER STORIES INPUT:
        {user_stories_text}

        PROJECT CONTEXT:
        {project_context if project_context else "Healthcare software system requiring regulatory compliance"}

        FOCUS AREAS:
        - Test Types: {test_types}
        - Compliance Standards: {standards}
        - Healthcare-specific considerations (patient safety, data privacy, regulatory compliance)

        CONVERSION REQUIREMENTS:
        1. Extract functional requirements from each user story
        2. Identify implicit security and compliance requirements
        3. Add healthcare-specific regulatory considerations
        4. Include acceptance criteria and edge cases
        5. Map to relevant compliance standards

        OUTPUT FORMAT (JSON):
        {{
            "source_type": "user_stories",
            "processing_timestamp": "{datetime.now().isoformat()}",
            "requirements": [
                {{
                    "id": "REQ-US-001",
                    "title": "Requirement title derived from user story",
                    "description": "Detailed requirement description",
                    "source_story": "Original user story text",
                    "category": "functional|security|compliance|performance",
                    "type": "functional|non-functional|compliance",
                    "priority": "critical|high|medium|low",
                    "risk_level": "high|medium|low",
                    "compliance_standards": ["FDA", "HIPAA", "IEC 62304"],
                    "acceptance_criteria": [
                        "Specific, testable acceptance criteria"
                    ],
                    "healthcare_considerations": [
                        "Patient safety implications",
                        "Data privacy requirements",
                        "Regulatory compliance needs"
                    ],
                    "section": "User Story Section",
                    "page_number": 1
                }}
            ],
            "summary": {{
                "total_requirements": 0,
                "by_category": {{}},
                "by_priority": {{}},
                "compliance_coverage": []
            }}
        }}

        HEALTHCARE-SPECIFIC FOCUS:
        - Patient data protection and privacy
        - Clinical decision support validation
        - Audit trail and logging requirements
        - User authentication and authorization
        - Data integrity and backup/recovery
        - Medical device software safety (if applicable)

        Convert ALL user stories provided into structured requirements. Each user story should generate 1-3 requirements depending on complexity.
        """

        # Generate structured requirements
        response = client.generate_content(prompt)
        
        if not response or not response.text:
            raise RuntimeError("Empty response from GenAI service")
        
        # Clean and parse the response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            # Try to fix common JSON issues
            response_text = _clean_requirements_json(response_text)
            result = json.loads(response_text)
        
        # Validate the result structure
        if not isinstance(result, dict) or 'requirements' not in result:
            raise ValueError("Invalid response structure from LLM")
        
        # Store the processed requirements in GCS
        output_folder = f"user_stories_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save requirements JSON
        requirements_filename = f"{output_folder}/user_stories_requirements.json"
        store_result = store_in_gcs(json.dumps(result, indent=2), requirements_filename, "application/json")
        
        logging.info(f"Successfully processed {len(result['requirements'])} requirements from user stories")
        
        return {
            "success": True,
            "message": f"Successfully processed user stories into {len(result['requirements'])} structured requirements",
            "requirements": result['requirements'],
            "summary": result.get('summary', {}),
            "gcs_location": store_result.get('gcs_path', ''),
            "processing_timestamp": result.get('processing_timestamp', datetime.now().isoformat()),
            "source_type": "user_stories"
        }
        
    except Exception as e:
        logging.error(f"Error processing user stories: {e}")
        return {"error": f"Failed to process user stories: {e}"}


def process_manual_input(manual_text: str, input_type: str = "requirements", project_context: str = "", test_types: str = "functional,security,compliance", standards: str = "FDA,HIPAA,IEC 62304") -> dict:
    """
    Process manual text input and convert it into structured requirements for test case generation.
    
    This addresses Shantanu's requirement for handling manual inputs alongside PRD documents and user stories.
    
    Args:
        manual_text (str): Manual text input (requirements, specifications, notes, etc.)
        input_type (str): Type of input ("requirements", "specifications", "notes", "features")
        project_context (str): Additional context about the healthcare project/system
        test_types (str): Types of tests to focus on (comma-separated)
        standards (str): Compliance standards to consider (comma-separated)
    
    Returns:
        dict: Structured requirements extracted from manual input
    """
    try:
        logging.info(f"Processing manual input of type: {input_type}")
        
        # Initialize GenAI client
        client = init_genai_client()
        if not client:
            raise RuntimeError("Failed to initialize GenAI client.")
        
        # Create prompt based on input type
        prompt = f"""
        You are a healthcare software requirements analyst expert. Convert the following manual input into structured requirements suitable for comprehensive healthcare test case generation.

        MANUAL INPUT ({input_type.upper()}):
        {manual_text}

        PROJECT CONTEXT:
        {project_context if project_context else "Healthcare software system requiring regulatory compliance"}

        PROCESSING PARAMETERS:
        - Input Type: {input_type}
        - Test Types: {test_types}
        - Compliance Standards: {standards}
        - Healthcare Domain Focus: Patient safety, data privacy, regulatory compliance

        CONVERSION REQUIREMENTS:
        1. Extract clear, testable requirements from the manual input
        2. Identify implicit security and compliance needs
        3. Add healthcare-specific regulatory considerations
        4. Include detailed acceptance criteria
        5. Map to relevant compliance standards
        6. Consider patient safety implications

        OUTPUT FORMAT (JSON):
        {{
            "source_type": "manual_input",
            "input_type": "{input_type}",
            "processing_timestamp": "{datetime.now().isoformat()}",
            "requirements": [
                {{
                    "id": "REQ-MI-001",
                    "title": "Requirement title derived from manual input",
                    "description": "Detailed requirement description with healthcare context",
                    "source_text": "Original manual input text section",
                    "category": "functional|security|compliance|performance|usability",
                    "type": "functional|non-functional|compliance",
                    "priority": "critical|high|medium|low",
                    "risk_level": "high|medium|low",
                    "compliance_standards": ["FDA", "HIPAA", "IEC 62304", "ISO 13485", "ISO 27001", "GDPR"],
                    "acceptance_criteria": [
                        "Specific, testable acceptance criteria"
                    ],
                    "healthcare_considerations": [
                        "Patient safety implications",
                        "Data privacy and security requirements",
                        "Regulatory compliance validation needs",
                        "Clinical workflow impact"
                    ],
                    "section": "Manual Input Section",
                    "page_number": 1
                }}
            ],
            "summary": {{
                "total_requirements": 0,
                "by_category": {{}},
                "by_priority": {{}},
                "by_risk_level": {{}},
                "compliance_coverage": []
            }}
        }}

        HEALTHCARE-SPECIFIC CONSIDERATIONS:
        - Patient data protection (HIPAA, GDPR)
        - Clinical decision support validation
        - Electronic records and signatures (FDA 21 CFR Part 11)
        - Medical device software safety (IEC 62304)
        - Audit trail and logging requirements
        - User authentication and access control
        - Data integrity and backup/recovery
        - System performance and reliability
        - Interoperability (HL7 FHIR)

        QUALITY REQUIREMENTS:
        - Each requirement must be specific and testable
        - Include both positive and negative scenarios
        - Consider edge cases and error conditions
        - Ensure traceability to source input
        - Add compliance validation points

        Process the entire manual input and extract ALL relevant requirements. Be thorough and comprehensive.
        """

        # Generate structured requirements
        response = client.generate_content(prompt)
        
        if not response or not response.text:
            raise RuntimeError("Empty response from GenAI service")
        
        # Clean and parse the response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            # Try to fix common JSON issues
            response_text = _clean_requirements_json(response_text)
            result = json.loads(response_text)
        
        # Validate the result structure
        if not isinstance(result, dict) or 'requirements' not in result:
            raise ValueError("Invalid response structure from LLM")
        
        # Store the processed requirements in GCS
        output_folder = f"manual_input_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save requirements JSON
        requirements_filename = f"{output_folder}/manual_input_requirements.json"
        store_result = store_in_gcs(json.dumps(result, indent=2), requirements_filename, "application/json")
        
        logging.info(f"Successfully processed manual input into {len(result['requirements'])} requirements")
        
        return {
            "success": True,
            "message": f"Successfully processed manual {input_type} input into {len(result['requirements'])} structured requirements",
            "requirements": result['requirements'],
            "summary": result.get('summary', {}),
            "gcs_location": store_result.get('gcs_path', ''),
            "processing_timestamp": result.get('processing_timestamp', datetime.now().isoformat()),
            "source_type": "manual_input",
            "input_type": input_type
        }
        
    except Exception as e:
        logging.error(f"Error processing manual input: {e}")
        return {"error": f"Failed to process manual input: {e}"}


# ===== HUMAN-IN-THE-LOOP FEEDBACK MECHANISM =====

def collect_user_feedback(test_cases_json: str, feedback_type: str = "accuracy", context: str = "") -> dict:
    """
    Collect human feedback on generated test cases to improve accuracy and completeness.
    
    This implements Shantanu's requirement for human-in-the-loop architecture to strengthen model accuracy.
    This tool PAUSES execution and waits for interactive human input, following ADK HIL pattern.
    
    Args:
        test_cases_json (str): JSON string of generated test cases to review
        feedback_type (str): Type of feedback ("epic_selection", "test_plan_approval", "quality_review", "general")
        context (str): Additional context about what feedback is needed
    
    Returns:
        dict: Processed feedback and human decision
    """
    try:
        logging.info(f"🔔 HUMAN-IN-THE-LOOP: Requesting {feedback_type} feedback")
        
        # Parse test cases
        try:
            test_cases_data = json.loads(test_cases_json)
        except json.JSONDecodeError:
            test_cases_data = json.loads(_clean_requirements_json(test_cases_json))
        
        # Display information to user based on feedback type
        print("\n" + "="*80)
        print("🔔 HUMAN-IN-THE-LOOP: Your Input Required")
        print("="*80)
        
        if feedback_type == "epic_selection":
            print("\n📋 EPIC SELECTION")
            print("-" * 80)
            epics = test_cases_data.get('epics', [])
            for i, epic in enumerate(epics, 1):
                print(f"\n{i}. {epic.get('key', 'N/A')}: {epic.get('summary', 'N/A')}")
                print(f"   Risk Level: {epic.get('risk_level', 'N/A')}")
                print(f"   Stories: {epic.get('story_count', 0)}")
                print(f"   Recommendation: {epic.get('recommendation', 'N/A')}")
            
            print("\n" + "-" * 80)
            selected = input("\n👉 Enter epic numbers to test (comma-separated, e.g., 1,3,5): ").strip()
            comments = input("👉 Any additional comments or requirements? (optional): ").strip()
            
            return {
                "success": True,
                "feedback_type": "epic_selection",
                "selected_epics": selected,
                "user_comments": comments,
                "timestamp": datetime.now().isoformat()
            }
        
        elif feedback_type == "test_plan_approval":
            print("\n📋 TEST PLAN APPROVAL")
            print("-" * 80)
            test_plan = test_cases_data.get('test_plan', {})
            print(f"\nScope: {test_plan.get('scope', 'N/A')}")
            print(f"Strategy: {test_plan.get('strategy', 'N/A')}")
            print(f"Test Types: {', '.join(test_plan.get('test_types', []))}")
            print(f"Compliance: {', '.join(test_plan.get('compliance_standards', []))}")
            print(f"\nEstimated Test Cases: {test_plan.get('estimated_test_cases', 'N/A')}")
            
            print("\n" + "-" * 80)
            approval = input("\n👉 Approve this test plan? (yes/no): ").strip().lower()
            if approval != 'yes':
                feedback = input("👉 What changes are needed?: ").strip()
            else:
                feedback = input("👉 Any additional requirements? (optional): ").strip()
            
            return {
                "success": True,
                "feedback_type": "test_plan_approval",
                "approved": approval == 'yes',
                "user_feedback": feedback,
                "timestamp": datetime.now().isoformat()
            }
        
        elif feedback_type == "quality_review":
            print("\n" + "="*80)
            print("✅ QUALITY REVIEW - TEST CASE VALIDATION RESULTS")
            print("="*80)
            
            # Parse context if provided (from quality_review_with_self_correction)
            validation_context = {}
            if context:
                try:
                    validation_context = json.loads(context)
                except:
                    pass
            
            # Display validation results
            quality_score = validation_context.get("quality_score", test_cases_data.get('quality_score', 'N/A'))
            passed_validation = validation_context.get("passed_validation", test_cases_data.get('passed_validation', False))
            issues_count = validation_context.get("issues_count", len(test_cases_data.get('issues_found', [])))
            
            print(f"\n📊 OVERALL QUALITY SCORE: {quality_score}%")
            print(f"✅ PASSED VALIDATION: {'Yes' if passed_validation else 'No'}")
            print(f"⚠️  ISSUES FOUND: {issues_count}")
            
            # Show validation details
            validation_details = validation_context.get("validation_details", {})
            if validation_details:
                print(f"\n📋 DETAILED SCORES:")
                print(f"   • Completeness: {validation_details.get('completeness', 'N/A')}%")
                print(f"   • Clarity: {validation_details.get('clarity', 'N/A')}%")
                print(f"   • Compliance: {validation_details.get('compliance', 'N/A')}%")
                print(f"   • Traceability: {validation_details.get('traceability', 'N/A')}%")
            
            # Show issues summary
            issues_summary = validation_context.get("issues_summary", [])
            if issues_summary:
                print(f"\n⚠️  TOP ISSUES:")
                for i, issue in enumerate(issues_summary[:5], 1):
                    print(f"   {i}. {issue}")
            
            # Show recommendations
            recommendations = validation_context.get("recommendations", [])
            if recommendations:
                print(f"\n💡 RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations[:5], 1):
                    print(f"   {i}. {rec}")
            
            print("\n" + "="*80)
            print(f"📄 TEST CASES SUMMARY: {len(test_cases_data.get('test_cases', []))} test cases generated")
            print("="*80)
            
            # Collect user decision
            print("\n" + "-" * 80)
            approval = input("\n👉 Do you approve these test cases? (yes/no/improve): ").strip().lower()
            
            if approval in ['yes', 'y', 'approved', 'approve']:
                improvements = input("👉 Any suggestions for future improvements? (optional): ").strip()
                approval_status = "approved"
            elif approval in ['improve', 'fix', 'no', 'n', 'rejected']:
                improvements = input("👉 What specific improvements are needed?: ").strip()
                approval_status = "rejected"
            else:
                improvements = input("👉 Please provide feedback: ").strip()
                approval_status = "pending"
            
            return {
                "success": True,
                "feedback_type": "quality_review",
                "approval_status": approval_status,
                "approved": approval_status == "approved",
                "user_comments": improvements,
                "suggested_improvements": improvements,
                "quality_score": quality_score,
                "timestamp": datetime.now().isoformat()
            }
        
        else:  # general feedback
            print("\n💬 GENERAL FEEDBACK")
            print("-" * 80)
            print(f"\nContext: {context}")
            print(f"\nTest Cases: {len(test_cases_data.get('test_cases', []))} generated")
            
            print("\n" + "-" * 80)
            comments = input("\n👉 Your feedback: ").strip()
            improvements = input("👉 Suggested improvements (optional): ").strip()
            
            # Store feedback with timestamp
            feedback_record = {
                "feedback_id": f"FB-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "feedback_type": feedback_type,
                "user_comments": comments,
                "suggested_improvements": improvements,
                "test_cases_reviewed": len(test_cases_data.get('test_cases', [])),
                "original_test_cases": test_cases_data
            }
            
            # Store feedback in GCS for future model improvement
            feedback_folder = f"user_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            feedback_filename = f"{feedback_folder}/feedback_record.json"
            store_result = store_in_gcs(json.dumps(feedback_record, indent=2), feedback_filename, "application/json")
            
            logging.info(f"Successfully collected user feedback: {feedback_record['feedback_id']}")
            
            return {
                "success": True,
                "message": f"User feedback collected successfully",
                "feedback_id": feedback_record['feedback_id'],
                "feedback_type": feedback_type,
                "user_comments": comments,
                "suggested_improvements": improvements,
                "test_cases_reviewed": feedback_record['test_cases_reviewed'],
                "gcs_location": store_result.get('gcs_path', ''),
                "next_steps": [
                    "Feedback has been stored for model improvement",
                    "Use improve_test_cases_with_feedback() to apply improvements",
                    "Consider running validation on updated test cases"
                ]
            }
        
    except Exception as e:
        logging.error(f"Error collecting user feedback: {e}")
        return {"error": f"Failed to collect user feedback: {e}"}


def improve_test_cases_with_feedback(test_cases_json: str, feedback_json: str = "", improvement_focus: str = "accuracy,completeness,compliance") -> dict:
    """
    Improve test cases based on collected user feedback using AI-assisted refinement.
    
    This implements the closed-loop process Shantanu described for continuous improvement.
    
    Args:
        test_cases_json (str): Original test cases JSON
        feedback_json (str): User feedback JSON (optional - can use latest if not provided)
        improvement_focus (str): Areas to focus on (comma-separated)
    
    Returns:
        dict: Improved test cases with change tracking
    """
    try:
        logger.info("="*70)
        logger.info("IMPROVING TEST CASES WITH FEEDBACK")
        logger.info(f"   Improvement Focus: {improvement_focus}")
        logger.info("="*70)
        
        # Initialize GenAI client
        logger.info("Initializing GenAI client...")
        client = init_genai_client()
        if not client:
            raise RuntimeError("Failed to initialize GenAI client.")
        logger.info("✓ GenAI client initialized")
        
        # Get the model
        logger.info("Getting Gemini 2.5 Pro model...")
        model = client.models.get('gemini-2.5-pro')
        logger.info("✓ Model retrieved")
        
        # Parse test cases
        logger.info("Parsing test cases JSON...")
        try:
            test_cases_data = json.loads(test_cases_json)
            logger.info(f"✓ Parsed test cases: {len(test_cases_data.get('test_cases', []))} test cases")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error, attempting to clean: {e}")
            test_cases_data = json.loads(_clean_requirements_json(test_cases_json))
            logger.info(f"✓ Cleaned and parsed: {len(test_cases_data.get('test_cases', []))} test cases")
        
        # Parse feedback if provided
        feedback_data = {}
        if feedback_json:
            logger.info("Parsing feedback JSON...")
            try:
                feedback_data = json.loads(feedback_json)
                logger.info(f"✓ Parsed feedback: {len(feedback_data)} items")
            except json.JSONDecodeError as e:
                logger.warning(f"Feedback JSON decode error, attempting to clean: {e}")
                feedback_data = json.loads(_clean_requirements_json(feedback_json))
                logger.info(f"✓ Cleaned and parsed feedback")
        else:
            logger.info("No specific feedback provided - using general improvements")
        
        focus_areas = [area.strip() for area in improvement_focus.split(',')]
        logger.info(f"Focus areas: {', '.join(focus_areas)}")
        
        # Create improvement prompt
        prompt = f"""
        You are a senior healthcare software testing expert. Improve the following test cases based on user feedback and focus areas.

        ORIGINAL TEST CASES:
        {json.dumps(test_cases_data, indent=2)}

        USER FEEDBACK:
        {json.dumps(feedback_data, indent=2) if feedback_data else "No specific feedback provided - focus on general improvements"}

        IMPROVEMENT FOCUS AREAS:
        {', '.join(focus_areas)}

        IMPROVEMENT REQUIREMENTS:
        1. Enhance test case accuracy and completeness
        2. Improve compliance validation steps
        3. Add missing edge cases and error scenarios
        4. Clarify test steps and expected results
        5. Strengthen healthcare-specific considerations
        6. Ensure full regulatory compliance coverage

        HEALTHCARE-SPECIFIC IMPROVEMENTS:
        - Patient safety validation steps
        - Data privacy and security checks
        - Audit trail verification
        - Clinical workflow integration
        - Regulatory compliance evidence
        - Error handling and recovery

        OUTPUT FORMAT (JSON):
        {{
            "improved_test_cases": [
                // Enhanced test cases with same structure but improved content
            ],
            "improvements_made": [
                {{
                    "test_case_id": "TC-REQ-001-01",
                    "improvement_type": "accuracy|completeness|compliance|clarity",
                    "description": "Specific improvement made",
                    "original_issue": "What was improved",
                    "new_content": "What was added/changed"
                }}
            ],
            "summary": {{
                "total_test_cases": 0,
                "improvements_count": 0,
                "focus_areas_addressed": [],
                "compliance_enhancements": []
            }}
        }}

        CRITICAL: Maintain all original test case IDs and structure. Only improve content quality, completeness, and accuracy.
        """

        # Generate improved test cases using the model
        logger.info("Calling LLM to improve test cases...")
        logger.info(f"   Prompt length: {len(prompt)} characters")
        
        response = model.generate_content(prompt)
        
        logger.info("✓ Received response from LLM")
        
        if not response or not response.text:
            logger.error("Empty response from GenAI service")
            raise RuntimeError("Empty response from GenAI service")
        
        logger.info(f"   Response length: {len(response.text)} characters")
        
        # Clean and parse the response
        logger.info("Parsing LLM response...")
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
            logger.info("   Removed ```json prefix")
        if response_text.endswith('```'):
            response_text = response_text[:-3]
            logger.info("   Removed ``` suffix")
        
        response_text = response_text.strip()
        
        try:
            result = json.loads(response_text)
            logger.info("✓ Successfully parsed JSON response")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error, attempting to clean: {e}")
            response_text = _clean_requirements_json(response_text)
            result = json.loads(response_text)
            logger.info("✓ Cleaned and parsed JSON response")
        
        # Extract improved test cases
        improved_test_cases = result.get('improved_test_cases', [])
        improvements_made = result.get('improvements_made', [])
        
        logger.info("="*70)
        logger.info("IMPROVEMENT RESULTS")
        logger.info(f"   Improved Test Cases: {len(improved_test_cases)}")
        logger.info(f"   Improvements Made: {len(improvements_made)}")
        logger.info("="*70)
        
        # Store improved test cases
        output_folder = f"improved_test_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save improved test cases
        improved_filename = f"{output_folder}/improved_test_cases.json"
        logger.info(f"Storing improved test cases in GCS: {improved_filename}")
        store_result = store_in_gcs(json.dumps(result, indent=2), improved_filename, "application/json")
        logger.info(f"✓ Stored in GCS: {store_result.get('gcs_path', 'N/A')}")
        
        logger.info(f"✅ Successfully improved {len(improved_test_cases)} test cases")
        
        return {
            "success": True,
            "message": f"Successfully improved test cases based on feedback",
            "improved_test_cases": improved_test_cases,
            "improvements_made": improvements_made,
            "summary": result.get('summary', {}),
            "gcs_location": store_result.get('gcs_path', ''),
            "improvement_timestamp": datetime.now().isoformat(),
            "focus_areas": focus_areas
        }
        
    except Exception as e:
        logger.error("="*70)
        logger.error("ERROR IN IMPROVE_TEST_CASES_WITH_FEEDBACK")
        logger.error(f"   Error: {e}")
        logger.exception("Full traceback:")
        logger.error("="*70)
        return {"error": f"Failed to improve test cases: {e}"}


def fix_malformed_function_output(malformed_output: str, expected_output_type: str = "test_cases") -> dict:
    """
    Simple tool to detect and correct malformed function outputs.
    
    This tool fixes common issues in LLM outputs:
    - Invalid JSON format (removes ```json markers, fixes quotes)
    - Missing required fields (adds defaults)
    - Basic structure validation
    
    Args:
        malformed_output (str): The malformed output from a previous function call
        expected_output_type (str): Type of output expected ("test_cases", "requirements", "json")
    
    Returns:
        dict: Corrected output or error information
    """
    try:
        logging.info(f"Attempting to fix malformed {expected_output_type} output")
        
        # Step 1: Clean up common JSON formatting issues
        cleaned_output = malformed_output
        if isinstance(malformed_output, str):
            cleaned_output = malformed_output.strip()
            
            # Remove markdown code blocks
            if cleaned_output.startswith('```json'):
                cleaned_output = cleaned_output[7:]
            if cleaned_output.startswith('```'):
                cleaned_output = cleaned_output[3:]
            if cleaned_output.endswith('```'):
                cleaned_output = cleaned_output[:-3]
            
            cleaned_output = cleaned_output.strip()
            
            # Fix common quote issues
            cleaned_output = cleaned_output.replace('"type": "security"', '"type": "security"')
            cleaned_output = cleaned_output.replace('"type": "', '"type": "')
            
        # Step 2: Try to parse JSON
        try:
            parsed_data = json.loads(cleaned_output)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing failed: {e}")
            
            # Try to fix common JSON issues
            try:
                # Fix missing quotes around field names
                import re
                fixed_json = re.sub(r'(\w+):', r'"\1":', cleaned_output)
                parsed_data = json.loads(fixed_json)
                logging.info("Fixed JSON by adding quotes around field names")
            except:
                return {"error": f"Could not parse JSON even after correction attempts: {e}"}
        
        # Step 3: Add missing required fields based on expected type
        if expected_output_type == "test_cases":
            if isinstance(parsed_data, dict):
                # Ensure test_cases array exists
                if "test_cases" not in parsed_data:
                    parsed_data["test_cases"] = []
                
                # Ensure summary exists
                if "summary" not in parsed_data:
                    parsed_data["summary"] = {
                        "total_test_cases": len(parsed_data.get("test_cases", [])),
                        "requirements_processed": 0,
                        "selected_requirement_ids": []
                    }
                
                # Fix each test case
                for i, tc in enumerate(parsed_data.get("test_cases", [])):
                    if not isinstance(tc, dict):
                        continue
                    
                    # Add missing required fields
                    if "test_case_id" not in tc:
                        tc["test_case_id"] = f"TC-REQ-001-{i+1:02d}"
                    
                    if "title" not in tc:
                        tc["title"] = f"Test Case {i+1}"
                    
                    if "description" not in tc:
                        tc["description"] = "Test case description"
                    
                    if "metadata" not in tc:
                        tc["metadata"] = {
                            "requirement_id": "REQ-001",
                            "test_type": "functional",
                            "priority": "medium",
                            "compliance_standards": ["HIPAA"],
                            "risk_level": "medium"
                        }
                    
                    if "test_steps" not in tc or not tc["test_steps"]:
                        tc["test_steps"] = [{
                            "step_number": 1,
                            "action": "Perform test action",
                            "input_data": "Test input",
                            "expected_result": "Expected result"
                        }]
                    
                    if "expected_results" not in tc:
                        tc["expected_results"] = {
                            "primary_result": "Test should pass",
                            "verification_criteria": ["Verify outcome"]
                        }
                    
                    if "pass_criteria" not in tc:
                        tc["pass_criteria"] = ["Test passes successfully"]
                    
                    if "fail_criteria" not in tc:
                        tc["fail_criteria"] = ["Test fails or errors"]
                
                logging.info(f"Successfully corrected {len(parsed_data['test_cases'])} test cases")
                return parsed_data
            
        elif expected_output_type == "requirements":
            if isinstance(parsed_data, dict):
                if "requirements" not in parsed_data:
                    parsed_data["requirements"] = []
                
                logging.info(f"Successfully corrected requirements data")
                return parsed_data
        
        # For generic JSON or other types
        logging.info(f"Successfully corrected malformed JSON output")
        return parsed_data
        
    except Exception as e:
        logging.error(f"Error fixing malformed output: {e}")
        return {"error": f"Failed to fix malformed output: {e}"}

def validate_processing_step(step_name: str, input_data: str, expected_keys: str = ""):
    """
    Validate the results of a processing step to ensure data quality before proceeding.
    
    Method Signature:
        validate_processing_step(step_name: str, input_data: str, expected_keys: str = "") -> dict
    
    Args:
        step_name (str): Name of the processing step being validated
        input_data (str): JSON string of data to validate
        expected_keys (str, optional): Comma-separated list of expected keys
    
    Returns:
        dict: Dictionary containing validation results with keys:
            - step_name (str): Name of the step validated
            - is_valid (bool): Whether the data passed validation
            - data_type (str): Type of data found
            - record_count (int): Number of records found
            - missing_keys (list): List of expected keys that are missing
            - validation_errors (list): List of validation errors found
            - data_quality_score (float): Quality score (0-100)
            - recommendations (list): Recommendations for improvement
            - validation_timestamp (str): ISO timestamp
            - error (str): Error message if validation failed
            
    Example:
        result = validate_processing_step("requirements_extraction", requirements_json, "id,title,description")
        # Returns: {"is_valid": True, "record_count": 25, "data_quality_score": 95.0, ...}
    """
    try:
        logging.info(f"Validating processing step: {step_name}")
        
        # Parse input data
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                return {
                    "step_name": step_name,
                    "is_valid": False,
                    "error": "Invalid JSON format in input data"
                }
        else:
            data = input_data
        
        # Parse expected keys
        if expected_keys:
            expected_key_list = [key.strip() for key in expected_keys.split(',') if key.strip()]
        else:
            expected_key_list = []
        
        validation_errors = []
        missing_keys = []
        record_count = 0
        data_type = type(data).__name__
        
        # Validate data structure
        if isinstance(data, list):
            record_count = len(data)
            if record_count == 0:
                validation_errors.append("Empty data list")
            else:
                # Check first few records for expected keys
                sample_size = min(3, record_count)
                for i in range(sample_size):
                    record = data[i]
                    if isinstance(record, dict):
                        for expected_key in expected_key_list:
                            if expected_key not in record:
                                if expected_key not in missing_keys:
                                    missing_keys.append(expected_key)
                    else:
                        validation_errors.append(f"Record {i} is not a dictionary")
        
        elif isinstance(data, dict):
            record_count = 1
            # Check for expected keys in the dictionary
            for expected_key in expected_key_list:
                if expected_key not in data:
                    missing_keys.append(expected_key)
        else:
            validation_errors.append(f"Unexpected data type: {data_type}")
        
        # Calculate data quality score
        quality_score = 100.0
        if validation_errors:
            quality_score -= len(validation_errors) * 20
        if missing_keys:
            quality_score -= len(missing_keys) * 10
        quality_score = max(0, quality_score)
        
        # Generate recommendations
        recommendations = []
        if missing_keys:
            recommendations.append(f"Add missing required keys: {', '.join(missing_keys)}")
        if validation_errors:
            recommendations.append("Fix validation errors before proceeding")
        if record_count == 0:
            recommendations.append("Ensure data processing step generates valid output")
        
        is_valid = len(validation_errors) == 0 and record_count > 0
        
        result = {
            "step_name": step_name,
            "is_valid": is_valid,
            "data_type": data_type,
            "record_count": record_count,
            "missing_keys": missing_keys,
            "validation_errors": validation_errors,
            "data_quality_score": quality_score,
            "recommendations": recommendations,
            "validation_timestamp": datetime.now().isoformat()
        }
        
        logging.info(f"Step validation completed: {step_name} - Valid: {is_valid}, Score: {quality_score:.1f}%")
        return result
        
    except Exception as e:
        logging.error(f"Error validating processing step {step_name}: {e}")
        return {"error": f"Failed to validate processing step: {e}"}

def generate_test_cases_preview(test_cases_json: str, max_preview: int = 10) -> dict:
    """
    Generate a formatted preview of generated test cases for user review.
    
    This function creates a readable summary of test cases with key details,
    displayed before final export so users can review what was generated.
    
    Args:
        test_cases_json (str): JSON string or file path containing test cases
        max_preview (int): Maximum number of test cases to show in preview (default: 10)
    
    Returns:
        dict: Test cases preview with summary and formatted display data
    """
    try:
        logger.info("="*70)
        logger.info("GENERATING TEST CASES PREVIEW")
        logger.info("="*70)
        
        # Read test cases (handle both JSON string and file path)
        if test_cases_json and os.path.exists(test_cases_json):
            logger.info(f"Reading test cases from file: {test_cases_json}")
            with open(test_cases_json, 'r', encoding='utf-8') as f:
                test_cases_data = json.load(f)
        else:
            logger.info("Parsing test cases from JSON string")
            test_cases_data = json.loads(test_cases_json)
        
        # Extract test cases
        if isinstance(test_cases_data, dict) and 'test_cases' in test_cases_data:
            test_cases = test_cases_data['test_cases']
        elif isinstance(test_cases_data, list):
            test_cases = test_cases_data
        else:
            return {"error": "Invalid test cases format"}
        
        logger.info(f"✓ Loaded {len(test_cases)} test cases")
        
        # Extract requirement IDs from metadata
        for tc in test_cases:
            if not tc.get('requirement_ids'):
                metadata = tc.get('metadata', {})
                if isinstance(metadata, dict):
                    req_id = metadata.get('requirement_id', '')
                    if req_id:
                        tc['requirement_ids'] = [req_id]
        
        # Build summary statistics
        priority_distribution = {}
        category_distribution = {}
        compliance_standards = set()
        requirement_ids = set()
        
        for tc in test_cases:
            # Priority distribution
            priority = tc.get('priority', tc.get('metadata', {}).get('priority', 'Medium'))
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
            
            # Category/Type distribution
            category = tc.get('category', tc.get('metadata', {}).get('test_type', 'Functional'))
            category_distribution[category] = category_distribution.get(category, 0) + 1
            
            # Compliance standards
            compliance = tc.get('compliance_standards', [])
            compliance_validation = tc.get('compliance_validation', {})
            if isinstance(compliance_validation, dict):
                standards = compliance_validation.get('regulatory_requirements', [])
                for std in standards:
                    if 'HIPAA' in str(std):
                        compliance_standards.add('HIPAA')
                    if 'FDA' in str(std) or '21 CFR' in str(std):
                        compliance_standards.add('FDA 21 CFR Part 11')
                    if 'IEC 62304' in str(std):
                        compliance_standards.add('IEC 62304')
                    if 'ISO 13485' in str(std):
                        compliance_standards.add('ISO 13485')
            if isinstance(compliance, list):
                compliance_standards.update(compliance)
            
            # Requirement IDs
            req_ids = tc.get('requirement_ids', [])
            if isinstance(req_ids, list):
                requirement_ids.update(req_ids)
        
        # Create formatted preview entries
        preview_entries = []
        for i, tc in enumerate(test_cases[:max_preview]):
            tc_id = tc.get('id', tc.get('test_case_id', f'TC-{i+1}'))
            tc_title = tc.get('title', 'N/A')
            
            # Get requirement ID from metadata if not at top level
            req_ids = tc.get('requirement_ids', [])
            if not req_ids:
                metadata = tc.get('metadata', {})
                if isinstance(metadata, dict):
                    req_id = metadata.get('requirement_id', '')
                    if req_id:
                        req_ids = [req_id]
            
            req_id_str = ', '.join(req_ids) if req_ids else 'N/A'
            
            # Get test steps count
            test_steps = tc.get('test_steps', [])
            steps_count = len(test_steps) if isinstance(test_steps, list) else 0
            
            # Get priority and category
            priority = tc.get('priority', tc.get('metadata', {}).get('priority', 'Medium'))
            category = tc.get('category', tc.get('metadata', {}).get('test_type', 'Functional'))
            
            preview_entries.append({
                'test_case_id': tc_id,
                'title': tc_title,
                'requirement_id': req_id_str,
                'priority': priority,
                'category': category,
                'test_steps_count': steps_count,
                'preconditions': len(tc.get('prerequisites', [])) if isinstance(tc.get('prerequisites'), list) else 0
            })
        
        result = {
            "preview_id": f"PREVIEW-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "summary": {
                "total_test_cases": len(test_cases),
                "total_previewed": min(len(test_cases), max_preview),
                "unique_requirements": len(requirement_ids),
                "total_compliance_standards": len(compliance_standards)
            },
            "priority_distribution": priority_distribution,
            "category_distribution": category_distribution,
            "compliance_standards": sorted(list(compliance_standards)),
            "requirement_ids": sorted(list(requirement_ids)),
            "preview_entries": preview_entries,
            "message": f"Generated {len(test_cases)} test cases covering {len(requirement_ids)} requirements. Showing {min(len(test_cases), max_preview)} in preview."
        }
        
        logger.info(f"✓ Preview generated: {len(test_cases)} test cases")
        return result
        
    except Exception as e:
        logger.error(f"✗ Failed to generate test cases preview: {e}")
        logger.exception("Full traceback:")
        return {"error": f"Failed to generate test cases preview: {e}"}


def generate_traceability_matrix_preview(test_cases_json: str, requirements_json: str = "") -> dict:
    """
    Generate a traceability matrix preview for user review BEFORE export.
    
    This function creates a summary of test case to requirement mappings,
    compliance coverage, and risk distribution for the user to review
    before finalizing exports.
    
    Args:
        test_cases_json (str): JSON string or file path containing test cases
        requirements_json (str, optional): JSON string containing requirements summary
    
    Returns:
        dict: Traceability matrix preview with statistics and mappings
    """
    try:
        logger.info("="*70)
        logger.info("GENERATING TRACEABILITY MATRIX PREVIEW")
        logger.info("="*70)
        
        # Read test cases (handle both JSON string and file path)
        if test_cases_json and os.path.exists(test_cases_json):
            logger.info(f"Reading test cases from file: {test_cases_json}")
            with open(test_cases_json, 'r', encoding='utf-8') as f:
                test_cases_data = json.load(f)
        else:
            logger.info("Parsing test cases from JSON string")
            test_cases_data = json.loads(test_cases_json)
        
        # Extract test cases
        if isinstance(test_cases_data, dict) and 'test_cases' in test_cases_data:
            test_cases = test_cases_data['test_cases']
        elif isinstance(test_cases_data, list):
            test_cases = test_cases_data
        else:
            return {"error": "Invalid test cases format"}
        
        logger.info(f"✓ Loaded {len(test_cases)} test cases")
        
        # Build traceability mappings
        requirement_coverage = {}
        compliance_coverage = {}
        risk_distribution = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        test_type_distribution = {}
        
        for tc in test_cases:
            tc_id = tc.get('id', tc.get('test_case_id', 'N/A'))
            tc_title = tc.get('title', 'N/A')
            tc_priority = tc.get('priority', 'Medium')
            tc_category = tc.get('category', 'Functional')
            
            # Track requirement coverage - check multiple possible locations
            req_ids = tc.get('requirement_ids', tc.get('linked_requirements', []))
            
            # If not found at top level, check metadata (CRITICAL FIX)
            if not req_ids:
                metadata = tc.get('metadata', {})
                if isinstance(metadata, dict):
                    req_id = metadata.get('requirement_id', '')
                    if req_id:
                        req_ids = [req_id]
            
            # Ensure it's a list
            if isinstance(req_ids, str):
                req_ids = [req_ids]
            
            for req_id in req_ids:
                if req_id not in requirement_coverage:
                    requirement_coverage[req_id] = []
                requirement_coverage[req_id].append({
                    'test_case_id': tc_id,
                    'test_case_title': tc_title,
                    'priority': tc_priority,
                    'category': tc_category
                })
            
            # Track compliance coverage
            compliance_validation = tc.get('compliance_validation', {})
            if isinstance(compliance_validation, dict):
                standards = compliance_validation.get('standards_validated', [])
                for standard in standards:
                    if standard not in compliance_coverage:
                        compliance_coverage[standard] = 0
                    compliance_coverage[standard] += 1
            
            # Track risk distribution
            if tc_priority in risk_distribution:
                risk_distribution[tc_priority] += 1
            
            # Track test type distribution
            if tc_category not in test_type_distribution:
                test_type_distribution[tc_category] = 0
            test_type_distribution[tc_category] += 1
        
        # Calculate coverage statistics
        total_requirements = len(requirement_coverage)
        covered_requirements = sum(1 for req, tcs in requirement_coverage.items() if len(tcs) > 0)
        coverage_percentage = (covered_requirements / total_requirements * 100) if total_requirements > 0 else 0
        
        # Find gaps (requirements with no test cases or only 1 test case)
        gaps = []
        for req_id, test_cases_list in requirement_coverage.items():
            if len(test_cases_list) == 0:
                gaps.append({
                    'requirement_id': req_id,
                    'issue': 'No test cases mapped',
                    'recommendation': 'Add test cases for this requirement'
                })
            elif len(test_cases_list) == 1:
                gaps.append({
                    'requirement_id': req_id,
                    'issue': 'Only 1 test case mapped',
                    'recommendation': 'Consider adding negative/edge case tests'
                })
        
        # Create detailed traceability table (for display)
        traceability_table = []
        for req_id, test_cases_list in sorted(requirement_coverage.items()):
            for tc in test_cases_list:
                traceability_table.append({
                    'requirement_id': req_id,
                    'test_case_id': tc['test_case_id'],
                    'test_case_title': tc['test_case_title'],
                    'priority': tc['priority'],
                    'category': tc['category']
                })
        
        result = {
            "traceability_id": f"TRACE-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "summary": {
                "total_test_cases": len(test_cases),
                "total_requirements": total_requirements,
                "covered_requirements": covered_requirements,
                "coverage_percentage": round(coverage_percentage, 1),
                "gaps_found": len(gaps)
            },
            "requirement_coverage": {
                req_id: {
                    "test_case_count": len(tcs),
                    "test_cases": [tc['test_case_id'] for tc in tcs]
                }
                for req_id, tcs in requirement_coverage.items()
            },
            "compliance_coverage": compliance_coverage,
            "risk_distribution": risk_distribution,
            "test_type_distribution": test_type_distribution,
            "gaps_and_recommendations": gaps[:10],  # Top 10 gaps
            "traceability_table": traceability_table[:50],  # First 50 entries for display
            "full_table_count": len(traceability_table),
            "message": f"Traceability matrix preview: {covered_requirements}/{total_requirements} requirements covered ({coverage_percentage:.1f}%)"
        }
        
        logger.info("="*70)
        logger.info("TRACEABILITY MATRIX SUMMARY")
        logger.info(f"   Total Test Cases: {len(test_cases)}")
        logger.info(f"   Total Requirements: {total_requirements}")
        logger.info(f"   Coverage: {coverage_percentage:.1f}%")
        logger.info(f"   Gaps Found: {len(gaps)}")
        logger.info("="*70)
        
        logger.info("Risk Distribution:")
        for risk, count in risk_distribution.items():
            logger.info(f"   {risk}: {count} test cases")
        
        logger.info("Compliance Coverage:")
        for standard, count in compliance_coverage.items():
            logger.info(f"   {standard}: {count} test cases")
        
        if gaps:
            logger.info("Top Gaps:")
            for i, gap in enumerate(gaps[:5], 1):
                logger.info(f"   {i}. {gap['requirement_id']}: {gap['issue']}")
        
        logger.info("="*70)
        
        return result
        
    except Exception as e:
        logger.error(f"✗ Failed to generate traceability matrix preview: {e}")
        logger.exception("Full traceback:")
        return {"error": f"Failed to generate traceability matrix preview: {e}"}


def export_traceability_matrix_to_csv(test_cases_json: str, requirements_json: str = "", output_filename: str = "") -> dict:
    """
    Export traceability matrix to CSV format in the local output folder.
    
    Creates a CSV file showing the mapping between requirements and test cases.
    
    Args:
        test_cases_json (str): JSON string of test cases
        requirements_json (str): JSON string of requirements (optional)
        output_filename (str): Custom filename (auto-generated if empty)
    
    Returns:
        dict: Export result with local file path
             
    Example:
        result = export_traceability_matrix_to_csv('[{"id": "TC-001", ...}]', '[{"id": "REQ-001", ...}]')
        # Returns: {"success": True, "file_path": "output/traceability_matrix_20250102_143022.csv"}
    """
    try:
        logging.info("Exporting traceability matrix to CSV in output folder")
        
        # Handle file path if provided instead of JSON string
        if isinstance(test_cases_json, str) and os.path.exists(test_cases_json):
            logging.info(f"Reading test cases from file: {test_cases_json}")
            with open(test_cases_json, 'r', encoding='utf-8') as f:
                test_cases_data = json.load(f)
        else:
            # Parse test cases
            try:
                test_cases_data = json.loads(test_cases_json) if isinstance(test_cases_json, str) else test_cases_json
            except json.JSONDecodeError:
                test_cases_data = json.loads(_clean_requirements_json(test_cases_json))
        
        # Parse requirements if provided
        requirements_data = []
        if requirements_json:
            try:
                requirements_data = json.loads(requirements_json) if isinstance(requirements_json, str) else requirements_json
            except json.JSONDecodeError:
                requirements_data = json.loads(_clean_requirements_json(requirements_json))
        
        # Ensure output folder exists
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_filename:
            output_filename = f"traceability_matrix_{timestamp}.csv"
        
        file_path = os.path.join(output_dir, output_filename)
        
        # Create traceability matrix data
        traceability_data = []
        
        # Extract test cases list
        if isinstance(test_cases_data, dict):
            test_cases_list = test_cases_data.get('test_cases', [])
        elif isinstance(test_cases_data, list):
            test_cases_list = test_cases_data
        else:
            test_cases_list = []
        
        # Extract requirements list
        if isinstance(requirements_data, dict):
            requirements_list = requirements_data.get('requirements', [])
        elif isinstance(requirements_data, list):
            requirements_list = requirements_data
        else:
            requirements_list = []
        
        # Build traceability matrix
        for test_case in test_cases_list:
            tc_id = test_case.get('id', test_case.get('test_case_id', 'N/A'))
            tc_title = test_case.get('title', test_case.get('name', 'N/A'))
            tc_priority = test_case.get('priority', 'N/A')
            
            # Get linked requirements - check multiple possible locations
            linked_reqs = test_case.get('requirement_ids', test_case.get('linked_requirements', []))
            
            # If not found at top level, check metadata
            if not linked_reqs:
                metadata = test_case.get('metadata', {})
                if isinstance(metadata, dict):
                    req_id = metadata.get('requirement_id', '')
                    if req_id:
                        linked_reqs = [req_id]
            
            # Handle string format
            if isinstance(linked_reqs, str):
                linked_reqs = [r.strip() for r in linked_reqs.split(',') if r.strip()]
            
            # Find requirement details
            for req_id in linked_reqs:
                req_details = next((r for r in requirements_list if r.get('id') == req_id), {})
                
                traceability_data.append({
                    'Requirement ID': req_id,
                    'Requirement Title': req_details.get('title', req_details.get('summary', 'N/A')),
                    'Requirement Type': req_details.get('type', 'N/A'),
                    'Test Case ID': tc_id,
                    'Test Case Title': tc_title,
                    'Test Case Priority': tc_priority,
                    'Coverage Status': 'Covered'
                })
        
        # If no traceability data, create from test cases only
        if not traceability_data:
            for test_case in test_cases_list:
                tc_id = test_case.get('id', test_case.get('test_case_id', 'N/A'))
                tc_title = test_case.get('title', test_case.get('name', 'N/A'))
                tc_priority = test_case.get('priority', 'N/A')
                linked_reqs = test_case.get('requirement_ids', test_case.get('linked_requirements', []))
                
                if isinstance(linked_reqs, str):
                    linked_reqs_str = linked_reqs
                elif isinstance(linked_reqs, list):
                    linked_reqs_str = ', '.join(linked_reqs)
                else:
                    linked_reqs_str = 'N/A'
                
                traceability_data.append({
                    'Requirement ID': linked_reqs_str,
                    'Requirement Title': 'N/A',
                    'Requirement Type': 'N/A',
                    'Test Case ID': tc_id,
                    'Test Case Title': tc_title,
                    'Test Case Priority': tc_priority,
                    'Coverage Status': 'Covered'
                })
        
        # Write to CSV using pandas
        import pandas as pd
        df = pd.DataFrame(traceability_data)
        df.to_csv(file_path, index=False, encoding='utf-8')
        
        logging.info(f"Successfully exported traceability matrix to CSV: {file_path}")
        
        # Upload to GCS
        logger.info("Uploading traceability matrix CSV to GCS...")
        gcs_filename = f"traceability_matrix/{output_filename}"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                csv_content = f.read()
            
            gcs_result = store_in_gcs(csv_content, gcs_filename, "text/csv")
            gcs_path = gcs_result.get('gcs_path', '')
            logger.info(f"✓ Uploaded to GCS: {gcs_path}")
        except Exception as e:
            logger.warning(f"Failed to upload to GCS: {e}")
            gcs_path = "GCS upload failed"
        
        return {
            "success": True,
            "message": f"Traceability matrix exported successfully",
            "file_path": file_path,
            "gcs_path": gcs_path,
            "total_mappings": len(traceability_data),
            "timestamp": timestamp
        }
        
    except Exception as e:
        logging.error(f"Error exporting traceability matrix to CSV: {e}")
        return {"error": f"Failed to export traceability matrix: {e}"}

def _create_traceability_matrix_pdf(traceability_matrix):
    """Create PDF content for traceability matrix."""
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    import io
    
    # Create PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Build content
    story = []
    
    # Title
    story.append(Paragraph("Healthcare Test Case Traceability Matrix", title_style))
    story.append(Spacer(1, 20))
    
    # Summary information
    coverage_analysis = traceability_matrix.get('coverage_analysis', {})
    story.append(Paragraph("Coverage Summary", heading_style))
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Requirements', str(coverage_analysis.get('total_requirements', 0))],
        ['Covered Requirements', str(coverage_analysis.get('covered_requirements', 0))],
        ['Uncovered Requirements', str(coverage_analysis.get('uncovered_requirements', 0))],
        ['Coverage Percentage', f"{coverage_analysis.get('coverage_percentage', 0):.1f}%"]
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Requirements and Test Cases mapping
    story.append(Paragraph("Requirements to Test Cases Mapping", heading_style))
    
    requirements = traceability_matrix.get('requirements', [])
    if requirements:
        # Create table data
        table_data = [['Requirement ID', 'Title', 'Type', 'Priority', 'Linked Test Cases', 'Coverage Status']]
        
        for req in requirements[:20]:  # Limit to first 20 for readability
            linked_tests = ', '.join(req.get('linked_test_cases', []))
            coverage_status = req.get('coverage_status', 'Unknown')
            
            table_data.append([
                req.get('id', ''),
                req.get('title', '')[:50] + '...' if len(req.get('title', '')) > 50 else req.get('title', ''),
                req.get('type', ''),
                req.get('priority', ''),
                linked_tests[:30] + '...' if len(linked_tests) > 30 else linked_tests,
                coverage_status
            ])
        
        # Create table
        req_table = Table(table_data, colWidths=[1*inch, 2*inch, 0.8*inch, 0.8*inch, 1.5*inch, 1*inch])
        req_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        
        story.append(req_table)
        
        if len(requirements) > 20:
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"... and {len(requirements) - 20} more requirements", styles['Normal']))
    
    story.append(PageBreak())
    
    # Test Cases details
    story.append(Paragraph("Test Cases Details", heading_style))
    
    test_cases = traceability_matrix.get('test_cases', [])
    if test_cases:
        tc_table_data = [['Test Case ID', 'Title', 'Type', 'Priority', 'Requirement ID', 'Compliance Standard']]
        
        for tc in test_cases[:30]:  # Limit to first 30 for readability
            tc_table_data.append([
                tc.get('id', ''),
                tc.get('title', '')[:40] + '...' if len(tc.get('title', '')) > 40 else tc.get('title', ''),
                tc.get('type', ''),
                tc.get('priority', ''),
                tc.get('requirement_id', ''),
                tc.get('compliance_standard', '')
            ])
        
        tc_table = Table(tc_table_data, colWidths=[1*inch, 2*inch, 1*inch, 0.8*inch, 1*inch, 1.2*inch])
        tc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        
        story.append(tc_table)
        
        if len(test_cases) > 30:
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"... and {len(test_cases) - 30} more test cases", styles['Normal']))
    
    # Compliance mapping
    compliance_mapping = traceability_matrix.get('compliance_mapping', {})
    if compliance_mapping:
        story.append(Spacer(1, 20))
        story.append(Paragraph("Compliance Standards Mapping", heading_style))
        
        comp_data = [['Standard', 'Requirements Count', 'Test Cases Count', 'Coverage %']]
        for standard, data in compliance_mapping.items():
            comp_data.append([
                standard,
                str(len(data.get('requirements', []))),
                str(len(data.get('test_cases', []))),
                f"{data.get('coverage_percentage', 0):.1f}%"
            ])
        
        comp_table = Table(comp_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1*inch])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(comp_table)
    
    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def store_in_gcs(artifacts_json: str, bucket_name: str, folder_path: str):
    """
    Store generated artifacts in Google Cloud Storage.
    
    Method Signature:
        store_in_gcs(artifacts_json: str, bucket_name: str, folder_path: str) -> dict
    
    Args:
        artifacts_json (str): JSON string of artifacts to store (test cases, traceability matrix, etc.)
        bucket_name (str): GCS bucket name where artifacts will be stored
        folder_path (str): Folder path in bucket for storing artifacts
    
    Returns:
        dict: Dictionary containing storage results with keys:
            - stored_files (list): List of stored file paths
            - bucket_name (str): GCS bucket name used
            - folder_path (str): Folder path used
            - storage_timestamp (str): ISO timestamp
            - total_files (int): Number of files stored
            - error (str): Error message if storage failed
            
    Example:
        result = store_in_gcs('{"test_cases": [...], "requirements": [...]}', "my-bucket", "output")
        # Returns: {"stored_files": [...], "total_files": 3, ...}
    """
    try:
        # Parse artifacts
        if isinstance(artifacts_json, str):
            artifacts = json.loads(artifacts_json)
        else:
            artifacts = artifacts_json
            
        logging.info(f"Storing artifacts in GCS bucket: {bucket_name}/{folder_path}")
        
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        stored_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Store test cases
        if "test_cases" in artifacts:
            test_cases_file = f"{folder_path}/test_cases_{timestamp}.json"
            blob = bucket.blob(test_cases_file)
            blob.upload_from_string(json.dumps(artifacts["test_cases"], indent=2))
            stored_files.append(test_cases_file)
        
        # Store traceability matrix
        if "traceability_matrix" in artifacts:
            traceability_file = f"{folder_path}/traceability_matrix_{timestamp}.json"
            blob = bucket.blob(traceability_file)
            blob.upload_from_string(json.dumps(artifacts["traceability_matrix"], indent=2))
            stored_files.append(traceability_file)
        
        # Store compliance report
        if "compliance_report" in artifacts:
            compliance_file = f"{folder_path}/compliance_report_{timestamp}.json"
            blob = bucket.blob(compliance_file)
            blob.upload_from_string(json.dumps(artifacts["compliance_report"], indent=2))
            stored_files.append(compliance_file)
        
        result = {
            "stored_files": stored_files,
            "bucket_name": bucket_name,
            "folder_path": folder_path,
            "storage_timestamp": datetime.now().isoformat(),
            "total_files": len(stored_files)
        }
        
        logging.info(f"Successfully stored {len(stored_files)} files in GCS")
        return result
        
    except Exception as e:
        logging.error(f"Error storing in GCS: {e}")
        return {"error": f"Failed to store in GCS: {e}"}

def upload_to_gcs(content: str, gcs_path: str, filename: str) -> str:
    """
    Uploads content to Google Cloud Storage.

    Args:
        content (str): The content to upload.
        gcs_path (str): The GCS path (gs://bucket-name/folder/).
        filename (str): The name of the file to create.

    Returns:
        str: The full GCS path of the uploaded file.
    """
    # Parse the GCS path
    parsed_url = urlparse(gcs_path)
    bucket_name = parsed_url.netloc
    folder_path = parsed_url.path.lstrip('/')
    
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Create the full blob path
    blob_path = f"{folder_path}{filename}" if folder_path else filename
    blob = bucket.blob(blob_path)
    
    # Upload the content
    blob.upload_from_string(content, content_type='text/plain')
    
    full_gcs_path = f"gs://{bucket_name}/{blob_path}"
    logging.info(f"Successfully uploaded {filename} to {full_gcs_path}")
    return full_gcs_path

# Test Case Generation Tools

def generate_test_cases(requirements_json: str, test_types: str, standards: str, risk_level: str):
    """
    Generate comprehensive test cases from analyzed requirements using LLM.
    
    Method Signature:
        generate_test_cases(requirements_json: str, test_types: str, standards: str, risk_level: str) -> dict
    
    Args:
        requirements_json (str): JSON string of analyzed requirements
        test_types (str): Comma-separated test types (e.g., "functional,security,compliance")
        standards (str): Comma-separated compliance standards (e.g., "FDA,HIPAA,IEC 62304")
        risk_level (str): Risk level of the software ("high", "medium", "low")
    
    Returns:
        dict: Dictionary containing generated test cases with keys:
            - test_cases (list): List of test case dictionaries
            - coverage_analysis (dict): Coverage analysis results
            - requirements_covered (int): Number of requirements covered
            - test_types (list): Applied test types
            - compliance_standards (list): Applied compliance standards
            - generation_timestamp (str): ISO timestamp
            - error (str): Error message if generation failed
            
    Example:
        result = generate_test_cases('{"requirements": [...]}', "functional,security", "FDA,HIPAA", "high")
        # Returns: {"test_cases": [...], "coverage_analysis": {...}, ...}
    """
    try:
        # Parse inputs
        if isinstance(requirements_json, str):
            requirements = json.loads(requirements_json)
        else:
            requirements = requirements_json
            
        if isinstance(test_types, str):
            test_types_list = [t.strip() for t in test_types.split(',') if t.strip()]
        else:
            test_types_list = test_types if test_types else ["functional", "security", "compliance"]
            
        if isinstance(standards, str):
            compliance_standards = [s.strip() for s in standards.split(',') if s.strip()]
        else:
            compliance_standards = standards if standards else ["FDA", "HIPAA", "IEC 62304"]
        
        logging.info(f"Generating test cases for {len(requirements)} requirements")
        
        client = init_genai_client()
        if not client:
            raise RuntimeError("Failed to initialize GenAI client.")
        
        # Create prompt for LLM to generate test cases
        prompt = f"""
        You are a healthcare software test engineer. Generate comprehensive test cases for the following requirements.
        
        Requirements: {json.dumps(requirements, indent=2)}
        Test Types: {', '.join(test_types_list)}
        Compliance Standards: {', '.join(compliance_standards)}
        Risk Level: {risk_level}
        
        Generate test cases in the following JSON format:
        {{
            "test_cases": [
                {{
                    "id": "TC-REQ-001-001",
                    "title": "Test Case Title",
                    "description": "Detailed test case description",
                    "test_type": "functional|security|compliance|performance|integration",
                    "priority": "critical|high|medium|low",
                    "requirement_id": "REQ-001",
                    "test_steps": [
                        "Step 1: Setup test environment",
                        "Step 2: Execute test action",
                        "Step 3: Verify results"
                    ],
                    "expected_result": "Expected outcome of the test",
                    "compliance_standard": "FDA|HIPAA|IEC 62304|ISO 13485",
                    "test_data": "Required test data",
                    "prerequisites": "Test prerequisites",
                    "risk_level": "high|medium|low"
                }}
            ],
            "coverage_analysis": {{
                "functional_coverage": "95%",
                "security_coverage": "90%",
                "compliance_coverage": "88%",
                "total_test_cases": 0
            }}
        }}
        
        Focus on healthcare-specific testing scenarios and ensure compliance with the specified standards.
        Generate test cases that cover positive, negative, and edge case scenarios.
        """
        
        # Generate content using LLM
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )
        
        # Parse LLM response
        result = json.loads(response.candidates[0].content.parts[0].text)
        
        # Add generation metadata
        result["requirements_covered"] = len(requirements)
        result["test_types"] = test_types_list
        result["compliance_standards"] = compliance_standards
        result["risk_level"] = risk_level
        result["generation_timestamp"] = datetime.now().isoformat()
        
        logging.info(f"Successfully generated {len(result.get('test_cases', []))} test cases")
        return result
        
    except Exception as e:
        logging.error(f"Error generating test cases: {e}")
        return {"error": f"Failed to generate test cases: {e}"}

def _generate_functional_tests(requirement, compliance_standards):
    """Generate functional test cases for a requirement."""
    test_cases = []
    
    # Positive test case
    test_cases.append({
        "id": f"TC-{requirement['id']}-001",
        "title": f"Verify {requirement['title']} - Positive Scenario",
        "description": f"Test that {requirement['title']} works correctly under normal conditions",
        "test_type": "functional",
        "priority": requirement.get("priority", "medium"),
        "requirement_id": requirement["id"],
        "test_steps": [
            "Set up test environment with valid data",
            f"Execute {requirement['title']} functionality",
            "Verify expected behavior occurs",
            "Validate all acceptance criteria are met"
        ],
        "expected_result": "Functionality works as specified in requirements",
        "compliance_standard": compliance_standards[0] if compliance_standards else None
    })
    
    # Negative test case
    test_cases.append({
        "id": f"TC-{requirement['id']}-002",
        "title": f"Verify {requirement['title']} - Negative Scenario",
        "description": f"Test that {requirement['title']} handles invalid inputs gracefully",
        "test_type": "functional",
        "priority": requirement.get("priority", "medium"),
        "requirement_id": requirement["id"],
        "test_steps": [
            "Set up test environment with invalid data",
            f"Execute {requirement['title']} functionality with invalid inputs",
            "Verify appropriate error handling occurs",
            "Validate system remains stable"
        ],
        "expected_result": "System handles invalid inputs gracefully without crashing",
        "compliance_standard": compliance_standards[0] if compliance_standards else None
    })
    
    return test_cases

def _generate_security_tests(requirement, compliance_standards):
    """Generate security test cases for a requirement."""
    if requirement.get("category") != "security":
        return []
    
    return [{
        "id": f"TC-{requirement['id']}-SEC-001",
        "title": f"Security Test - {requirement['title']}",
        "description": f"Test security aspects of {requirement['title']}",
        "test_type": "security",
        "priority": "high",
        "requirement_id": requirement["id"],
        "test_steps": [
            "Attempt unauthorized access to the functionality",
            "Verify access is properly denied",
            "Check audit logs for security events",
            "Validate encryption and data protection measures"
        ],
        "expected_result": "Security measures are properly implemented and enforced",
        "compliance_standard": "HIPAA" if "HIPAA" in compliance_standards else compliance_standards[0] if compliance_standards else None
    }]

def _generate_compliance_tests(requirement, compliance_standards):
    """Generate compliance test cases for a requirement."""
    if requirement.get("category") != "compliance":
        return []
    
    test_cases = []
    for standard in compliance_standards:
        test_cases.append({
            "id": f"TC-{requirement['id']}-COMP-{standard}-001",
            "title": f"Compliance Test - {requirement['title']} - {standard}",
            "description": f"Test {standard} compliance for {requirement['title']}",
            "test_type": "compliance",
            "priority": "high",
            "requirement_id": requirement["id"],
            "test_steps": [
                f"Verify {standard} requirements are met",
                "Check documentation and audit trails",
                "Validate regulatory compliance measures",
                "Confirm adherence to {standard} standards"
            ],
            "expected_result": f"Requirement meets {standard} compliance standards",
            "compliance_standard": standard
        })
    
    return test_cases

# Compliance Validation Tools

def validate_compliance_requirements(requirements_json: str, standards: str, software_class: str):
    """
    Validate that requirements meet healthcare compliance standards using LLM.
    
    Method Signature:
        validate_compliance_requirements(requirements_json: str, standards: str, software_class: str) -> dict
    
    Args:
        requirements_json (str): JSON string of requirements to validate
        standards (str): Comma-separated compliance standards (e.g., "FDA,HIPAA,IEC 62304")
        software_class (str): Medical device software class ("A", "B", or "C")
    
    Returns:
        dict: Dictionary containing validation results with keys:
            - validated_requirements (list): List of validated requirement dictionaries
            - compliance_gaps (list): List of overall compliance gaps
            - recommendations (list): List of recommendations
            - overall_compliance_score (float): Overall compliance score (0-100)
            - standards_summary (dict): Summary by compliance standard
            - validation_timestamp (str): ISO timestamp
            - software_class (str): Applied software class
            - total_requirements (int): Total number of requirements validated
            - error (str): Error message if validation failed
            
    Example:
        result = validate_compliance_requirements('{"requirements": [...]}', "FDA,HIPAA", "B")
        # Returns: {"validated_requirements": [...], "overall_compliance_score": 85.5, ...}
    """
    try:
        # Parse inputs
        if isinstance(requirements_json, str):
            requirements = json.loads(requirements_json)
        else:
            requirements = requirements_json
            
        if isinstance(standards, str):
            compliance_standards = [s.strip() for s in standards.split(',') if s.strip()]
        else:
            compliance_standards = standards if standards else ["FDA", "HIPAA", "IEC 62304"]
        
        logging.info(f"Validating compliance for {len(requirements)} requirements against {compliance_standards}")
        
        client = init_genai_client()
        if not client:
            raise RuntimeError("Failed to initialize GenAI client.")
        
        # Create prompt for LLM to validate compliance
        prompt = f"""
        You are a healthcare compliance expert. Validate the following requirements against healthcare compliance standards.
        
        Requirements: {json.dumps(requirements, indent=2)}
        Compliance Standards: {', '.join(compliance_standards)}
        Software Class: {software_class}
        
        Validate compliance and return results in the following JSON format:
        {{
            "validated_requirements": [
                {{
                    "requirement_id": "REQ-001",
                    "title": "Requirement Title",
                    "compliance_status": "compliant|non-compliant|partial",
                    "compliance_gaps": ["Gap 1", "Gap 2"],
                    "standards_met": ["FDA", "HIPAA"],
                    "standards_failed": ["IEC 62304"],
                    "risk_level": "high|medium|low",
                    "recommendations": ["Recommendation 1", "Recommendation 2"]
                }}
            ],
            "compliance_gaps": ["Overall gap 1", "Overall gap 2"],
            "recommendations": ["Overall recommendation 1", "Overall recommendation 2"],
            "overall_compliance_score": 85.5,
            "standards_summary": {{
                "FDA": {{"status": "compliant", "score": 90}},
                "HIPAA": {{"status": "partial", "score": 75}},
                "IEC 62304": {{"status": "non-compliant", "score": 60}}
            }}
        }}
        
        Focus on healthcare-specific compliance requirements and provide detailed analysis.
        Consider the software class when evaluating compliance requirements.
        """
        
        # Generate content using LLM
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        # Parse LLM response
        validation_results = json.loads(response.candidates[0].content.parts[0].text)
        
        # Add validation metadata
        validation_results["validation_timestamp"] = datetime.now().isoformat()
        validation_results["software_class"] = software_class
        validation_results["total_requirements"] = len(requirements)
        
        logging.info(f"Compliance validation completed. Score: {validation_results.get('overall_compliance_score', 0):.1f}%")
        return validation_results
        
    except Exception as e:
        logging.error(f"Error validating compliance: {e}")
        return {"error": f"Failed to validate compliance: {e}"}

def _validate_requirement_compliance(requirement, compliance_standards, software_class: str):
    """Validate compliance for a single requirement."""
    gaps = []
    
    # Check for required fields based on compliance standards
    if "FDA" in compliance_standards:
        if not requirement.get("acceptance_criteria"):
            gaps.append("Missing acceptance criteria required by FDA")
        if not requirement.get("priority"):
            gaps.append("Missing priority classification required by FDA")
    
    if "IEC 62304" in compliance_standards:
        if not requirement.get("category"):
            gaps.append("Missing requirement category required by IEC 62304")
        if software_class in ["B", "C"] and not requirement.get("risk_assessment"):
            gaps.append("Missing risk assessment required for software class " + software_class)
    
    if "HIPAA" in compliance_standards and requirement.get("category") == "security":
        if "encryption" not in requirement.get("description", "").lower():
            gaps.append("Security requirements should specify encryption for HIPAA compliance")
    
    return {
        "requirement_id": requirement["id"],
        "title": requirement["title"],
        "compliance_gaps": gaps,
        "is_compliant": len(gaps) == 0
    }

def _generate_compliance_recommendations(compliance_gaps):
    """Generate recommendations based on compliance gaps."""
    recommendations = []
    
    gap_counts = {}
    for gap in compliance_gaps:
        gap_type = gap.split(" required by")[0].split(" required for")[0]
        gap_counts[gap_type] = gap_counts.get(gap_type, 0) + 1
    
    for gap_type, count in gap_counts.items():
        if "acceptance criteria" in gap_type.lower():
            recommendations.append("Add detailed acceptance criteria to all functional requirements")
        elif "priority" in gap_type.lower():
            recommendations.append("Implement priority classification system for all requirements")
        elif "category" in gap_type.lower():
            recommendations.append("Categorize all requirements by type (functional, non-functional, etc.)")
        elif "risk assessment" in gap_type.lower():
            recommendations.append("Conduct risk assessment for all requirements in software classes B and C")
        elif "encryption" in gap_type.lower():
            recommendations.append("Specify encryption requirements for all security-related requirements")
    
    return recommendations

# Traceability Tools

def process_healthcare_prd_to_test_cases(bucket_name: str, standards: str = ""):
    """
    Main function to process PRD documents from input folder and generate test cases CSV in output folder.
    
    Method Signature:
        process_healthcare_prd_to_test_cases(bucket_name: str, standards: str = "") -> dict
    
    Args:
        bucket_name (str): GCS bucket name containing PRD documents
        standards (str, optional): Comma-separated compliance standards (e.g., "FDA,HIPAA,IEC 62304")
                                  Uses default ["FDA", "HIPAA", "IEC 62304", "ISO 13485"] if empty
    
    Returns:
        dict: Dictionary containing complete processing results with keys:
            - status (str): "success" or "error"
            - bucket_name (str): GCS bucket name used
            - input_folder (str): Input folder path
            - output_folder (str): Output folder path
            - prd_documents_processed (int): Number of PRD documents processed
            - total_requirements (int): Total requirements extracted
            - total_test_cases (int): Total test cases generated
            - csv_output_path (str): GCS path to CSV file with test cases
            - pdf_traceability_path (str): GCS path to PDF traceability matrix
            - jira_export_result (dict): Results of Jira export (if enabled)
            - processing_results (list): Detailed results for each PRD document
            - compliance_standards (list): Applied compliance standards
            - processing_timestamp (str): ISO timestamp
            - error (str): Error message if processing failed
            
    Example:
        result = process_healthcare_prd_to_test_cases("my-healthcare-bucket", "FDA,HIPAA")
        # Returns: {"status": "success", "total_test_cases": 45, "csv_output_path": "gs://...", ...}
    """
    try:
        if isinstance(standards, str) and standards:
            compliance_standards = [s.strip() for s in standards.split(',') if s.strip()]
        else:
            compliance_standards = ["FDA", "HIPAA", "IEC 62304", "ISO 13485"]
        
        logging.info(f"Starting healthcare PRD to test cases processing for bucket: {bucket_name}")
        
        # Step 1: List PRD documents from input folder
        prd_documents = list_prd_documents(bucket_name)
        if not prd_documents:
            return {"error": "No PRD documents found in input folder", "prd_documents": []}
        
        all_requirements = []
        all_test_cases = []
        processing_results = []
        
        # Step 2: Process each PRD document
        for prd_path in prd_documents:
            logging.info(f"Processing PRD document: {prd_path}")
            
            # Extract requirements from PRD
            requirements_result = process_requirements_document(prd_path, compliance_standards)
            if "error" in requirements_result:
                logging.error(f"Error processing {prd_path}: {requirements_result['error']}")
                continue
            
            requirements = requirements_result.get("requirements", [])
            all_requirements.extend(requirements)
            
            # Generate test cases for these requirements
            test_cases_result = generate_test_cases(requirements, ["functional", "security", "compliance", "performance"], compliance_standards, "medium")
            if "error" in test_cases_result:
                logging.error(f"Error generating test cases for {prd_path}: {test_cases_result['error']}")
                continue
            
            test_cases = test_cases_result.get("test_cases", [])
            all_test_cases.extend(test_cases)
            
            processing_results.append({
                "prd_path": prd_path,
                "requirements_count": len(requirements),
                "test_cases_count": len(test_cases),
                "status": "success"
            })
        
        # Step 3: Create traceability matrix
        traceability_result = create_traceability_matrix(json.dumps(all_requirements), json.dumps(all_test_cases), ",".join(compliance_standards))
        
        # Create a single timestamp for this processing run
        processing_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 4: Export test cases to CSV in output folder
        csv_result = export_test_cases_to_csv(json.dumps(all_test_cases))
        
        # Step 5: Export traceability matrix to CSV in output folder
        csv_matrix_result = export_traceability_matrix_to_csv(json.dumps(all_test_cases), json.dumps(all_requirements))
        
        # Step 6: Export test cases to Jira (optional)
        jira_export_result = None
        jira_project_key = os.getenv("JIRA_PROJECT_KEY")
        if jira_project_key:
            try:
                jira_export_result = export_to_jira(json.dumps(all_test_cases), jira_project_key, "", json.dumps(traceability_result))
                logging.info(f"Jira export completed: {jira_export_result.get('total_exported', 0)} test cases exported")
            except Exception as e:
                logging.warning(f"Jira export failed: {e}")
                jira_export_result = {"error": str(e)}
        
        # Step 7: Store additional artifacts
        artifacts = {
            "test_cases": all_test_cases,
            "requirements": all_requirements,
            "traceability_matrix": traceability_result,
            "processing_results": processing_results
        }
        
        store_result = store_in_gcs(json.dumps(artifacts), bucket_name, f"output_{processing_timestamp}")
        
        # Return comprehensive results
        result = {
            "status": "success",
            "bucket_name": bucket_name,
            "output_folder": "output",
            "prd_documents_processed": len(prd_documents),
            "total_requirements": len(all_requirements),
            "total_test_cases": len(all_test_cases),
            "csv_output_path": csv_result.get('file_path', 'N/A'),
            "csv_traceability_path": csv_matrix_result.get('file_path', 'N/A'),
            "jira_export_result": jira_export_result,
            "processing_results": processing_results,
            "compliance_standards": compliance_standards,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        logging.info(f"Successfully processed {len(prd_documents)} PRD documents, generated {len(all_test_cases)} test cases")
        return result
        
    except Exception as e:
        logging.error(f"Error in main processing workflow: {e}")
        return {"error": f"Failed to process PRD documents: {e}"}

def create_traceability_matrix(requirements_json: str, test_cases_json: str, standards: str):
    """
    Create comprehensive traceability matrix linking requirements to test cases using LLM.
    
    Method Signature:
        create_traceability_matrix(requirements_json: str, test_cases_json: str, standards: str) -> dict
    
    Args:
        requirements_json (str): JSON string of source requirements
        test_cases_json (str): JSON string of generated test cases
        standards (str): Comma-separated compliance standards (e.g., "FDA,HIPAA,IEC 62304")
    
    Returns:
        dict: Dictionary containing traceability matrix with keys:
            - requirements (list): Requirements with linked test cases
            - test_cases (list): Test cases with requirement links
            - traceability_links (list): Explicit requirement-to-test-case links
            - coverage_analysis (dict): Coverage analysis results
            - compliance_mapping (dict): Mapping by compliance standard
            - creation_timestamp (str): ISO timestamp
            - total_requirements (int): Total number of requirements
            - total_test_cases (int): Total number of test cases
            - error (str): Error message if creation failed
            
    Example:
        matrix = create_traceability_matrix('{"requirements": [...]}', '{"test_cases": [...]}', "FDA,HIPAA")
        # Returns: {"requirements": [...], "traceability_links": [...], ...}
    """
    try:
        # Parse inputs
        if isinstance(requirements_json, str):
            requirements = json.loads(requirements_json)
        else:
            requirements = requirements_json
            
        if isinstance(test_cases_json, str):
            test_cases = json.loads(test_cases_json)
        else:
            test_cases = test_cases_json
            
        if isinstance(standards, str):
            compliance_standards = [s.strip() for s in standards.split(',') if s.strip()]
        else:
            compliance_standards = standards if standards else ["FDA", "HIPAA", "IEC 62304"]
        
        logging.info(f"Creating traceability matrix for {len(requirements)} requirements and {len(test_cases)} test cases")
        
        client = init_genai_client()
        if not client:
            raise RuntimeError("Failed to initialize GenAI client.")
        
        # Create prompt for LLM to create traceability matrix
        prompt = f"""
        You are a healthcare software quality assurance expert. Create a comprehensive traceability matrix linking requirements to test cases.
        
        Requirements: {json.dumps(requirements, indent=2)}
        Test Cases: {json.dumps(test_cases, indent=2)}
        Compliance Standards: {', '.join(compliance_standards)}
        
        Create a traceability matrix in the following JSON format:
        {{
            "requirements": [
                {{
                    "id": "REQ-001",
                    "title": "Requirement Title",
                    "type": "functional|non-functional|security|compliance",
                    "priority": "critical|high|medium|low",
                    "category": "security|compliance|performance|usability|reliability",
                    "linked_test_cases": ["TC-001", "TC-002"],
                    "coverage_status": "covered|partial|not_covered",
                    "compliance_standards": ["FDA", "HIPAA"]
                }}
            ],
            "test_cases": [
                {{
                    "id": "TC-001",
                    "title": "Test Case Title",
                    "type": "functional|security|compliance|performance|integration",
                    "priority": "critical|high|medium|low",
                    "requirement_id": "REQ-001",
                    "compliance_standard": "FDA|HIPAA|IEC 62304",
                    "coverage_type": "positive|negative|edge_case"
                }}
            ],
            "traceability_links": [
                {{
                    "requirement_id": "REQ-001",
                    "test_case_id": "TC-001",
                    "link_type": "covers|validates|tests",
                    "compliance_standard": "FDA",
                    "coverage_percentage": 100
                }}
            ],
            "coverage_analysis": {{
                "total_requirements": 10,
                "covered_requirements": 8,
                "uncovered_requirements": 2,
                "coverage_percentage": 80.0,
                "requirements_without_tests": ["REQ-009", "REQ-010"],
                "test_coverage_by_type": {{
                    "functional": 90,
                    "security": 85,
                    "compliance": 75,
                    "performance": 60
                }}
            }},
            "compliance_mapping": {{
                "FDA": {{
                    "requirements": ["REQ-001", "REQ-002"],
                    "test_cases": ["TC-001", "TC-002"],
                    "coverage_percentage": 85.5
                }},
                "HIPAA": {{
                    "requirements": ["REQ-003", "REQ-004"],
                    "test_cases": ["TC-003", "TC-004"],
                    "coverage_percentage": 90.0
                }}
            }}
        }}
        
        Focus on healthcare-specific traceability requirements and ensure comprehensive coverage.
        Analyze the relationship between requirements and test cases to create meaningful links.
        """
        
        # Generate content using LLM
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        # Parse LLM response
        matrix = json.loads(response.candidates[0].content.parts[0].text)
        
        # Add matrix metadata
        matrix["creation_timestamp"] = datetime.now().isoformat()
        matrix["total_requirements"] = len(requirements)
        matrix["total_test_cases"] = len(test_cases)
        
        logging.info(f"Successfully created traceability matrix with {len(matrix.get('traceability_links', []))} links")
        return matrix
        
    except Exception as e:
        logging.error(f"Error creating traceability matrix: {e}")
        return {"error": f"Failed to create traceability matrix: {e}"}

def _calculate_coverage_analysis(requirements, test_cases):
    """Calculate test coverage analysis."""
    total_requirements = len(requirements)
    covered_requirements = sum(1 for req in requirements if req["linked_test_cases"])
    
    coverage_percentage = (covered_requirements / total_requirements * 100) if total_requirements > 0 else 0
    
    return {
        "total_requirements": total_requirements,
        "covered_requirements": covered_requirements,
        "uncovered_requirements": total_requirements - covered_requirements,
        "coverage_percentage": round(coverage_percentage, 2),
        "requirements_without_tests": [req["id"] for req in requirements if not req["linked_test_cases"]]
    }

def _create_compliance_mapping(requirements, test_cases, compliance_standards):
    """Create compliance mapping for requirements and test cases."""
    mapping = {}
    
    for standard in compliance_standards:
        mapping[standard] = {
            "requirements": [],
            "test_cases": [],
            "coverage_percentage": 0
        }
        
        # Map requirements to standards
        for req in requirements:
            if req.get("category") in ["compliance", "security"] or standard.lower() in req.get("title", "").lower():
                mapping[standard]["requirements"].append(req["id"])
        
        # Map test cases to standards
        for test_case in test_cases:
            if test_case.get("compliance_standard") == standard:
                mapping[standard]["test_cases"].append(test_case["id"])
        
        # Calculate coverage
        req_count = len(mapping[standard]["requirements"])
        test_count = len(mapping[standard]["test_cases"])
        mapping[standard]["coverage_percentage"] = (test_count / req_count * 100) if req_count > 0 else 0
    
    return mapping
