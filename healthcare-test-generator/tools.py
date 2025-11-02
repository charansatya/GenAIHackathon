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
# Simplified type hints for ADK compatibility
from datetime import datetime, timedelta
import requests
from google.cloud import storage
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from typing import Union

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
        # Parse inputs
        if isinstance(test_cases_json, str):
            test_cases = json.loads(test_cases_json)
        else:
            test_cases = test_cases_json
            
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
        jira_username = os.getenv("JIRA_USERNAME")
        jira_token = os.getenv("JIRA_API_TOKEN")
        
        if not all([jira_url, jira_username, jira_token]):
            raise ValueError("Jira credentials not configured")
        
        jira_client = JIRA(
            server=jira_url,
            basic_auth=(jira_username, jira_token)
        )
        
        exported_cases = []
        for i, test_case in enumerate(test_cases):
            try:
                # Create Jira issue
                issue_dict = {
                    'project': {'key': project_key},
                    'summary': test_case.get("title", f"Test Case {i + 1}"),
                    'description': test_case.get("description", ""),
                    'issuetype': {'name': 'Test'},
                    'labels': test_case.get("labels", [])
                }
                
                # Only add priority if it exists in the test case (avoid Jira field issues)
                if test_case.get("priority"):
                    try:
                        issue_dict['priority'] = {'name': test_case.get("priority", "Medium")}
                    except:
                        # Skip priority if it causes issues
                        pass
                
                # Add epic link if provided
                if epic_key:
                    issue_dict['customfield_10014'] = epic_key  # Epic Link custom field
                
                # Create the issue
                new_issue = jira_client.create_issue(fields=issue_dict)
                
                exported_cases.append({
                    "key": new_issue.key,
                    "summary": new_issue.fields.summary,
                    "description": new_issue.fields.description,
                    "issue_type": "Test",
                    "priority": test_case.get("priority", "Medium"),
                    "labels": test_case.get("labels", []),
                    "epic_link": epic_key,
                    "status": "To Do"
                })
                
            except Exception as e:
                logging.error(f"Error creating Jira ticket for test case {i + 1}: {e}")
                continue
        
        result = {
            "exported_cases": exported_cases,
            "project_key": project_key,
            "epic_key": epic_key,
            "export_timestamp": datetime.now().isoformat(),
            "total_exported": len(exported_cases)
        }
        
        if traceability_matrix:
            result["traceability_matrix"] = traceability_matrix
        
        logging.info(f"Successfully exported {len(exported_cases)} test cases to Jira")
        return result
        
    except Exception as e:
        logging.error(f"Error exporting to Jira: {e}")
        return {"error": f"Failed to export to Jira: {e}"}


# Google Cloud Storage Tools

def export_test_cases_to_csv(test_cases_json: str, bucket_name: str, filename: str = "", timestamp: str = ""):
    """
    Export test cases to CSV format in the GCS bucket with datetime-stamped folder.
    
    Method Signature:
        export_test_cases_to_csv(test_cases_json: str, bucket_name: str, filename: str = "", timestamp: str = "") -> str
    
    Args:
        test_cases_json (str): JSON string of test cases to export
        bucket_name (str): GCS bucket name where CSV will be stored
        filename (str, optional): Filename for CSV file (auto-generated if empty)
    
    Returns:
        str: GCS path to the exported CSV file (gs://bucket-name/path/to/file.csv)
              Returns error message string if export failed
              
    Example:
        csv_path = export_test_cases_to_csv('{"test_cases": [...]}', "my-bucket", "test_cases.csv")
        # Returns: "gs://my-bucket/hackathon-11/output/test_cases.csv"
    """
    try:
        # Parse test cases
        if isinstance(test_cases_json, str):
            test_cases = json.loads(test_cases_json)
        else:
            test_cases = test_cases_json
            
        logging.info(f"Exporting {len(test_cases)} test cases to CSV in bucket: {bucket_name}")
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Create datetime-stamped folder and filename
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"output_{timestamp}"
        
        if not filename or filename == "":
            filename = f"healthcare_test_cases_{timestamp}.csv"
        
        # Create CSV content
        csv_content = _create_test_cases_csv(test_cases)
        
        # Upload to GCS with datetime folder
        blob_path = f"{output_folder}/{filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(csv_content, content_type='text/csv')
        
        gcs_path = f"gs://{bucket_name}/{blob_path}"
        logging.info(f"Successfully exported test cases to CSV: {gcs_path}")
        return gcs_path
        
    except Exception as e:
        logging.error(f"Error exporting test cases to CSV: {e}")
        return f"Error: {e}"

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

def export_traceability_matrix_to_pdf(traceability_matrix_json: str, bucket_name: str, filename: str = "", timestamp: str = ""):
    """
    Export traceability matrix to PDF format in the GCS bucket with datetime-stamped folder.
    
    Method Signature:
        export_traceability_matrix_to_pdf(traceability_matrix_json: str, bucket_name: str, filename: str = "", timestamp: str = "") -> str
    
    Args:
        traceability_matrix_json (str): JSON string of traceability matrix data
        bucket_name (str): GCS bucket name where PDF will be stored
        filename (str, optional): Filename for PDF file (auto-generated if empty)
    
    Returns:
        str: GCS path to the exported PDF file (gs://bucket-name/path/to/file.pdf)
             Returns error message string if export failed
             
    Example:
        pdf_path = export_traceability_matrix_to_pdf('{"requirements": [...]}', "my-bucket", "matrix.pdf")
        # Returns: "gs://my-bucket/hackathon-11/output/matrix.pdf"
    """
    try:
        # Parse traceability matrix
        if isinstance(traceability_matrix_json, str):
            traceability_matrix = json.loads(traceability_matrix_json)
        else:
            traceability_matrix = traceability_matrix_json
            
        logging.info(f"Exporting traceability matrix to PDF in bucket: {bucket_name}")
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Create datetime-stamped folder and filename
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"output_{timestamp}"
        
        if not filename or filename == "":
            filename = f"traceability_matrix_{timestamp}.pdf"
        
        # Create PDF content
        pdf_content = _create_traceability_matrix_pdf(traceability_matrix)
        
        # Upload to GCS with datetime folder
        blob_path = f"{output_folder}/{filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(pdf_content, content_type='application/pdf')
        
        gcs_path = f"gs://{bucket_name}/{blob_path}"
        logging.info(f"Successfully exported traceability matrix to PDF: {gcs_path}")
        return gcs_path
        
    except Exception as e:
        logging.error(f"Error exporting traceability matrix to PDF: {e}")
        return f"Error: {e}"

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
        csv_path = export_test_cases_to_csv(json.dumps(all_test_cases), bucket_name, "", processing_timestamp)
        
        # Step 5: Export traceability matrix to PDF in output folder
        pdf_path = export_traceability_matrix_to_pdf(json.dumps(traceability_result), bucket_name, "", processing_timestamp)
        
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
            "output_folder": f"output_{processing_timestamp}",
            "prd_documents_processed": len(prd_documents),
            "total_requirements": len(all_requirements),
            "total_test_cases": len(all_test_cases),
            "csv_output_path": csv_path,
            "pdf_traceability_path": pdf_path,
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
