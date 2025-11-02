# Healthcare Test Case Generator

AI-powered system that automatically converts healthcare software requirements into compliant, traceable test cases integrated with enterprise toolchains.

## 🏗️ Architecture

This project follows the Google ADK samples structure pattern with the following components:

### Core Files

- **`__init__.py`** - Package initialization and exports
- **`agent.py`** - Main healthcare test generator agent using Google ADK
- **`prompts.py`** - Healthcare-specific prompts for LLM interactions
- **`tools.py`** - LLM-powered tools for document processing, compliance validation, and test generation
- **`requirements.txt`** - Python dependencies

### Key Features

- **LLM-Powered Processing**: Uses Google's Gemini 2.5 Pro for intelligent document analysis and test case generation
- **Healthcare Compliance**: Validates against FDA, HIPAA, IEC 62304, ISO 13485, ISO 27001, and GDPR standards
- **Automated PRD Processing**: Processes PRD documents from GCS input folder and generates test cases CSV in output folder
- **CSV Export**: Exports generated test cases to CSV format in the specified output folder
- **PDF Traceability Report**: Creates comprehensive traceability matrix as a professional PDF document
- **Jira Integration**: Export test cases to Jira (extraction commented out - focus on PRD processing)
- **Google Cloud Storage**: Store generated artifacts in GCS
- **Traceability Matrix**: Create comprehensive requirement-to-test traceability

## 🚀 Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Google Cloud authentication:
```bash
gcloud auth application-default login
```

3. Set environment variables:
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GCS_BUCKET="hackathon-11"
```

### Usage

The healthcare test generator is designed to work with Google ADK. Import and use the agent:

```python
from healthcare_test_generator import root_agent

# The agent is ready to use with ADK
# It will automatically handle PDF processing from GCS
# and generate healthcare-compliant test cases
```

#### Example Agent Usage

```python
# Process PRD documents from input folder and generate test cases CSV
# The agent will use its tools to:
# 1. List PRD documents from GCS input folder (hackathon-11/Input)
# 2. Extract requirements from PDF documents using LLM
# 3. Analyze requirements and validate compliance
# 4. Generate comprehensive test cases
# 5. Create traceability matrix
# 6. Export test cases to CSV in output folder (hackathon-11/output)
# 7. Export traceability matrix to PDF in output folder
# 8. Store additional artifacts in GCS
```

## 🛠️ Agentic Tools & Orchestration

The Healthcare Test Generator provides **independent, orchestrable tools** that can work together or separately for maximum flexibility.

### 📁 Document Discovery & Input
- `list_prd_documents()` - List available PRD documents in GCS bucket
- `extract_pdf_text()` - Extract raw text from PDF documents
- `analyze_requirements_from_text()` - Analyze text and extract structured requirements

### 🔄 Document Processing
- `process_requirements_document()` - Complete PDF → requirements workflow

### ✅ Compliance & Validation
- `validate_compliance_requirements()` - Validate requirements against healthcare standards
- `validate_processing_step()` - Validate intermediate processing results for quality assurance

### 🧪 Test Case Generation
- `generate_test_cases()` - Generate comprehensive test cases from requirements

### 🔗 Traceability & Analysis
- `create_traceability_matrix()` - Create requirement-to-test traceability matrix

### 📊 Workflow Monitoring
- `get_workflow_status()` - Check status and progress of workflow processing

### 📤 Export & Output
- `export_test_cases_to_csv()` - Export test cases to CSV format in timestamped folders
- `export_traceability_matrix_to_pdf()` - Export traceability matrix to PDF format
- `export_to_jira()` - Export test cases to Jira as tickets (optional)
- `store_in_gcs()` - Store artifacts in Google Cloud Storage

### 🎯 Agentic Orchestration
The agent intelligently orchestrates the individual tools above to create complete workflows. No single "do everything" function - each step is visible and controllable.

## 🤖 Agentic Orchestration Examples

### Step-by-Step Orchestration
```python
# 1. Discover documents
documents = list_prd_documents("hackathon-11")

# 2. Validate discovery step
validation = validate_processing_step("document_discovery", json.dumps(documents))

# 3. Process each document individually
for doc in documents:
    requirements = process_requirements_document(doc, "FDA,HIPAA")
    
    # 4. Validate requirements extraction
    req_validation = validate_processing_step("requirements_extraction", 
                                            json.dumps(requirements), 
                                            "id,title,description")
    
    # 5. Generate test cases if validation passes
    if req_validation.get("is_valid"):
        test_cases = generate_test_cases(json.dumps(requirements), 
                                       "functional,security", 
                                       "FDA,HIPAA", "high")

# 6. Check workflow status
status = get_workflow_status("hackathon-11")
```

### Monitoring and Quality Assurance
```python
# Check current workflow status
status = get_workflow_status("hackathon-11", "output_20250921_153045")

# Validate each processing step
validation = validate_processing_step("test_generation", test_cases_json, 
                                    "id,title,test_type,priority")

# Monitor data quality throughout the process
if validation.get("data_quality_score", 0) < 80:
    # Take corrective action or retry
    pass
```

## 📋 Compliance Standards

- **FDA 21 CFR Part 11** - Electronic Records and Signatures
- **IEC 62304** - Medical Device Software Life Cycle Processes
- **ISO 13485** - Medical Devices Quality Management Systems
- **ISO 27001** - Information Security Management Systems
- **HIPAA** - Health Insurance Portability and Accountability Act
- **GDPR** - General Data Protection Regulation

## 🔧 Configuration

Set the following environment variables:

```bash
# Required
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# GCS Bucket Configuration
GCS_BUCKET=hackathon-11

# Optional: Jira Integration (for test case export)
JIRA_URL=https://your-domain.atlassian.net
JIRA_USERNAME=your-email@company.com
JIRA_API_TOKEN=your-api-token
JIRA_PROJECT_KEY=MED
```

## 📄 License

Copyright 2025 Google LLC. Licensed under the Apache License, Version 2.0.
