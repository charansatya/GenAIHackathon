# Healthcare Test Case Generator

An AI-powered healthcare test case generator using Google's Agent Development Kit (ADK). Automatically converts Jira project requirements into comprehensive, compliant test cases with full traceability.

## 🏥 Healthcare-Focused Features

- **📋 Jira-Native**: Direct integration with Jira projects as primary input
- **👥 Human-in-the-Loop**: Interactive workflows for epic selection, test plan approval, and quality review
- **🔒 Compliance-Ready**: Built-in support for FDA 21 CFR Part 11, HIPAA, IEC 62304, ISO 13485
- **📊 Quality Assurance**: Automated validation and quality scoring
- **🎯 Risk-Based Testing**: Intelligent epic prioritization and test case generation
- **📈 Full Traceability**: Complete requirement-to-test-case mapping
- **🔄 Continuous Improvement**: Feedback collection and iterative refinement

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set up Google Cloud credentials
gcloud auth application-default login

# Configure environment variables
export GCP_PROJECT_ID="your-project-id"
export GCS_BUCKET_NAME="your-bucket-name"
export JIRA_URL="https://your-domain.atlassian.net"
export JIRA_EMAIL="your-email@example.com"
export JIRA_API_TOKEN="your-api-token"
```

### Basic Usage

```python
from healthcare_test_generator import root_agent

# The agent will guide you through the workflow:
# 1. Analyze Jira project
# 2. Select epics (Human-in-the-Loop)
# 3. Generate test plan (Human-in-the-Loop)
# 4. Generate test cases
# 5. Quality review (Human-in-the-Loop)
# 6. Export to Jira and CSV

# Simply interact with the agent:
# "Generate test cases for Jira project MED"
```

## 🏗️ Architecture

### Standard Google ADK Structure

```
healthcare-test-generator/
├── __init__.py          # Package initialization
├── agent.py             # Main agent definition (root_agent)
├── tools.py             # All tool implementations
├── prompts.py           # Healthcare-specific prompts
├── Dockerfile           # Container configuration
└── README.md            # This file
```

### How It Works

1. **`agent.py`**: Defines the main `root_agent` with:
   - Model: `gemini-2.5-pro`
   - Description: Complete workflow and capabilities
   - Tools: 17 specialized tools for the entire workflow

2. **`tools.py`**: Implements all tools:
   - Jira project analysis
   - Test plan generation
   - Test case generation
   - Quality validation
   - Compliance checking
   - Export functions (Jira, CSV, PDF)
   - Human-in-the-Loop feedback collection

3. **Workflow**: User → `agent.py` → calls tools from `tools.py` → returns results

## 📋 Complete Workflow

### 1. 🔍 Project Analysis

```python
# Agent analyzes Jira project structure
analyze_jira_project_structure(project_key="MED")

# Returns:
# - List of epics with risk levels
# - User stories categorized by type
# - Testing recommendations
# - Compliance requirements
```

### 2. 🎯 Epic Selection (Human-in-the-Loop)

```python
# Agent presents available epics
# User selects which epics to test
# AI provides risk-based recommendations

collect_user_feedback(
    test_cases="Available epics with risk assessment",
    context="epic_selection"
)
```

### 3. 📋 Test Plan Generation (Human-in-the-Loop)

```python
# Agent generates comprehensive test plan
generate_comprehensive_test_plan(
    project_key="MED",
    selected_epics="MED-1,MED-2"
)

# User reviews and approves test plan
collect_user_feedback(
    test_cases="Generated test plan",
    context="test_plan_approval"
)
```

### 4. 🧪 Test Case Generation

```python
# Agent generates detailed test cases
generate_test_cases_for_requirements(
    requirements_json='[{"id": "REQ-001", ...}]'
)

# Returns:
# - Test case ID, title, priority
# - Preconditions and test steps
# - Expected results and test data
# - Compliance mapping
# - Full traceability
```

### 5. ✅ Quality Assurance (Human-in-the-Loop)

```python
# Agent validates test case quality
validate_test_case_quality(
    test_cases_json='[{"id": "TC-001", ...}]',
    requirements_json='[{"id": "REQ-001", ...}]'
)

# User reviews quality report
collect_user_feedback(
    test_cases="Quality validation report",
    context="quality_review"
)

# Agent improves based on feedback
improve_test_cases_with_feedback(
    test_cases_json='[...]',
    feedback_json='{"issues": [...], "suggestions": [...]}'
)
```

### 6. 📤 Export & Delivery

```python
# Export to Jira as Test tasks
export_to_jira(
    test_cases_json='[...]',
    project_key="MED"
)

# Export to CSV
export_test_cases_to_csv(
    test_cases_json='[...]',
    output_filename="test_cases_MED.csv"
)

# Generate traceability matrix PDF
export_traceability_matrix_to_pdf(
    test_cases_json='[...]',
    requirements_json='[...]',
    output_filename="traceability_matrix_MED.pdf"
)

# Store in Google Cloud Storage
store_in_gcs(
    local_file_path="output/test_cases_MED.csv",
    gcs_path="projects/MED/test_cases.csv"
)
```

## 🛠️ Available Tools

### Jira Project Analysis
- `analyze_jira_project_structure`: Analyze project, identify epics and stories
- `get_epic_requirements_summary`: Get detailed epic and story analysis

### Test Planning & Generation
- `generate_comprehensive_test_plan`: Create healthcare-compliant test plan
- `generate_test_cases_for_requirements`: Generate detailed test cases
- `validate_test_case_quality`: Automated quality validation and scoring

### Compliance & Validation
- `validate_compliance_requirements`: Validate against healthcare standards

### Human-in-the-Loop
- `collect_user_feedback`: Collect user feedback at decision points
- `improve_test_cases_with_feedback`: Improve test cases based on feedback

### Export & Integration
- `export_test_cases_to_csv`: Export test cases to CSV
- `export_traceability_matrix_to_pdf`: Generate traceability matrix PDF
- `export_to_jira`: Create Jira Test tasks
- `store_in_gcs`: Store artifacts in Google Cloud Storage

### Workflow Orchestration
- `get_workflow_status`: Check workflow processing status
- `create_test_case_reports`: Create PDF and CSV reports
- `create_traceability_report`: Create traceability matrix

## 🔒 Healthcare Compliance

### Supported Standards

- **FDA 21 CFR Part 11**: Electronic records and signatures
- **HIPAA**: Protected health information security
- **IEC 62304**: Medical device software lifecycle
- **ISO 13485**: Medical devices quality management
- **ISO 27001**: Information security management
- **GDPR**: Data protection and privacy

### Compliance Features

- Automated compliance requirement validation
- Test case mapping to regulatory standards
- Audit trail and traceability
- Risk-based testing approach
- Quality assurance validation

## 📊 Output Formats

### 1. Jira Test Tasks

```
Issue Type: Task (or Test if available)
Summary: TC-001: Verify patient registration with valid data
Description:
  Preconditions: ...
  Test Steps: ...
  Expected Results: ...
  Test Data: ...
Labels: test-case, healthcare, compliance
```

### 2. CSV Export (in `output/` folder)

```csv
Test Case ID,Title,Priority,Category,Preconditions,Test Steps,Expected Results,Test Data,Compliance Standards,Requirement IDs
TC-001,Verify patient registration,High,Functional,...,...,...,...,"FDA 21 CFR Part 11, HIPAA",REQ-001
```

**File Location**: `output/test_cases_20250102_143022.csv`

### 3. Traceability Matrix CSV (in `output/` folder)

```csv
Requirement ID,Requirement Title,Requirement Type,Test Case ID,Test Case Title,Test Case Priority,Coverage Status
REQ-001,Patient Registration,Functional,TC-001,Verify patient registration,High,Covered
```

**File Location**: `output/traceability_matrix_20250102_143022.csv`

## 🚢 Deployment

### Local Development

```bash
# Run locally
python -m healthcare_test_generator
```

### Docker Deployment

```bash
# Build container
docker build -t healthcare-test-generator .

# Run container
docker run -p 8080:8080 \
  -e GCP_PROJECT_ID="your-project" \
  -e GCS_BUCKET_NAME="your-bucket" \
  -e JIRA_URL="https://your-domain.atlassian.net" \
  -e JIRA_EMAIL="your-email" \
  -e JIRA_API_TOKEN="your-token" \
  healthcare-test-generator
```

### Google Cloud Deployment

```bash
# Deploy to Cloud Run
gcloud run deploy healthcare-test-generator \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

## 📝 Example Interaction

```
User: "Generate test cases for Jira project MED"

Agent:
✓ Analyzing Jira project MED...
  Found 5 epics, 23 user stories

✓ Epic Analysis:
  - MED-1: Patient Registration (Critical Risk) - 8 stories
  - MED-2: Appointment Scheduling (High Risk) - 6 stories
  - MED-3: Medical Records (Critical Risk) - 5 stories
  - MED-4: Billing System (Medium Risk) - 3 stories
  - MED-5: Reporting (Low Risk) - 1 story

👥 HUMAN INPUT REQUIRED:
Which epics would you like to generate test cases for?
Recommendations: MED-1, MED-3 (Critical Risk)

User: "MED-1 and MED-3"

Agent:
✓ Generating comprehensive test plan...
  - Scope: Patient Registration + Medical Records
  - Strategy: Risk-based, compliance-focused
  - Test Types: Functional, Security, Compliance, Integration
  - Compliance: FDA 21 CFR Part 11, HIPAA, IEC 62304

👥 HUMAN INPUT REQUIRED:
Please review the test plan. Approve or provide feedback?

User: "Approved"

Agent:
✓ Generating test cases...
  - Generated 45 test cases
  - Coverage: 13 user stories
  - Compliance mapping: 100%
  - Traceability: Complete

✓ Quality Validation...
  - Completeness: 98%
  - Clarity: 95%
  - Compliance: 100%
  - Traceability: 100%
  - Overall Score: 97%

👥 HUMAN INPUT REQUIRED:
Quality score: 97%. Any feedback or improvements needed?

User: "Looks good, export to Jira"

Agent:
✓ Exporting to Jira...
  - Created 45 Test tasks in project MED
  - Linked to requirements

✓ Generating reports...
  - CSV: test_cases_MED_20250102_143022.csv
  - PDF: traceability_matrix_MED_20250102_143022.pdf

✓ Storing in Google Cloud Storage...
  - gs://your-bucket/projects/MED/20250102_143022/

✅ Complete! Generated 45 test cases with full traceability.
```

## 🤝 Contributing

This project follows Google ADK best practices. When contributing:

1. Keep all agent definitions in `agent.py`
2. Keep all tool implementations in `tools.py`
3. Follow type hints and docstring conventions
4. Test with real Jira projects
5. Ensure compliance validation works

## 📄 License

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0

## 🔗 Resources

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Jira REST API](https://developer.atlassian.com/cloud/jira/platform/rest/v3/)
- [FDA 21 CFR Part 11](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/part-11-electronic-records-electronic-signatures-scope-and-application)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
- [IEC 62304](https://www.iso.org/standard/38421.html)

---

**Version**: 1.0.0  
**Last Updated**: 2025-11-02  
**Status**: Production Ready
