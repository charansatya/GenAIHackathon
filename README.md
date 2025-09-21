# 🏥 Healthcare Test Case Generator

An AI-powered agentic system that automatically converts healthcare software requirements (PRD documents) into compliant, traceable test cases and traceability matrices. Built using Google's Agent Development Kit (ADK) for enterprise deployment on Vertex AI Agent Engine.

## 🎯 Overview

This prototype demonstrates the future of healthcare software validation - where AI agents handle the complexity of regulatory compliance while maintaining full transparency and human oversight.

### Key Features
- **Automated PRD Processing**: Extracts requirements from PDF documents using Gemini 2.5 Pro
- **Multi-Standard Compliance**: Supports FDA, HIPAA, IEC 62304, ISO 13485, ISO 27001
- **Agentic Orchestration**: 16 independent, transparent tools working together
- **Professional Outputs**: CSV test cases and PDF traceability matrices
- **Enterprise Ready**: Built for Google Cloud Agent Engine deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Cloud Project with Vertex AI enabled
- Google Cloud Storage bucket
- ADK (Agent Development Kit) installed

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/charansatya/GenAIHackathon.git
cd GenAIHackathon
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GCS_BUCKET="hackathon-11"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

4. **Run the agent locally**
```bash
adk run healthcare-test-generator
```

5. **Access the web interface**
```bash
adk web --port 8001
```

## 🏗️ Architecture

### Agentic Design
The system uses 16 independent tools organized into categories:

- **📁 Document Discovery**: `list_prd_documents`, `extract_pdf_text`
- **🔄 Processing**: `analyze_requirements_from_text`, `process_requirements_document`
- **✅ Validation**: `validate_compliance_requirements`, `validate_processing_step`
- **🧪 Generation**: `generate_test_cases`, `generate_test_cases_simple`
- **🔗 Traceability**: `create_traceability_matrix`, `create_traceability_matrix_simple`
- **📊 Monitoring**: `get_workflow_status`
- **📤 Export**: `export_test_cases_to_csv`, `export_traceability_matrix_to_pdf`, `export_to_jira`

### Technology Stack
- **Google ADK**: Agent framework for enterprise deployment
- **Gemini 2.5 Pro**: Large language model for intelligent processing
- **Google Cloud Storage**: Document input/output management
- **Pandas**: Robust CSV generation and data handling
- **ReportLab**: Professional PDF report generation

## 📋 Compliance Standards

- **FDA 21 CFR Part 11**: Electronic Records and Signatures
- **HIPAA**: Healthcare data privacy and security
- **IEC 62304**: Medical device software lifecycle processes
- **ISO 13485**: Medical device quality management systems
- **ISO 27001**: Information security management

## 🛠️ Usage Examples

### Basic Workflow
```python
from healthcare_test_generator import root_agent

# The agent will automatically:
# 1. List PRD documents from GCS bucket
# 2. Extract and analyze requirements
# 3. Generate comprehensive test cases
# 4. Create traceability matrix
# 5. Export to CSV and PDF formats
```

### Step-by-Step Processing
```python
# 1. Discover documents
documents = list_prd_documents("hackathon-11")

# 2. Generate test cases
test_cases = generate_test_cases_simple("17", "functional,security", "FDA,HIPAA", "high")

# 3. Create traceability matrix
matrix = create_traceability_matrix_simple("17", "45", "FDA,HIPAA")

# 4. Check workflow status
status = get_workflow_status("hackathon-11")
```

## 📊 Output Structure

### Generated Files
```
hackathon-11/
├── prd_document.pdf (input)
├── output_20250921_143052/
│   ├── healthcare_test_cases_20250921_143052.csv
│   ├── traceability_matrix_20250921_143052.pdf
│   └── artifacts_20250921_143052.json
```

### CSV Test Cases
- Test Case ID, Title, Description
- Test Type, Priority, Requirement ID
- Test Steps, Expected Results
- Compliance Standard, Test Data
- Prerequisites, Risk Level, Status

### PDF Traceability Matrix
- Requirements to test cases mapping
- Coverage analysis by compliance standard
- Professional formatting for regulatory submissions

## 🚀 Deployment

### Agent Engine Deployment
```bash
# Deploy to Vertex AI Agent Engine
adk deploy healthcare-test-generator --project your-project-id --region us-central1
```

### AgentSpace Registration
Once deployed to Agent Engine, the agent can be registered in Google AgentSpace for enterprise discovery and integration.

## 💰 Business Value

- **80% Time Reduction**: Automated test case generation vs. manual creation
- **100% Compliance Coverage**: Ensures all regulatory requirements are tested
- **Audit-Ready Documentation**: Professional reports for regulatory submissions
- **Enterprise Scaling**: Handles multiple PRD documents simultaneously
- **Cost Effective**: Significantly lower than traditional compliance solutions

## 🔧 Configuration

### Environment Variables
```bash
# Required
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GCS_BUCKET=hackathon-11

# Optional: Jira Integration
JIRA_URL=https://your-domain.atlassian.net
JIRA_USERNAME=your-email@company.com
JIRA_API_TOKEN=your-api-token
JIRA_PROJECT_KEY=MED
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🏥 Healthcare Compliance Notice

This tool is designed to assist with healthcare software testing and compliance documentation. It should be used as part of a comprehensive quality assurance process and does not replace professional regulatory guidance.

## 📞 Support

For questions, issues, or enterprise deployment assistance:
- Create an issue in this repository
- Contact: [Your contact information]

## 🎯 Roadmap

- [ ] Deploy to Vertex AI Agent Engine
- [ ] Register in Google AgentSpace
- [ ] Add support for additional compliance standards
- [ ] Integration with more healthcare systems
- [ ] Multi-language support for international regulations

---

**Built with ❤️ for the healthcare community using Google AI technologies**
