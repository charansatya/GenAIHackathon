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
Healthcare Test Case Generator Prompts

Contains all prompts used by the healthcare test generator agent.
Follows the Google ADK samples structure pattern.
"""

# Main system prompt for the healthcare test generator
SYSTEM_PROMPT = """
You are a Healthcare Test Case Generator, an AI-powered system that automatically converts healthcare software requirements into compliant, traceable test cases integrated with enterprise toolchains.

Your primary responsibilities:
1. Analyze healthcare software requirements from various sources (documents, Jira, etc.)
2. Validate compliance with healthcare regulations (FDA, IEC 62304, HIPAA, ISO 13485, ISO 27001, GDPR)
3. Generate comprehensive test cases covering functional, security, compliance, and performance aspects
4. Create traceability matrices linking requirements to test cases
5. Export results to enterprise tools like Jira and store artifacts in Google Cloud Storage

Key capabilities:
- Process multiple document formats (PDF, Word, XML, Markup)
- Understand healthcare-specific regulations and domain requirements
- Generate test cases with proper traceability and compliance validation
- Integrate with ALM platforms (Jira, Polarion, Azure DevOps)
- Ensure GDPR-compliant data handling and processing

Always prioritize:
- Patient safety and data protection
- Regulatory compliance and audit readiness
- Comprehensive test coverage
- Clear traceability and documentation
- Healthcare domain expertise

Use the available tools to process requirements, validate compliance, generate test cases, and manage traceability.
"""

# Prompt for requirements analysis
REQUIREMENTS_ANALYSIS_PROMPT = """
Analyze the provided healthcare software requirements and extract:

1. **Functional Requirements**:
   - Core healthcare functionality
   - Clinical workflows and processes
   - User interactions and interfaces
   - Data processing and storage requirements

2. **Non-Functional Requirements**:
   - Performance and scalability requirements
   - Security and privacy requirements
   - Reliability and availability requirements
   - Usability and accessibility requirements

3. **Compliance Requirements**:
   - FDA 21 CFR Part 11 compliance
   - IEC 62304 medical device software requirements
   - HIPAA privacy and security rules
   - ISO 13485 quality management
   - ISO 27001 information security
   - GDPR data protection

4. **Healthcare-Specific Requirements**:
   - Patient data handling and protection
   - Clinical decision support functionality
   - Medical device integration requirements
   - Healthcare interoperability standards (HL7 FHIR, DICOM)

5. **Risk and Safety Requirements**:
   - Patient safety considerations
   - Risk management requirements
   - Hazard analysis and mitigation
   - Safety classification (IEC 62304)

Provide a structured analysis with clear categorization and priority levels.
"""

# Prompt for compliance validation
COMPLIANCE_VALIDATION_PROMPT = """
Validate the healthcare software requirements against applicable regulatory standards:

**FDA Compliance (21 CFR Part 11)**:
- Electronic records and signatures validation
- Audit trail requirements
- System validation and verification
- Software as Medical Device (SaMD) guidelines

**IEC 62304 Medical Device Software**:
- Software safety classification (Class A, B, C)
- Software lifecycle processes
- Risk management and hazard analysis
- Software development and maintenance processes

**HIPAA Privacy and Security**:
- Patient data protection requirements
- Access control and authentication
- Data encryption and transmission security
- Breach notification and incident response

**ISO 13485 Quality Management**:
- Quality management system requirements
- Design and development controls
- Risk management processes
- Post-market surveillance

**ISO 27001 Information Security**:
- Information security management
- Risk assessment and treatment
- Security controls implementation
- Continuous monitoring and improvement

**GDPR Data Protection**:
- Data subject rights and consent
- Data processing lawfulness
- Privacy by design principles
- Data breach notification

Provide compliance status, gaps identified, and recommendations for each applicable standard.
"""

# Prompt for test case generation
TEST_GENERATION_PROMPT = """
Generate comprehensive test cases for the healthcare software requirements:

**Test Case Categories**:

1. **Functional Test Cases**:
   - Core healthcare functionality testing
   - Clinical workflow validation
   - User interface and interaction testing
   - Data processing and storage testing

2. **Security Test Cases**:
   - Authentication and authorization testing
   - Data encryption and protection testing
   - Access control and audit trail testing
   - Vulnerability and penetration testing

3. **Compliance Test Cases**:
   - FDA 21 CFR Part 11 compliance testing
   - IEC 62304 safety classification testing
   - HIPAA privacy and security testing
   - ISO 13485 quality management testing
   - GDPR data protection testing

4. **Performance Test Cases**:
   - Load and stress testing
   - Scalability and capacity testing
   - Response time and throughput testing
   - Resource utilization testing

5. **Integration Test Cases**:
   - Healthcare system interoperability testing
   - HL7 FHIR data exchange testing
   - DICOM medical imaging integration
   - Third-party system integration

6. **Usability Test Cases**:
   - Healthcare workflow optimization
   - User experience and interface testing
   - Accessibility and compliance testing
   - Training and documentation validation

**Test Case Structure**:
- Test Case ID and Title
- Test Objective and Scope
- Prerequisites and Setup
- Test Steps (detailed)
- Expected Results
- Pass/Fail Criteria
- Traceability to Requirements
- Compliance Standards Addressed
- Risk Level and Priority

Ensure comprehensive coverage and clear traceability to requirements.
"""

# Prompt for traceability matrix creation
TRACEABILITY_PROMPT = """
Create a comprehensive traceability matrix for the healthcare software project:

**Matrix Components**:

1. **Requirements Traceability**:
   - Map each requirement to corresponding test cases
   - Identify test coverage gaps
   - Ensure bidirectional traceability
   - Include requirement priority and risk level

2. **Compliance Traceability**:
   - Link requirements to applicable regulatory standards
   - Map test cases to compliance validation
   - Track compliance coverage and gaps
   - Document regulatory evidence

3. **Test Coverage Analysis**:
   - Calculate test coverage percentage
   - Identify untested requirements
   - Prioritize test case execution
   - Assess risk-based testing needs

4. **Impact Analysis**:
   - Requirement change impact assessment
   - Test case modification requirements
   - Compliance impact evaluation
   - Risk assessment updates

**Matrix Format**:
- Requirement ID and Description
- Test Case ID and Title
- Compliance Standard(s)
- Coverage Status (Covered/Partial/Not Covered)
- Risk Level (High/Medium/Low)
- Priority (Critical/High/Medium/Low)
- Notes and Comments

Ensure complete traceability and audit readiness for regulatory compliance.
"""
