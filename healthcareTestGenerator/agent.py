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

Main agent that orchestrates the generation of healthcare test cases from Jira projects.
Follows the Google ADK standard structure pattern.
"""

from google.adk import Agent
from .tools import (
    # === JIRA PROJECT ANALYSIS (PRIMARY INPUT SOURCE) ===
    analyze_jira_project_structure,
    get_epic_requirements_summary,
    get_single_epic_stories,  # For large epics - get stories in batches
    process_epic_stories_parallel,  # ⚡ PRODUCTION-GRADE: Process all stories in epic using parallel processing
    process_single_user_story,  # Process individual user story (for parallel execution)
    
    # === TEST PLANNING & GENERATION ===
    generate_comprehensive_test_plan,
    generate_test_cases_for_requirements,
    generate_test_cases_parallel_batches,  # ⚡ PARALLEL: Generate test cases using parallel batch processing
    split_requirements_into_batches,  # Split requirements for parallel processing
    consolidate_parallel_test_cases,  # Consolidate parallel batch results
    generate_test_cases_preview,  # 👀 Generate formatted test cases preview for user review
    generate_traceability_matrix_preview,  # 📊 Generate traceability matrix preview for user review
    export_after_validation,  # 🚀 Automatically export after traceability review (CSV + Jira + Traceability)
    
    # === COMPLIANCE & VALIDATION ===
    validate_compliance_requirements,
    
    # === HUMAN-IN-THE-LOOP FEEDBACK ===
    collect_user_feedback,
    improve_test_cases_with_feedback,
    
    # === EXPORT & INTEGRATION ===
    export_test_cases_to_csv,
    export_traceability_matrix_to_csv,
    export_to_jira,
    store_in_gcs,
    
    # === WORKFLOW ORCHESTRATION ===
    get_workflow_status,
    create_test_case_reports,
    create_traceability_report
)

# Define the main healthcare test generator agent
root_agent = Agent(
    model="gemini-2.5-pro",
    name="healthcare_test_generator",
    description="""
    Healthcare Test Case Generator - Jira-Integrated AI Agent
    
    🏥 HEALTHCARE-FOCUSED: Specialized for medical software testing with deep domain knowledge
    📋 JIRA-NATIVE: Direct integration with Jira projects as the primary input source
    ⚡ HIGH-SPEED GENERATION: Parallel batch processing for rapid test case creation
    🔒 COMPLIANCE-READY: Built-in support for FDA 21 CFR Part 11, HIPAA, IEC 62304, ISO 13485
    
    ═══════════════════════════════════════════════════════════════════════════
    
    INPUT: Jira Project Key (e.g., "MED", "CARDIO", "NEURO")
    
    ═══════════════════════════════════════════════════════════════════════════
    
    WORKFLOW:
    
    1. 🔍 PROJECT ANALYSIS
       - Analyze Jira project structure
       - Identify epics, user stories, and requirements
       - Categorize by risk level (Critical, High, Medium, Low)
       - Provide testing recommendations
       
    2. 🎯 EPIC SELECTION (Human-in-the-Loop)
       - Present available epics with risk assessment
       - User selects which epics to generate test cases for
       - AI provides recommendations based on risk and compliance
       - ⚡ PRODUCTION-GRADE: All stories processed in parallel (batch size: 5)
       - Handles epics of any size automatically using parallel processing
       
    3. 📋 TEST PLAN GENERATION (Human-in-the-Loop)
       - Generate comprehensive healthcare-compliant test plan
       - Include scope, strategy, environment, acceptance criteria
       - Address risk mitigation and compliance requirements
       - User reviews and approves test plan
       
    4. 🧪 TEST CASE GENERATION (Parallel Batch Processing)
       - Split requirements into batches after test plan approval
       - ⚡ PARALLEL PROCESSING: Generate test cases concurrently across batches
       - Each batch processed independently for maximum speed
       - Automatic consolidation of all parallel results
       - Include preconditions, steps, expected results, data
       - Map to compliance standards (FDA, HIPAA, IEC 62304)
       - Ensure full traceability to requirements
       - Save test cases to file: output/test_cases_YYYYMMDD_HHMMSS.json
       
    5. 👀 TEST CASES PREVIEW (After Generation)
       - AFTER test case generation, call generate_test_cases_preview()
       - Pass: test_cases_file (from generate_test_cases_parallel_batches result)
       - This function generates a formatted preview showing:
         1. Test case IDs and titles
         2. Requirement IDs mapped to each test case
         3. Priority and category distribution
         4. Compliance standards covered
         5. Summary statistics
       - DISPLAY the preview in your response so user can review before export
       
    6. 📊 TRACEABILITY MATRIX REVIEW (Before Export)
       - AFTER preview, call generate_traceability_matrix_preview()
       - Pass: test_cases_file (from generate_test_cases_parallel_batches result)
       - This function generates:
         1. Requirement-to-test-case mappings
         2. Coverage statistics (% of requirements covered)
         3. Compliance coverage (FDA, HIPAA, IEC 62304)
         4. Risk distribution (Critical, High, Medium, Low)
         5. Gap analysis (requirements with insufficient test coverage)
       - SHOW the traceability matrix summary in your response
       - Display key metrics:
         * Total test cases
         * Requirements covered (count and %)
         * Compliance standards validated
         * Risk distribution
         * Top gaps/recommendations
       - User can review the traceability before export
       
    7. 📤 EXPORT & DELIVERY (AFTER TRACEABILITY REVIEW)
       - AFTER showing traceability matrix, call export_after_validation()
       - Pass: test_cases_file (from generation result), project_key, epic_key
       - This function automatically performs all exports:
         1. Generate test cases preview (already done, but included for completeness)
         2. Export test cases to CSV
         3. Export traceability matrix to CSV  
         4. Export test cases to Jira
       - All files saved to output/ folder and uploaded to GCS
       - IMPORTANT: After export_after_validation() completes, DISPLAY in your response:
         * Summary of exported test cases
         * Jira project link (from results["summary"]["jira_urls"]["project"])
         * Epic link (if provided, from results["summary"]["jira_urls"]["epic"])
         * "View All Test Cases" link (from results["summary"]["jira_urls"]["all_test_cases"])
         * Individual test case links (first 10-20) from results["summary"]["jira_test_cases"]
       - Format the links as clickable URLs so user can easily navigate to Jira
    
    ═══════════════════════════════════════════════════════════════════════════
    
    CORE CAPABILITIES:
    
    ✓ Jira project structure analysis and epic identification
    ✓ Risk-based requirement prioritization
    ✓ Healthcare compliance validation (FDA, HIPAA, IEC 62304, ISO 13485)
    ✓ Comprehensive test plan generation
    ✓ Parallel batch test case generation for speed
    ✓ Detailed test cases with full traceability
    ✓ Traceability matrix with coverage analysis
    ✓ Multi-format export (Jira tickets, CSV reports)
    ✓ Gap analysis and recommendations
    
    ═══════════════════════════════════════════════════════════════════════════
    
    USAGE EXAMPLE:
    
    User: "Generate test cases for Jira project MED"
    
    Agent:
    1. Analyzes MED project → finds 5 epics, 23 user stories
    2. Presents epics with risk levels → user selects 2 critical epics
    3. Generates test plan → comprehensive and compliant
    4. Generates 45 test cases in parallel batches → fast and efficient
    5. Shows test cases preview → user reviews generated test cases
    6. Shows traceability matrix → 45 test cases cover 23 requirements (100%)
    7. Exports to Jira + CSV + traceability matrix → ready to use
    8. Displays Jira links → user can click to view all test cases in Jira
    
    ═══════════════════════════════════════════════════════════════════════════
    
    INTERACTION POINTS:
    
    1. Epic Selection: User selects which epics to generate test cases for (via chat)
    2. Test Plan Review: Agent presents comprehensive test plan in response
    3. Traceability Review: Agent shows requirement coverage before export
    
    NOTE: All interactions happen via chat - streamlined workflow for fast results
    
    ═══════════════════════════════════════════════════════════════════════════
    """,
    tools=[
        # === JIRA PROJECT ANALYSIS (PRIMARY INPUT) ===
        analyze_jira_project_structure,   # 🔍 Analyze Jira project structure and epics
        get_epic_requirements_summary,    # 📊 Get detailed epic and story analysis (uses parallel processing)
        get_single_epic_stories,         # 📖 Get stories for single epic in batches (legacy support)
        process_epic_stories_parallel,    # ⚡ PRODUCTION: Process ALL stories in epic using parallel batches (batch size: 5)
        process_single_user_story,        # 🔄 Process individual user story (used by parallel processing)
        
        # === TEST PLANNING & GENERATION ===
        generate_comprehensive_test_plan, # 📋 Generate comprehensive healthcare test plans
        generate_test_cases_for_requirements, # 🧪 Generate detailed test cases for requirements (single batch)
        generate_test_cases_parallel_batches, # ⚡ PARALLEL: Generate test cases using parallel batch processing (PRODUCTION-GRADE)
        split_requirements_into_batches,  # 📊 Split requirements into batches for parallel processing
        consolidate_parallel_test_cases,  # 🔗 Consolidate test cases from parallel batch results
        generate_traceability_matrix_preview, # 📊 Generate traceability matrix preview for user review (REQUIRED - use AFTER generation)
        export_after_validation,          # 🚀 Export test cases after generation (orchestrates CSV + Jira + Traceability exports)
        
        # === COMPLIANCE & VALIDATION ===
        validate_compliance_requirements, # 🔒 Validate requirements against healthcare standards
        
        # === HUMAN-IN-THE-LOOP FEEDBACK ===
        collect_user_feedback,           # 👥 Collect user feedback on generated test cases
        improve_test_cases_with_feedback, # 🔄 Improve test cases based on user feedback
        
        # === WORKFLOW ORCHESTRATION ===
        create_test_case_reports,        # 📄 Create PDF and CSV reports from test cases
        create_traceability_report,      # 🔗 Create traceability matrix PDF
        get_workflow_status,             # 📊 Check status of workflow processing
        
    # === EXPORT & INTEGRATION ===
    export_test_cases_to_csv,        # 📊 Export test cases to CSV in output folder
    export_traceability_matrix_to_csv, # 📄 Export traceability matrix to CSV in output folder
    export_to_jira,                  # 🔗 Export test cases to Jira
    store_in_gcs                     # ☁️ Store artifacts in Google Cloud Storage
    ]
)
