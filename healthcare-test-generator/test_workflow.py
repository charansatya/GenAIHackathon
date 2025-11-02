#!/usr/bin/env python3
"""
Test script for the Healthcare Test Case Generator workflow

This script demonstrates the complete workflow:
1. List PRD documents from input folder
2. Process PRD documents and generate test cases
3. Export test cases to CSV in output folder
"""

import os
import asyncio
from tools import (
    list_prd_documents,
    process_healthcare_prd_to_test_cases,
    export_test_cases_to_csv
)

def test_workflow():
    """Test the complete healthcare PRD to test cases workflow."""
    
    print("🏥 Healthcare Test Case Generator - Workflow Test")
    print("=" * 60)
    
    # Check environment variables
    bucket_name = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not bucket_name:
        print("❌ Error: GOOGLE_CLOUD_PROJECT environment variable not set")
        return
    
    input_folder = os.getenv("GCS_INPUT_FOLDER", "hackathon-11/Input")
    output_folder = os.getenv("GCS_OUTPUT_FOLDER", "hackathon-11/output")
    
    print(f"📁 Input Folder: {input_folder}")
    print(f"📁 Output Folder: {output_folder}")
    print(f"🪣 Bucket: {bucket_name}")
    print("=" * 60)
    
    # Step 1: List PRD documents
    print("🔍 Step 1: Listing PRD documents...")
    prd_documents = list_prd_documents(bucket_name)
    
    if not prd_documents:
        print("❌ No PRD documents found in input folder")
        print(f"   Please ensure PDF documents are uploaded to: gs://{bucket_name}/{input_folder}")
        return
    
    print(f"✅ Found {len(prd_documents)} PRD documents:")
    for i, doc in enumerate(prd_documents, 1):
        print(f"   {i}. {doc}")
    
    # Step 2: Process PRD documents and generate test cases
    print("\n🔄 Step 2: Processing PRD documents and generating test cases...")
    compliance_standards = ["FDA", "HIPAA", "IEC 62304", "ISO 13485"]
    
    result = process_healthcare_prd_to_test_cases(bucket_name, compliance_standards)
    
    if "error" in result:
        print(f"❌ Error processing PRD documents: {result['error']}")
        return
    
    print("✅ Processing completed successfully!")
    print(f"   📄 PRD documents processed: {result['prd_documents_processed']}")
    print(f"   📋 Total requirements: {result['total_requirements']}")
    print(f"   🧪 Total test cases: {result['total_test_cases']}")
    print(f"   📊 CSV output path: {result['csv_output_path']}")
    print(f"   📋 PDF traceability path: {result['pdf_traceability_path']}")
    
    # Step 3: Show processing results
    print("\n📊 Step 3: Processing Results:")
    for i, pr_result in enumerate(result['processing_results'], 1):
        print(f"   {i}. {pr_result['prd_path']}")
        print(f"      Requirements: {pr_result['requirements_count']}")
        print(f"      Test Cases: {pr_result['test_cases_count']}")
        print(f"      Status: {pr_result['status']}")
    
    print("\n🎉 Workflow test completed successfully!")
    print(f"📁 Test cases CSV available at: {result['csv_output_path']}")
    print(f"📋 Traceability matrix PDF available at: {result['pdf_traceability_path']}")

if __name__ == "__main__":
    test_workflow()
