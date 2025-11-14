#!/usr/bin/env python3
"""
Script to extract Excel documentation from all e2e tests and save as individual CSV files.

This script:
1. Searches through all folders in tests/e2e_tests/
2. Finds all .xlsx documentation files
3. Reads each sheet from the Excel files
4. Saves each sheet as a separate CSV file in a unified folder
5. Preserves original folder structure in the output folder
"""

import os
import pandas as pd
import glob
from pathlib import Path

def extract_e2e_documentation():
    """
    Extract all Excel documentation from e2e tests and convert to CSV files.
    """
    # Define paths
    e2e_tests_dir = "tests/e2e_tests"
    output_dir = "e2e_documentation_csvs"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔍 Searching for Excel documentation files in {e2e_tests_dir}...")
    
    # Find all .xlsx files in e2e_tests directory and subdirectories
    xlsx_files = glob.glob(os.path.join(e2e_tests_dir, "**", "*.xlsx"), recursive=True)
    
    if not xlsx_files:
        print("❌ No Excel files found!")
        return
    
    print(f"📊 Found {len(xlsx_files)} Excel files:")
    for file in xlsx_files:
        print(f"   - {file}")
    
    total_sheets = 0
    
    # Process each Excel file
    for xlsx_file in xlsx_files:
        print(f"\n📖 Processing: {xlsx_file}")
        
        try:
            # Get relative path from e2e_tests_dir to preserve structure
            rel_path = os.path.relpath(xlsx_file, e2e_tests_dir)
            rel_dir = os.path.dirname(rel_path)
            filename = os.path.basename(xlsx_file)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Create corresponding directory structure in output
            if rel_dir:
                output_subdir = os.path.join(output_dir, rel_dir)
                os.makedirs(output_subdir, exist_ok=True)
            else:
                output_subdir = output_dir
            
            # Read all sheets from the Excel file
            excel_file = pd.ExcelFile(xlsx_file)
            sheet_names = excel_file.sheet_names
            
            print(f"   📋 Found {len(sheet_names)} sheets: {sheet_names}")
            
            # Process each sheet
            for sheet_name in sheet_names:
                try:
                    # Read the sheet
                    df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
                    
                    # Create CSV filename
                    # Replace spaces and special characters with underscores
                    safe_sheet_name = sheet_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                    csv_filename = f"{name_without_ext}_{safe_sheet_name}.csv"
                    csv_path = os.path.join(output_subdir, csv_filename)
                    
                    # Save as CSV
                    df.to_csv(csv_path, index=False)
                    
                    print(f"   ✅ Saved: {csv_filename} ({len(df)} rows)")
                    total_sheets += 1
                    
                except Exception as e:
                    print(f"   ❌ Error processing sheet '{sheet_name}': {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error processing file {xlsx_file}: {e}")
            continue
    
    print(f"\n🎉 Extraction complete!")
    print(f"📊 Total sheets converted: {total_sheets}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Original files preserved in: {e2e_tests_dir}")

def main():
    """
    Main function to run the extraction.
    """
    print("=" * 60)
    print("🔧 E2E Documentation Extractor")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("tests/e2e_tests"):
        print("❌ Error: tests/e2e_tests directory not found!")
        print("   Please run this script from the corgidrp root directory.")
        return
    
    extract_e2e_documentation()

if __name__ == "__main__":
    main()


