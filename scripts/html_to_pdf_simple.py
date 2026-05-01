#!/usr/bin/env python3
"""
Simple HTML to PDF converter using Chrome headless
"""

import subprocess
import sys
from pathlib import Path

def convert_html_to_pdf(html_file, pdf_file):
    """Convert HTML to PDF using Chrome headless mode"""

    # Check if Chrome is available
    chrome_paths = [
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        '/Applications/Chromium.app/Contents/MacOS/Chromium',
        '/usr/bin/google-chrome',
        '/usr/bin/chromium'
    ]

    chrome_path = None
    for path in chrome_paths:
        if Path(path).exists():
            chrome_path = path
            break

    if not chrome_path:
        print("Error: Chrome/Chromium not found.")
        print("Please install Google Chrome or use browser print (Cmd+P → Save as PDF)")
        return False

    # Convert using Chrome headless
    cmd = [
        chrome_path,
        '--headless',
        '--disable-gpu',
        '--print-to-pdf=' + str(pdf_file),
        '--no-pdf-header-footer',
        'file://' + str(Path(html_file).absolute())
    ]

    try:
        print(f"Converting {html_file} to {pdf_file}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if Path(pdf_file).exists():
            print(f"✓ Successfully created {pdf_file}")
            return True
        else:
            print(f"Error: PDF not created")
            if result.stderr:
                print(f"  {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("Error: Conversion timed out")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    # Convert both HTML files
    files = [
        ('EXPERIMENTAL_FINDINGS.html', 'EXPERIMENTAL_FINDINGS.pdf'),
        ('PROJECT_COMPLETION_SUMMARY.html', 'PROJECT_COMPLETION_SUMMARY.pdf')
    ]

    success_count = 0
    for html_file, pdf_file in files:
        if Path(html_file).exists():
            if convert_html_to_pdf(html_file, pdf_file):
                success_count += 1
        else:
            print(f"Warning: {html_file} not found, skipping...")

    print(f"\n{'='*60}")
    print(f"Converted {success_count}/{len(files)} files successfully")
    print(f"{'='*60}")

    if success_count < len(files):
        print("\nAlternative method:")
        print("  1. Open the HTML file in any browser")
        print("  2. Press Cmd+P (or File → Print)")
        print("  3. Select 'Save as PDF' from the PDF dropdown")

if __name__ == "__main__":
    main()
