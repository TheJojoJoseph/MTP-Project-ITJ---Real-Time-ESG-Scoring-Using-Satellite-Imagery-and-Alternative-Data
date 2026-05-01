#!/usr/bin/env python3
"""
Convert Markdown files to PDF using markdown and weasyprint
"""

import sys
import markdown
from pathlib import Path

def convert_markdown_to_html(md_file):
    """Convert markdown to HTML"""
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML with extensions
    html = markdown.markdown(
        md_content,
        extensions=[
            'tables',
            'fenced_code',
            'codehilite',
            'toc',
            'nl2br'
        ]
    )

    # Add CSS styling
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 40px auto;
                padding: 0 20px;
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 5px;
            }}
            h3 {{
                color: #555;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px 12px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            pre code {{
                background-color: transparent;
                padding: 0;
            }}
            ul, ol {{
                margin: 10px 0;
                padding-left: 30px;
            }}
            hr {{
                border: none;
                border-top: 2px solid #ddd;
                margin: 30px 0;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                padding-left: 20px;
                margin: 20px 0;
                color: #666;
                font-style: italic;
            }}
            .checkmark {{
                color: #27ae60;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """

    return styled_html

def main():
    if len(sys.argv) < 2:
        print("Usage: python markdown_to_pdf.py <markdown_file>")
        sys.exit(1)

    md_file = Path(sys.argv[1])

    if not md_file.exists():
        print(f"Error: File {md_file} not found")
        sys.exit(1)

    # Convert to HTML
    print(f"Converting {md_file} to HTML...")
    html_content = convert_markdown_to_html(md_file)

    # Save HTML
    html_file = md_file.with_suffix('.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✓ HTML saved to {html_file}")

    # Try to convert to PDF using wkhtmltopdf via pdfkit
    try:
        import pdfkit
        pdf_file = md_file.with_suffix('.pdf')

        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'enable-local-file-access': None
        }

        print(f"Converting HTML to PDF...")
        pdfkit.from_file(str(html_file), str(pdf_file), options=options)
        print(f"✓ PDF saved to {pdf_file}")

    except Exception as e:
        print(f"Could not create PDF directly: {e}")
        print(f"HTML file saved. You can:")
        print(f"  1. Open {html_file} in browser and print to PDF")
        print(f"  2. Use: wkhtmltopdf {html_file} {md_file.stem}.pdf")

if __name__ == "__main__":
    main()
