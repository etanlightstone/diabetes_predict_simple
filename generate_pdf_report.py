#!/usr/bin/env python3
"""
Generate PDF from Diabetes Prediction Model Optimization Report
Preserves embedded images and formatting for sharing
"""

import os
import sys
import subprocess
from pathlib import Path

def install_requirements():
    """Install required packages for PDF generation"""
    packages = [
        'markdown2',
        'weasyprint',
        'Pillow'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    return True

def convert_markdown_to_pdf():
    """Convert markdown report to PDF with embedded images"""
    try:
        import markdown2
        import weasyprint
        from PIL import Image
        
        # Define file paths
        markdown_file = "Diabetes_Prediction_Model_Optimization_Report.md"
        pdf_file = "Diabetes_Prediction_Model_Optimization_Report.pdf"
        
        # Check if markdown file exists
        if not os.path.exists(markdown_file):
            print(f"❌ Markdown file not found: {markdown_file}")
            return False
            
        # Read markdown content
        print(f"📖 Reading markdown file: {markdown_file}")
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Convert markdown to HTML with extras for better formatting
        print("🔄 Converting markdown to HTML...")
        html_content = markdown2.markdown(
            markdown_content, 
            extras=[
                'fenced-code-blocks',
                'tables',
                'task_list',
                'strike',
                'break-on-newline'
            ]
        )
        
        # Create complete HTML document with styling
        html_document = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Diabetes Prediction Model Optimization Report</title>
    <style>
        @page {{
            margin: 1in;
            size: letter;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: none;
            margin: 0;
            padding: 0;
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            page-break-after: avoid;
        }}
        
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 5px;
            margin-top: 30px;
            page-break-after: avoid;
        }}
        
        h3 {{
            color: #2c3e50;
            margin-top: 25px;
            page-break-after: avoid;
        }}
        
        h4 {{
            color: #34495e;
            margin-top: 20px;
            page-break-after: avoid;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 0.9em;
            page-break-inside: avoid;
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            page-break-inside: avoid;
        }}
        
        code {{
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.9em;
        }}
        
        pre {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            overflow-x: auto;
            margin: 15px 0;
            page-break-inside: avoid;
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 15px 0;
            padding-left: 15px;
            color: #555;
            font-style: italic;
        }}
        
        .page-break {{
            page-break-before: always;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #bdc3c7;
            margin: 30px 0;
        }}
        
        ul, ol {{
            margin: 10px 0;
            padding-left: 30px;
        }}
        
        li {{
            margin: 5px 0;
        }}
        
        .emoji {{
            font-size: 1.2em;
        }}
        
        /* Ensure proper spacing for achievement bullets */
        li strong {{
            color: #2c3e50;
        }}
        
        /* Style for the table ranking */
        td:first-child {{
            font-weight: bold;
        }}
        
        /* Better spacing for sections */
        .section {{
            margin-bottom: 40px;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
        """
        
        # Check if image files exist and warn if missing
        image_files = [
            "diabetes_data_overview.png",
            "feature_distributions_by_class.png", 
            "feature_interactions_analysis.png"
        ]
        
        missing_images = []
        for img_file in image_files:
            if not os.path.exists(img_file):
                missing_images.append(img_file)
                print(f"⚠️  Warning: Image file not found: {img_file}")
        
        if missing_images:
            print(f"📝 Note: {len(missing_images)} image(s) missing but proceeding with PDF generation")
        
        # Convert HTML to PDF
        print("🎯 Converting HTML to PDF...")
        weasyprint.HTML(string=html_document, base_url='.').write_pdf(pdf_file)
        
        # Check if PDF was created successfully
        if os.path.exists(pdf_file):
            file_size = os.path.getsize(pdf_file) / (1024 * 1024)  # Size in MB
            print(f"✅ PDF generated successfully: {pdf_file}")
            print(f"📄 File size: {file_size:.2f} MB")
            
            # List all files in current directory for verification
            print("\n📁 Files in current directory:")
            for file in sorted(os.listdir('.')):
                if os.path.isfile(file):
                    size = os.path.getsize(file) / 1024  # Size in KB
                    print(f"   {file} ({size:.1f} KB)")
            
            return True
        else:
            print("❌ Failed to generate PDF")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required packages are installed")
        return False
    except Exception as e:
        print(f"❌ Error during PDF generation: {e}")
        return False

def main():
    """Main function to orchestrate PDF generation"""
    print("🚀 Starting PDF generation for Diabetes Prediction Model Report")
    print("=" * 60)
    
    # Install required packages
    print("\n📦 Installing required packages...")
    if not install_requirements():
        print("❌ Failed to install required packages")
        sys.exit(1)
    
    # Convert markdown to PDF
    print("\n🔄 Converting markdown to PDF...")
    if convert_markdown_to_pdf():
        print("\n🎉 PDF generation completed successfully!")
        print("📧 The PDF is ready to share with friends!")
    else:
        print("\n❌ PDF generation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
