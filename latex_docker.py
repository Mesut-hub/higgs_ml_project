import docker
import os
import subprocess

client = docker.from_env()
project_dir = os.path.abspath('D:\\higgs_ml_project')

# 1. Convert to HTML
subprocess.run([
    'jupyter', 'nbconvert', 
    '--to', 'html',
    'notebooks/final_report.ipynb',
    '--output-dir', 'reports/'
])

# 2. Fix network issue: Remove any wkhtmltopdf references from notebook
html_path = os.path.join(project_dir, 'reports', 'final_report.html')
with open(html_path, 'r', encoding='utf-8') as f:
    content = f.read()
    
# Remove problematic references
content = content.replace('http://wkhtmltopdf/', '')
content = content.replace('wkhtmltopdf', '')

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(content)

# 3. Convert HTML to PDF using Docker
try:
    subprocess.run([
        'docker', 'run', '--rm',
        '-v', f'{project_dir}:/data',
        'madnight/docker-alpine-wkhtmltopdf',
        'wkhtmltopdf',
        '--load-error-handling', 'ignore',
        '--enable-local-file-access',
        '/data/reports/final_report.html',
        '/data/reports/final_report.pdf'
    ], check=True)
    
    print("\nPDF generated successfully at D:\\higgs_ml_project\\reports\\final_report.pdf")
    
except subprocess.CalledProcessError as e:
    print(f"PDF generation failed: {e}")
    
# 4. (Optional) LaTeX conversion if needed - with proper command syntax
try:
    # Generate LaTeX first
    subprocess.run([
        'jupyter', 'nbconvert', 
        '--to', 'latex',
        'notebooks/final_report.ipynb',
        '--output-dir', 'reports/'
    ])
    
    # Run LaTeX compilation
    container = client.containers.run(
        "blang/latex",
        command="sh -c 'tlmgr update --self && tlmgr install collection-fontsrecommended && pdflatex -interaction=nonstopmode final_report.tex'",
        volumes={project_dir: {'bind': '/data', 'mode': 'rw'}},
        working_dir='/data/reports',
        detach=True
    )
    
    # Stream logs
    for line in container.logs(stream=True):
        print(line.decode('utf-8'), end='')
    
    # Wait for completion
    exit_status = container.wait()
    print(f"\nLaTeX container exited with status: {exit_status}")
    
finally:
    if 'container' in locals():
        container.remove(force=True)