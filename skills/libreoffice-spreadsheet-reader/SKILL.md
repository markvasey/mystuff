---
name: libreoffice-spreadsheet-reader
description: "Read LibreOffice (.ods) spreadsheets using a standalone Python script without external dependencies like pandas or odfpy."
---

# LibreOffice Spreadsheet Reader

This skill enables Gemini CLI to read data from LibreOffice Calc (.ods) files by parsing their internal XML structure directly using Python's standard library.

## Why use this skill?
- No external Python libraries (`pandas`, `odfpy`) required.
- **Strictly Read-Only:** The script only extracts data and does not modify the source file.

## Usage
1.  **Locate Script:** The core extraction logic is in the `scripts/read_ods.py` file within this skill.
2.  **Execute:** Run the script with Python 3, passing the path to your `.ods` file.

## Script Template
```python
import zipfile
import xml.etree.ElementTree as ET

def read_ods(file_path):
    with zipfile.ZipFile(file_path, 'r') as z:
        with z.open('content.xml') as f:
            content = f.read()
    root = ET.fromstring(content)
    ns = {'table': 'urn:oasis:names:tc:opendocument:xmlns:table:1.0',
          'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}
    
    for table in root.findall('.//table:table', ns):
        table_name = table.get('{urn:oasis:names:tc:opendocument:xmlns:table:1.0}name')
        print(f"\nTable: {table_name}")
        for row in table.findall('.//table:table-row', ns):
            row_data = [ (cell.find('.//text:p', ns).text or "") if cell.find('.//text:p', ns) is not None else "" 
                         for cell in row.findall('.//table:table-cell', ns) ]
            if any(row_data): print("\t".join(row_data))
```

## Integration
- Can be used to ingest data for further processing (e.g., converting to JSON, CSV, or a SQL database).
- Always ensure the source file path is correctly handled.
