import zipfile
import xml.etree.ElementTree as ET
import sys

def read_ods(file_path):
    with zipfile.ZipFile(file_path, 'r') as z:
        with z.open('content.xml') as f:
            content = f.read()
    
    root = ET.fromstring(content)
    ns = {
        'table': 'urn:oasis:names:tc:opendocument:xmlns:table:1.0',
        'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'
    }
    
    for table in root.findall('.//table:table', ns):
        table_name = table.get('{urn:oasis:names:tc:opendocument:xmlns:table:1.0}name')
        print(f"\nTable: {table_name}")
        for row in table.findall('.//table:table-row', ns):
            row_data = []
            for cell in row.findall('.//table:table-cell', ns):
                text_node = cell.find('.//text:p', ns)
                row_data.append(text_node.text if text_node is not None else "")
            if any(row_data):
                print("\t".join(row_data))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 read_ods.py <path_to_ods_file>")
    else:
        read_ods(sys.argv[1])
