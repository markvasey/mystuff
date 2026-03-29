import zipfile
import xml.etree.ElementTree as ET

def read_ods(file_path):
    # ODS files are ZIP archives containing content.xml
    with zipfile.ZipFile(file_path, 'r') as z:
        with z.open('content.xml') as f:
            content = f.read()
    
    root = ET.fromstring(content)
    
    # Define namespaces
    ns = {
        'table': 'urn:oasis:names:tc:opendocument:xmlns:table:1.0',
        'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'
    }
    
    # Extract data
    data = []
    for table in root.findall('.//table:table', ns):
        table_name = table.get('{urn:oasis:names:tc:opendocument:xmlns:table:1.0}name')
        print(f"\nTable: {table_name}")
        
        for row in table.findall('.//table:table-row', ns):
            row_data = []
            for cell in row.findall('.//table:table-cell', ns):
                # Get text from the cell
                text_node = cell.find('.//text:p', ns)
                if text_node is not None:
                    row_data.append(text_node.text or "")
                else:
                    row_data.append("")
            
            # Print only non-empty rows
            if any(row_data):
                print("\t".join(row_data))

if __name__ == "__main__":
    try:
        read_ods("Data/TeamPeople.ods")
    except Exception as e:
        print(f"Error: {e}")
