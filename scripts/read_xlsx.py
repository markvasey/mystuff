import zipfile
import xml.etree.ElementTree as ET
import os
import sys

def get_sheet_names(file_path):
    ns = {'rels': 'http://schemas.openxmlformats.org/package/2006/relationships',
          'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(file_path, 'r') as z:
        with z.open('xl/workbook.xml') as f:
            tree = ET.parse(f)
            sheets = tree.findall('.//main:sheet', ns)
            return [(s.get('name'), s.get('sheetId'), s.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')) for s in sheets]

def read_xlsx(file_path, sheet_rel_id=None):
    ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    
    with zipfile.ZipFile(file_path, 'r') as z:
        # Load shared strings
        shared_strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            with z.open('xl/sharedStrings.xml') as f:
                tree = ET.parse(f)
                for t in tree.findall('.//main:t', ns):
                    shared_strings.append(t.text)

        # Find the sheet file
        sheet_file = f'xl/worksheets/sheet1.xml' # Default
        if sheet_rel_id:
            # Open the workbook relationships to find the path
            with z.open('xl/_rels/workbook.xml.rels') as f:
                tree = ET.parse(f)
                rels = tree.findall('.//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship')
                for rel in rels:
                    if rel.get('Id') == sheet_rel_id:
                        sheet_file = 'xl/' + rel.get('Target')
                        break

        with z.open(sheet_file) as f:
            tree = ET.parse(f)
            rows = tree.findall('.//main:row', ns)
            
            for row in rows:
                cells = row.findall('.//main:c', ns)
                row_data = []
                for cell in cells:
                    value_el = cell.find('main:v', ns)
                    if value_el is not None:
                        val = value_el.text
                        t = cell.get('t')
                        if t == 's': # shared string
                            val = shared_strings[int(val)]
                        row_data.append(val)
                    else:
                        row_data.append("")
                if any(row_data):
                    print("\t".join(row_data))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 read_xlsx.py <file_path> [sheet_rel_id]")
    elif len(sys.argv) == 2:
        sheets = get_sheet_names(sys.argv[1])
        print("Sheets found:")
        for name, sid, rid in sheets:
            print(f"Name: {name}, ID: {rid}")
    else:
        read_xlsx(sys.argv[1], sys.argv[2])
