import socket
import concurrent.futures
import xml.etree.ElementTree as ET
import os

def load_cameras_for_house(house_name):
    cameras = []
    xml_path = os.path.join("src", "main", "resources", "cameras.xml")
    if not os.path.exists(xml_path):
        xml_path = os.path.join(os.path.dirname(__file__), "..", "src", "main", "resources", "cameras.xml")
        
    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for house in root.findall("House"):
                if house.get("name") == house_name:
                    for cam in house.findall("Camera"):
                        cameras.append({
                            "name": cam.get("name"),
                            "ip": cam.get("ip")
                        })
        except Exception as e:
            print(f"Error parsing cameras.xml: {e}")
    return cameras

def scan_ip(name, ip):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0) # 1s timeout
        result = s.connect_ex((ip, 554))
        s.close()
        if result == 0:
            return name, ip, "PORT 554 OPEN"
    except Exception:
        pass
    return name, ip, "PORT 554 CLOSED"

def main():
    house_name = "2MFC"
    cameras = load_cameras_for_house(house_name)
    if not cameras:
        print(f"Error: No cameras found for house '{house_name}' in cameras.xml")
        return
        
    print(f"Scanning RTSP port 554 for house '{house_name}'...")
    print("-" * 55)
    print(f"{'Camera Name':<20} | {'IP Address':<15} | {'Port Status'}")
    print("-" * 55)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(scan_ip, cam["name"], cam["ip"]) for cam in cameras]
        for fut in concurrent.futures.as_completed(futures):
            name, ip, status = fut.result()
            print(f"{name:<20} | {ip:<15} | {status}")
    print("-" * 55)

if __name__ == "__main__":
    main()
