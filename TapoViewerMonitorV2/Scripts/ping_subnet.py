import subprocess
import concurrent.futures
import xml.etree.ElementTree as ET
import os
import platform

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

def ping_ip(name, ip):
    param = '-n' if platform.system().lower() == 'windows' else '-c'
    command = ['ping', param, '1', '-W', '1', ip]
    try:
        res = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0:
            return name, ip, "UP"
    except Exception:
        pass
    return name, ip, "DOWN"

def main():
    house_name = "2MFC"
    cameras = load_cameras_for_house(house_name)
    if not cameras:
        print(f"Error: No cameras found for house '{house_name}' in cameras.xml")
        return
        
    print(f"Pinging cameras configured for house '{house_name}'...")
    print("-" * 50)
    print(f"{'Camera Name':<20} | {'IP Address':<15} | {'Host Status'}")
    print("-" * 50)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(ping_ip, cam["name"], cam["ip"]) for cam in cameras]
        for fut in concurrent.futures.as_completed(futures):
            name, ip, status = fut.result()
            print(f"{name:<20} | {ip:<15} | {status}")
    print("-" * 50)

if __name__ == "__main__":
    main()
