import subprocess
import concurrent.futures
import xml.etree.ElementTree as ET
import os

def load_secrets():
    username = ""
    password = ""
    # Try local resources directory relative to script run location (project root)
    secret_path = os.path.join("src", "main", "resources", "secret.txt")
    if not os.path.exists(secret_path):
        # Fallback to current directory or script's parent resources
        secret_path = os.path.join(os.path.dirname(__file__), "..", "src", "main", "resources", "secret.txt")
        
    if os.path.exists(secret_path):
        with open(secret_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, val = line.strip().split("=", 1)
                    if key.strip() == "username":
                        username = val.strip()
                    elif key.strip() == "password":
                        password = val.strip()
    return username, password

def load_cameras():
    cameras_dict = {}
    xml_path = os.path.join("src", "main", "resources", "cameras.xml")
    if not os.path.exists(xml_path):
        xml_path = os.path.join(os.path.dirname(__file__), "..", "src", "main", "resources", "cameras.xml")
        
    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for house in root.findall("House"):
                house_name = house.get("name")
                cameras_dict[house_name] = []
                for cam in house.findall("Camera"):
                    cameras_dict[house_name].append({
                        "name": cam.get("name"),
                        "ip": cam.get("ip")
                    })
        except Exception as e:
            print(f"Error parsing cameras.xml: {e}")
    return cameras_dict

def check_rtsp(name, ip, username, password):
    url = f"rtsp://{username}:{password}@{ip}:554/stream1"
    cmd = [
        "ffmpeg", "-rtsp_transport", "tcp", 
        "-stimeout", "3000000",
        "-i", url, 
        "-t", "1", 
        "-f", "null", "-"
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        if res.returncode == 0:
            return name, ip, "SUCCESS"
        else:
            err_log = res.stderr
            if "401 Unauthorized" in err_log:
                return name, ip, "FAILED: 401 Unauthorized"
            elif "400 Bad Request" in err_log:
                return name, ip, "FAILED: 400 Bad Request"
            elif "Connection refused" in err_log:
                return name, ip, "FAILED: Connection Refused"
            elif "Connection timed out" in err_log:
                return name, ip, "FAILED: Timeout"
            elif "No route to host" in err_log:
                return name, ip, "FAILED: No Route to Host"
            else:
                lines = [line.strip() for line in err_log.split("\n") if line.strip()]
                last_err = lines[-1] if lines else "Unknown Error"
                return name, ip, f"FAILED: {last_err}"
    except subprocess.TimeoutExpired:
        return name, ip, "FAILED: Command Timeout"
    except Exception as e:
        return name, ip, f"FAILED: {str(e)}"

def main():
    username, password = load_secrets()
    if not username or not password:
        print("Error: Could not load credentials from secret.txt")
        return
        
    cameras_by_house = load_cameras()
    if not cameras_by_house:
        print("Error: Could not load any cameras from cameras.xml")
        return

    print("==============================================================")
    print("                RTSP CAMERA STREAM DIAGNOSTIC                 ")
    print("==============================================================")
    
    for house, cameras in cameras_by_house.items():
        print(f"\n🏠 House: {house}")
        print("-" * 62)
        print(f"{'Camera Name':<20} | {'IP Address':<15} | {'RTSP Stream Status'}")
        print("-" * 62)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(check_rtsp, cam["name"], cam["ip"], username, password) for cam in cameras]
            for fut in concurrent.futures.as_completed(futures):
                name, ip, status = fut.result()
                print(f"{name:<20} | {ip:<15} | {status}")
        print("-" * 62)

if __name__ == "__main__":
    main()
