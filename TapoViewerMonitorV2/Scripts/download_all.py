import os
import re
import subprocess

test_videos_root = "TestVideos"
yt_dlp_path = "./venv/bin/yt-dlp"

subfolders = [
    "Training_Calibration_Seizures",
    "Training_Calibration_NonSeizures",
    "Evaluation_Seizures",
    "Evaluation_NonSeizures"
]

print("Scanning dataset directories for URL list files...")

for subfolder in subfolders:
    folder_path = os.path.join(test_videos_root, subfolder)
    if not os.path.exists(folder_path):
        continue
        
    # Find any .txt file in this folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    if not txt_files:
        continue
        
    print(f"\nProcessing folder: {subfolder}")
    for txt_file in txt_files:
        txt_path = os.path.join(folder_path, txt_file)
        print(f"  Reading URLs from {txt_file}...")
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        urls = []
        for line in lines:
            line = line.strip()
            if line.startswith("http"):
                urls.append(line)
                
        print(f"  Found {len(urls)} URLs.")
        
        for url in urls:
            # Match standard v=... query
            match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
            if not match:
                # Match youtu.be/... short url
                match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url)
            if not match:
                # Match youtube.com/shorts/... url
                match = re.search(r"shorts/([a-zA-Z0-9_-]{11})", url)
            if not match:
                continue
            video_id = match.group(1)
            
            output_path = os.path.join(folder_path, f"{video_id}.mp4")
            
            # Special case for existing seizure_video.mp4 reference name
            if video_id == "NuGUfUymfFs" and os.path.exists(os.path.join(folder_path, "seizure_video.mp4")):
                print(f"    Using existing seizure_video.mp4 for {video_id}.")
                continue
                
            if os.path.exists(output_path) and os.path.getsize(output_path) > 100000:
                print(f"    Video {video_id} already exists, skipping.")
                continue
                
            print(f"    Downloading {url} to {output_path}...")
            cmd = [
                yt_dlp_path,
                "--no-playlist",
                "-f", "best[ext=mp4]/best",
                "-o", output_path,
                url
            ]
            try:
                subprocess.run(cmd, check=True)
                print(f"    Successfully downloaded {video_id}.")
            except Exception as e:
                print(f"    Failed to download {video_id}: {e}")
