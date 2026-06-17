from datasets import load_dataset
import os

print("Streaming roneneldan/TinyStories dataset from Hugging Face...")
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

target_stories = 50000
print(f"Extracting exactly {target_stories} stories...")
sample_stories = [item["text"] for item in dataset.take(target_stories)]

output_file = "children_stories.txt"
print(f"Saving stories to {output_file}...")
with open(output_file, "w", encoding="utf-8") as f:
    for i, story in enumerate(sample_stories, 1):
        f.write(f"--- Story {i} ---\n{story.strip()}\n\n")

# Clean up the old 5000 stories file if it exists to avoid double loading
old_file = "5000_children_stories.txt"
if os.path.exists(old_file):
    os.remove(old_file)
    print(f"Removed old {old_file} file.")

print("Download and extraction completed successfully!")
