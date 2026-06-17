from datasets import load_dataset

print("Streaming roneneldan/TinyStories dataset from Hugging Face...")
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

target_stories = 200000
print(f"Extracting exactly {target_stories} stories...")
sample_stories = [item["text"] for item in dataset.take(target_stories)]

output_file = "children_stories_200k.txt"
print(f"Saving stories to {output_file}...")
with open(output_file, "w", encoding="utf-8") as f:
    for i, story in enumerate(sample_stories, 1):
        f.write(f"--- Story {i} ---\n{story.strip()}\n\n")

print(f"Extraction completed successfully! File saved to {output_file}.")
