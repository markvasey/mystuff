from datasets import load_dataset

# 1. Stream the dataset (loads instances one by one without a huge download)
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

# 2. Extract exactly 5000 stories
sample_stories = [item["text"] for item in dataset.take(5000)]

# 3. Save them out as a single local file for inspection
with open("5000_children_stories.txt", "w", encoding="utf-8") as f:
    for i, story in enumerate(sample_stories, 1):
        f.write(f"--- Story {i} ---\n{story.strip()}\n\n")
