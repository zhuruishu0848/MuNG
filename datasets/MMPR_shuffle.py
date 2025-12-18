import json
import os
import random
from tqdm import tqdm
random.seed(1299)

def convert_mmpr_to_qwen(mmpr_item, root_path, dataset_name, item_counter):
    if isinstance(mmpr_item["image"], list):
        image_tags = []
        for i, img_path in enumerate(mmpr_item["image"]):
            clean_path = os.path.join(root_path, img_path)
            image_tags.append(clean_path)
        clean_path = image_tags
    else:
        img_path = mmpr_item["image"]
        clean_path = os.path.join(root_path, img_path)

    qwen_item = {
        "image":clean_path,
        "question":mmpr_item['question'],
        "answer":mmpr_item['chosen']
    }
    
    return qwen_item

def process_dataset(meta_entry, dataset_name):
    input_path = os.path.join("datasets", meta_entry["annotation"])
    root_path = os.path.join("datasets", meta_entry["root"])

    qwen_data = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        
        if meta_entry.get("repeat_time", 1) < 1:
            lines = random.sample(lines, k=int(len(lines) * meta_entry["repeat_time"]))*10
        elif meta_entry.get("repeat_time", 1) > 1:
            lines = lines * int(meta_entry["repeat_time"])*10
            
        item_counter = 0
        for line in tqdm(lines, desc=f"Processing {dataset_name}"):
            try:
                mmpr_item = json.loads(line)
                qwen_item = convert_mmpr_to_qwen(mmpr_item, root_path, dataset_name,item_counter)
                qwen_data.append(qwen_item)
                item_counter += 1
            except json.JSONDecodeError as e:
                print(f"Error parsing line in {dataset_name}: {e}")
                continue
    
    print(f"Processed {len(qwen_data)} items from {dataset_name}")
    return qwen_data

def main(meta_path, output_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    combined_data = []
    for dataset_name, dataset_meta in meta.items():
        dataset_data = process_dataset(dataset_meta, dataset_name)
        combined_data.extend(dataset_data)
    
    random.shuffle(combined_data) 

    with open(output_path, 'w') as f:
        for item in combined_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Conversion complete! Saved {len(combined_data)} items to {output_path}")

if __name__ == "__main__":
    meta_path = "datasets/MMPR-v1.1/meta.json" 
    output_path = "datasets/MMPR-v1.1/shuff_mmpr.jsonl"
    
    main(meta_path, output_path)