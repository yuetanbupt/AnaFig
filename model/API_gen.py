import re
import json
import argparse
import os
import base64
import time
import requests
from PIL import Image
import io
from openai import OpenAI

def encode_image_to_base64(image):
    """Encode PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format=image.format or "JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def build_input_sequence(data_entry):
    """
    Construct multimodal input sequence from data entry
    Replaces LaTeX references with corresponding images and captions
    """
    text_content = data_entry['text']
    text_sequence = f"<text>{text_content}<text/>"
    
    # Process label references
    for key, label_value in data_entry.items():
        if not key.startswith('label'):
            continue
            
        num = ''.join(filter(str.isdigit, key))
        figure_key = f"figure{num}"
        caption_key = f"caption{num}"
        figure_path = f"physics1000/{data_entry[figure_key]}.jpg"
        caption = data_entry.get(caption_key, "")
        
        replacement = f"<text/>|{figure_path}|<caption>{caption}<caption/>|<text>"
        text_sequence = text_sequence.replace(r"{" + label_value + "}", replacement)

    # Clean residual LaTeX commands
    text_sequence = re.sub(r"\\(label|[a-zA-Z]+ref|ref|fig)", "", text_sequence)
    text_sequence = text_sequence.replace("||", "|")
    
    # Build final sequence
    sequence = []
    for segment in text_sequence.split("|"):
        if segment.endswith(".jpg"):
            sequence.append({"type": "image", "content": segment})
        elif segment.strip():
            sequence.append({"type": "text", "content": segment.strip()})
                
    return sequence

def get_target_image_order(data_entry, sequence):
    """Determine the order index of the target image"""
    target_image = f"{data_entry['summarize_figure']}.jpg"
    order = 1
    for item in sequence:
        if item['type'] == 'image':
            if target_image in item['content']:
                return order
            order += 1
    return 1

def generate_summary(inputs, order, api_key, api_url, model_name):
    """
    Generate summary using API call
    """
    system_prompt = (
        "Generate a chart summary based on the following input information:\n"
        "1. Charts (figures): May contain multiple subfigures\n"
        "2. Chart titles (captions): Provide overall context\n"
        "3. Text descriptions: Contain relevant background knowledge\n\n"
        
        "Requirements:\n"
        f"1. For multiple images, focus on image #{order} and treat others as supplementary\n"
        "2. Captions and text descriptions should supplement but not replace chart analysis\n"
        "3. Generate a concise English paragraph (<200 words) using UTF-8 characters\n\n"
        
        "Evaluation dimensions:\n"
        "1) Faithfulness: Accuracy to source content\n"
        "2) Completeness: Coverage of key information\n"
        "3) Conciseness: Avoidance of redundancy\n"
        "4) Logicality: Logical coherence\n"
        "5) Analysis: Deep understanding beyond surface data"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": []}
    ]
    
    # Build multimodal input
    for item in inputs:
        if item['type'] == 'text':
            messages[1]["content"].append({"type": "text", "text": item['content']})
        elif item['type'] == 'image':
            try:
                image = Image.open(item['content'])
                base64_image = encode_image_to_base64(image)
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            except Exception as e:
                print(f"Error processing image: {str(e)}")
    
    # API call with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=api_key, base_url=api_url)
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                timeout=180
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API error (attempt {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(10)
    
    print("API call failed after retries")
    return "error"

def main():
    parser = argparse.ArgumentParser(description="Multimodal summarization API")
    parser.add_argument('--dataSet_name', required=True, help='Input dataset filename')
    parser.add_argument('--model_name', required=True, help='Model name for API call')
    parser.add_argument('--api_key', required=True, help='API key for authentication')
    parser.add_argument('--api_link', required=True, help='API endpoint URL')
    args = parser.parse_args()
    
    # Configure paths
    input_path = f"{args.dataSet_name}.json"
    output_dir = "output/summarization_pre"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{args.dataSet_name}_{args.model_name.replace('/', '_')}_gen.json"
    output_path = os.path.join(output_dir, output_file)
    
    # Load dataset
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Process dataset
    error_items = []
    total_items = len(dataset)
    
    for i, (key, entry) in enumerate(dataset.items()):
        print(f"Processing item {i+1}/{total_items}: {key}")
        
        try:
            # Build input sequence
            input_seq = build_input_sequence(entry)
            
            # Determine target image order
            order = get_target_image_order(entry, input_seq)
            
            # Generate summary
            summary = generate_summary(
                inputs=input_seq,
                order=order,
                api_key=args.api_key,
                api_url=args.api_link,
                model_name=args.model_name
            )
            
            # Store results
            entry['summarization_pre'] = summary
            entry['order'] = order
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            entry['summarization_pre'] = "processing_error"
            error_items.append(key)
        
        # Periodic save every 10 items
        if (i + 1) % 10 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"Checkpoint saved: Processed {i+1} items")
    
    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete. Results saved to: {output_path}")
    if error_items:
        print(f"Errors occurred with {len(error_items)} items: {error_items}")

if __name__ == "__main__":
    main()