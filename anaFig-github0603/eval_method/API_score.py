import re
import json
import argparse
import os
import base64
import time
from PIL import Image
import io
from openai import OpenAI

# Set maximum image pixels to prevent decompression bomb errors
Image.MAX_IMAGE_PIXELS = 2300000000

def encode_image_to_base64(image):
    """Encode PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format=image.format or "JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def build_input_sequence(data_entry, base_image_path="../Figure-CS/"):
    """
    Construct multimodal input sequence from data entry
    Replaces LaTeX references with corresponding images and captions
    """
    text_content = data_entry['paragraphs']
    text_sequence = f"<text>{text_content}<text/>"
    
    # Process label references
    for key, label_value in data_entry.items():
        if not key.startswith('label'):
            continue
            
        num = ''.join(filter(str.isdigit, key))
        figure_key = f"figure{num}"
        caption_key = f"caption{num}"
        figure_path = f"{base_image_path}{data_entry[figure_key]}.jpg"
        caption = data_entry.get(caption_key, "")
        
        replacement = f"<text/>|{figure_path}|<caption>{caption}<caption/>|<text>"
        text_sequence = text_sequence.replace(r"{" + label_value + "}", replacement)

    # Clean residual LaTeX commands
    text_sequence = re.sub(r"\\(label|[a-zA-Z]+ref|ref|fig)", "", text_sequence)
    text_sequence = text_sequence.replace("||", "|")
    
    # Build final sequence
    segments = text_sequence.split("|")
    
    # Add pre-summarization if available
    if 'summarization_pre' in data_entry:
        segments.append(
            f"<pre_summarization_pre>{data_entry['summarization_pre']}<pre_summarization/>"
        )
    
    sequence = []
    target_order = None
    image_counter = 1
    
    for segment in segments:
        if segment.endswith(".jpg"):
            # Determine if this is the target image
            if data_entry['summarize_figure'] + ".jpg" in segment:
                target_order = image_counter
            image_counter += 1
            sequence.append({"type": "image", "content": segment})
        elif segment.strip():
            sequence.append({"type": "text", "content": segment.strip()})
                
    return sequence, target_order

def generate_score(inputs, target_order, api_key, api_url, model_name):
    """
    Generate evaluation score using API call
    """
    system_prompt = (
        "Evaluate the pre_summarization based on the following input:\n"
        "1. Figures: May contain multiple subfigures\n"
        "2. Captions: Provide overall context\n"
        "3. Text descriptions: Contain background knowledge\n"
        "4. Target summary: Reference for comparison\n"
        "5. Pre_summarization: Model-generated summary focusing on the first image\n\n"
        
        "Scoring Requirements:\n"
        "1. Score based strictly on correspondence with figures and supplementary information\n"
        "2. Evaluate independently across 5 dimensions\n"
        "3. Output format: 'Faithfulness (4/5); Completeness (4/5); Conciseness (5/5); "
        "Logicality (4/5); Information Analysis (4/5)'\n\n"
        
        "Scoring Dimensions:\n"
        "1) Faithfulness: Accuracy to source content (5-point scale)\n"
        "2) Completeness: Coverage of key information (5-point scale)\n"
        "3) Conciseness: Avoidance of redundancy (5-point scale)\n"
        "4) Logicality: Logical coherence (5-point scale)\n"
        "5) Information Analysis: Depth of understanding (5-point scale)"
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
                image = image.resize((224, 224))
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
    return "error!"

def main():
    parser = argparse.ArgumentParser(description="Multimodal summary evaluation")
    parser.add_argument('--file_name', required=True, help='Input data filename')
    parser.add_argument('--model_name', required=True, help='Model name for evaluation')
    parser.add_argument('--api_key', required=True, help='API key for authentication')
    parser.add_argument('--api_link', required=True, help='API endpoint URL')
    args = parser.parse_args()
    
    # Configure paths
    input_path = f"{args.file_name}.json"
    output_dir = "output/score"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename with sanitized model name
    sanitized_model = args.model_name.replace('/', '_')
    output_file = f"{os.path.basename(args.file_name)}-{sanitized_model}_score.json"
    output_path = os.path.join(output_dir, output_file)
    
    # Load dataset
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Process dataset
    processed_count = 0
    total_items = len(dataset)
    
    for key, entry in dataset.items():
        # Skip entries that already have valid scores
        if 'score' in entry and entry['score'] != "error!":
            continue
            
        print(f"Processing item {processed_count+1}/{total_items}: {key}")
        
        try:
            # Build input sequence
            input_seq, target_order = build_input_sequence(entry)
            
            # Generate score
            score = generate_score(
                inputs=input_seq,
                target_order=target_order,
                api_key=args.api_key,
                api_url=args.api_link,
                model_name=args.model_name
            )
            
            # Store results
            entry['score'] = score
            processed_count += 1
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            entry['score'] = "processing_error"
        
        # Periodic save every 10 items
        if processed_count % 10 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"Checkpoint saved: Processed {processed_count} items")
    
    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation complete. Results saved to: {output_path}")
    print(f"Processed {processed_count} new evaluations")

if __name__ == "__main__":
    main()