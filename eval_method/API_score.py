import argparse
import json
import os
import re
import time
import base64
import io
from PIL import Image
from openai import OpenAI
# Increase image pixel limit
Image.MAX_IMAGE_PIXELS = 2300000000

def encode_image_to_base64(image):
    """Encode PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format=image.format or 'JPEG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_score(inputs,api_key, base_url, model_name):
    """
    Generate summary score using multimodal API

    Args:
        inputs: List of input items (text/image)
        api_key: API secret key
        base_url: API base URL
        model_name: Model name to use

    Returns:
        Generated score text or "error" on failure
    """
    system_prompt = (
        "Evaluate the summary based on these criteria with reference summary:\n"
        "1. Faithfulness: Strict adherence to figure and supplementary content (5-point scale)\n"
        "2. Completeness: Coverage of all key information (5-point scale)\n"
        "3. Conciseness: Brevity and clarity (5-point scale)\n"
        "4. Logicality: Logical coherence and expert knowledge (5-point scale)\n"
        "5. Analysis: Depth of understanding and interpretation (5-point scale)\n\n"
        "Output format: 'Faithfulness (X/5); Completeness (X/5); Conciseness (X/5); Logicality (X/5); Analysis (X/5)'"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": []}
    ]

    for item in inputs:
        if item['type'] == 'text':
            messages[1]["content"].append({"type": "text", "text": item['content']})
        elif item['type'] == 'image':
            img = Image.open(item['content']).resize((224, 224))
            base64_img = encode_image_to_base64(img)
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            timeout=180
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API error: {str(e)}")
        time.sleep(10)
        return "error"

def extract_digits(s):
    """Extract digits from a string"""
    return ''.join(filter(str.isdigit, s))

def preprocess_input(data, img_dir="images/AnaFig-image/main-images"):
    """
    Preprocess input data for scoring

    Args:
        data: Dictionary containing text, figures, and captions
        img_dir: Directory containing images

    Returns:
        tuple: (processed_inputs, target_figure_index, rich_text)
    """
    text = data['context']
    rich_text = f"<text>{text}<text/>"

    # Process labeled references
    for key, value in data.items():
        if "label" in key:
            num = extract_digits(key)
            fig_key = f"figure{num}"
            cap_key = f"caption{num}"

            img_path = f"{img_dir}/{data[fig_key]}.jpg"
            caption = data.get(cap_key, "")

            # Replace references with structured format
            rich_text = rich_text.replace(
                f"{{{value}}}",
                f"<text/>|<figure>{img_path}<figure/>|<caption>{caption}<caption/><text>"
            )

    # Clean special patterns
    clean_pattern = r"(\\label|\\[a-zA-Z]+ref|\\ref|\\fig)"
    rich_text = re.sub(clean_pattern, "", rich_text).replace("<text><text/>", "")

    # Build input sequence
    segments = rich_text.split("|")
    inputs = []

    for seg in segments:
        if ".jpg" in seg:
            seg = seg.replace("<figure>","")
            seg = seg.replace("<figure/>","")
            inputs.append({"type": "image", "content": seg})
        elif seg.strip():
            inputs.append({"type": "text", "content": seg})
    
    inputs.append({
        "type": "text",
        "content": f"<reference summary>{data['summary']}<reference summary/>"
        })
    # Add summary for evaluation
    if 'summary' in data:
        inputs.append({
            "type": "text",
            "content": f"<summary>{data['summary_pre']}<summary/>"
        })
    return inputs, rich_text

def main():
    parser = argparse.ArgumentParser(description='Summary Scoring with API')
    parser.add_argument('--file_name', required=True, help='Name of the input data file')
    parser.add_argument('--model_name', required=True, help='API model name to use')
    parser.add_argument('--api_key', required=True, help='API secret key')
    parser.add_argument('--api_link', required=True, help='API base URL')
    args = parser.parse_args()

    # Prepare paths
    input_path = f"{args.file_name}"
    output_dir = "output/score"
    os.makedirs(output_dir, exist_ok=True)
    output_name = f"{args.file_name.replace('.json','')}_{args.model_name}_score.json"
    #output_path = os.path.join(output_dir, output_name)
    output_path = output_name.replace(r"/summary_pre/","/score/")
    # Process dataset
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    errors = []
    for key, entry in dataset.items():
        # Skip already processed entries
        if 'score' in entry and entry.get('score') != "error!":
            continue
        # Skip entries without required summary
        if 'summary_pre' not in entry:
            continue

        try:
            inputs, rich_text = preprocess_input(entry)

            score = generate_score(
                inputs,
                api_key=args.api_key,
                base_url=args.api_link,
                model_name=args.model_name
            )
            entry['score'] = score
            print(key,score)
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            errors.append(key)

    # Save final results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"Processing complete. Saved to {output_path}")
    print(f"Errors: {len(errors)}")
    if errors:
        error_path = os.path.join(output_dir, f"{args.model_name}_errors.txt")
        with open(error_path, "w") as f:
            f.write("\n".join(errors))

if __name__ == '__main__':
    main()
