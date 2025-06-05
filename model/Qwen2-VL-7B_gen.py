import argparse
import json
import os
import re
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Increase image pixel limit
Image.MAX_IMAGE_PIXELS = 2300000000

def generate_summary(inputs, model, processor):
    """
    Generate summary based on multimodal inputs using Qwen2-VL model

    Args:
        inputs: List of dictionaries with 'type' and 'content'
        model: Pretrained Qwen2-VL model
        processor: AutoProcessor for the model

    Returns:
        Generated summary text or "error" on failure
    """
    system_prompt = (
        "Generate a chart summary based on,focusing primarily on the first image.: "
        "1. Figures (focus on the specified one), "
        "2. Chart titles/captions, "
        "3. Related text descriptions. "
        f"Focus exclusively on figure 1. "
        "Generate concise English summary (<200 words) in a single paragraph. "
        "Ensure faithfulness, completeness, conciseness, logicality, and analysis depth."
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
            messages[1]["content"].append({"type": "image", "image": img})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    # Process inputs
    model_inputs = processor(
        text=[text],
        images=image_inputs, 
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    # Generate summary
    output_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = output_ids[:, model_inputs.input_ids.shape[1]:]
    return processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )

def extract_digits(s):
    """Extract digits from a string"""
    return ''.join(filter(str.isdigit, s))

def preprocess_input(data, img_dir="images/AnaFig-image/main-images"):
    """
    Preprocess input data into model format

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

    return inputs, rich_text

def main():

    # Prepare paths
    input_path = f"data/Summary-2000.json"
    output_dir = "output/summary_pre"
    os.makedirs(output_dir, exist_ok=True)
    output_name = f"Summary-2000_Qwen2-VL-7B_gen.json"
    output_path = os.path.join(output_dir, output_name)


    # Load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", 
        torch_dtype="float16", 
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    # Process dataset
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    errors = []
    for key, entry in dataset.items():
        try:
            inputs, rich_text = preprocess_input(entry)
            summary = generate_summary(inputs, model, processor)
            entry['summary_pre'] = summary
            print(key,summary)
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            errors.append(key)

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"Processing complete. Saved to {output_path}")
    print(f"Errors: {len(errors)}")
    if errors:
        error_path = os.path.join(output_dir, f"Summary-2000_errors.txt")
        with open(error_path, "w") as f:
            f.write("\n".join(errors))

if __name__ == '__main__':
    main()
