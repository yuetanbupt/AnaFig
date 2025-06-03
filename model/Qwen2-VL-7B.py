import os
import re
import json
import argparse
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Set maximum image pixels to prevent decompression bomb errors
Image.MAX_IMAGE_PIXELS = 2300000000

def build_input_sequence(data_entry):
    """
    Process input data to build a multimodal input sequence
    Replaces LaTeX references with corresponding images and captions
    """
    text_content = data_entry['text']
    text_sequence = f"<text>{text_content}<text/>"
    
    # Process all label references in the input
    for key, label_value in data_entry.items():
        if not key.startswith('label'):
            continue
            
        num = ''.join(filter(str.isdigit, key))
        figure_key = f"figure{num}"
        caption_key = f"caption{num}"
        figure_path = f"../img/{data_entry[figure_key]}.jpg"
        
        caption = data_entry.get(caption_key, "")
        replacement = f"<text/>|{figure_path}|<caption>{caption}<caption/>|<text>"
        text_sequence = text_sequence.replace(r"{" + label_value + "}", replacement)

    # Clean residual LaTeX commands
    text_sequence = re.sub(r"\\(label|[a-zA-Z]+ref|ref|fig)", "", text_sequence)
    text_sequence = text_sequence.replace("||", "|")
    
    # Build final input sequence
    sequence = []
    for segment in text_sequence.split("|"):
        if segment.endswith(".jpg"):
            sequence.append({"type": "image", "content": segment})
        elif segment.strip():
            clean_segment = re.sub(r'<[^>]*>', '', segment)
            if clean_segment.strip():
                sequence.append({"type": "text", "content": clean_segment})
                
    return sequence

def generate_summary(input_sequence, model, processor):
    """
    Generate summary using Qwen-VL model with multimodal input
    """
    messages = [
        {
            "role": "system",
            "content": (
                "1. Analyze charts/figures (may contain subfigures), focusing primarily on the first image.\n"
                "2. Use chart title (caption) for context.\n"
                "3. Incorporate relevant text descriptions.\n"
                "4. Generate concise English summary (<200 words) focusing on:\n"
                "   - Faithfulness: Stick to source content\n"
                "   - Comprehension: Accurate interpretation\n"
                "   - Comprehensiveness: Cover key points\n"
                "   - Conciseness: Avoid redundancy\n"
                "   - Logical Coherence: Maintain logical flow\n"
                "5. Avoid special formatting ($, \\ce{}, etc.), use standard UTF-8 characters."
            )
        },
        {"role": "user", "content": []}
    ]

    # Build multimodal input
    for item in input_sequence:
        if item['type'] == 'text':
            messages[1]["content"].append({"type": "text", "text": item['content']})
        elif item['type'] == 'image':
            img = Image.open(item['content']).resize((224, 224))
            messages[1]["content"].append({"type": "image", "image": img})

    # Process inputs for model
    text_input = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    # Generate output
    output = model.generate(**model_inputs, max_new_tokens=512)
    return processor.batch_decode(
        output[:, model_inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataSet_name', required=True, help='Dataset name without extension')
    args = parser.parse_args()

    # Configure paths
    input_path = f"output/{args.dataSet_name}.json"
    output_dir = "output/summarization_pre"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{args.dataSet_name}_Qwen2-VL-7B_gen.json"

    # Load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype="float16",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # Process dataset
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    for key, entry in dataset.items():
        try:
            input_seq = build_input_sequence(entry)
            summary = generate_summary(input_seq, model, processor)
            entry['summarization_pre'] = summary
            print(f"Processed: {key}")
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            entry['summarization_pre'] = "Generation Error"

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()