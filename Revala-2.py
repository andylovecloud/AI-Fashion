import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
from io import BytesIO
import base64
import json
import csv
import os
import uuid
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
import logging

# ------------------- Config -------------------
JSON_PATH = "wardrobe_analysis.json"
CSV_PATH = "wardrobe_analysis.csv"
LOG_PATH = "model_responses.log"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
API_KEY = "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"  # Replace with your HF token
# ---------------------------------------------

# Setup logging
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format='%(asctime)s - %(message)s')

client = InferenceClient(model=LLM_MODEL, token=API_KEY)
embedder = SentenceTransformer(EMBEDDING_MODEL)

image_data_store = {}
image_id_map = {}
index = None

# ------------------- Utilities -------------------

def load_wardrobe():
    global image_data_store, image_id_map
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r") as jf:
                loaded_data = json.load(jf)
                image_data_store.update(loaded_data)
            # Rebuild image_id_map (images not saved, so regenerate placeholders)
            for img_id in image_data_store:
                image_id_map[img_id] = None  # Placeholder; images need re-upload for display
            logging.info(f"Loaded {len(image_data_store)} items from {JSON_PATH}")
        except Exception as e:
            logging.error(f"Failed to load wardrobe: {e}")
            st.error(f"Failed to load wardrobe data: {e}")

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

def ensure_defaults(parsed):
    return {
        "type": parsed.get("type") or "Unknown",
        "color": parsed.get("color") or "Unknown",
        "style": parsed.get("style") or "Casual",
        "season": parsed.get("season") or "All Seasons",
        "occasion": parsed.get("occasion") or "General",
        "suggestions": parsed.get("suggestions") or ["No suggestions available"]
    }

def parse_response(text):
    parsed = {
        "type": None,
        "color": None,
        "style": None,
        "season": None,
        "occasion": None,
        "suggestions": []
    }

    type_match = re.search(r"(?i)(?:type|item)\s*[:=]\s*([^:\n]+)", text, re.IGNORECASE)
    color_match = re.search(r"(?i)color\s*[:=]\s*([^:\n]+)", text, re.IGNORECASE)
    style_match = re.search(r"(?i)style\s*[:=]\s*([^:\n]+)", text, re.IGNORECASE)
    season_match = re.search(r"(?i)season\s*[:=]\s*([^:\n]+)", text, re.IGNORECASE)
    occasion_match = re.search(r"(?i)occasion\s*[:=]\s*([^:\n]+)", text, re.IGNORECASE)

    parsed["type"] = type_match.group(1).strip() if type_match else None
    parsed["color"] = color_match.group(1).strip() if color_match else None
    parsed["style"] = style_match.group(1).strip() if style_match else None
    parsed["season"] = season_match.group(1).strip() if season_match else None
    parsed["occasion"] = occasion_match.group(1).strip() if occasion_match else None

    suggestions = re.findall(r"(?i)(?:^\d+\.\s*|\-\s*|pair with\s*)([^.\n]+)", text, re.MULTILINE)
    parsed["suggestions"] = [s.strip() for s in suggestions][:5] if suggestions else []

    return ensure_defaults(parsed)

def reprocess_response(raw_response, image_base64):
    prompt = (
        f"""A vision model analyzed an outfit and gave this response:\n\n{raw_response}\n\n
Reformat it into this structured style only, correcting any errors:

Type: [e.g., Shirt, Jeans]
Color: [e.g., Black, White]
Style: [e.g., Casual, Formal]
Season: [e.g., Summer]
Occasion: [e.g., Work]
Suggestions:
1. [e.g., Pair with white sneakers]
2. [...]
3. [...]
4. [...]
5. [...]

Only return in this format. Use the image + raw text to extract missing or vague fields.
"""
    )
    contents = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_base64}},
    ]
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": contents}],
            max_tokens=1500,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Reprocess error: {e}")
        return f"Error: {e}"

def save_json():
    try:
        with open(JSON_PATH, "w") as jf:
            json.dump(image_data_store, jf, indent=2)
        logging.info(f"Saved wardrobe to {JSON_PATH}")
    except Exception as e:
        logging.error(f"Failed to save JSON: {e}")
        st.error(f"Failed to save wardrobe data: {e}")

def save_csv():
    try:
        with open(CSV_PATH, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["ImageID", "Type", "Color", "Style", "Occasion", "Season", "Suggestions"])
            for image_id, entry in image_data_store.items():
                writer.writerow([
                    image_id,
                    entry["type"],
                    entry["color"],
                    entry["style"],
                    entry["occasion"],
                    entry["season"],
                    ", ".join(entry["suggestions"])
                ])
        logging.info(f"Saved wardrobe to {CSV_PATH}")
    except Exception as e:
        logging.error(f"Failed to save CSV: {e}")
        st.error(f"Failed to save CSV: {e}")

def rebuild_index():
    global index
    if not image_data_store:
        index = None
        logging.info("No items in wardrobe, index set to None")
        return
    try:
        vectors = [embedder.encode(data['type'] + " " + data['style']) for data in image_data_store.values()]
        index = faiss.IndexFlatL2(len(vectors[0]))
        index.add(np.array(vectors).astype("float32"))
        logging.info(f"Rebuilt FAISS index with {len(vectors)} items")
    except Exception as e:
        logging.error(f"Failed to rebuild index: {e}")
        st.error(f"Failed to rebuild index: {e}")

def analyze_image(image: Image.Image, image_id: str):
    image_base64 = encode_image_to_base64(image)
    prompt = (
        "Look at this image of a clothing item and identify:\n"
        "- The **type of clothing** (e.g., Shirt, Jeans, Sneakers)\n"
        "- The **color** (e.g., Blue, White, Black)\n\n"
        "Return your response strictly in the following format:\n"
        "Type: <value>\n"
        "Color: <value>\n\n"
        "Only return these two fields. Do not return any suggestions, metadata, or additional explanation."
    )

    contents = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_base64}},
    ]
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": contents}],
            max_tokens=1500,
        )
        response = completion.choices[0].message.content.strip()
        logging.info(f"Image {image_id} raw response:\n{response}")
        parsed = parse_response(response)

        if parsed["type"] == "Unknown" or parsed["color"] == "Unknown" or not parsed["suggestions"] or parsed["suggestions"] == ["No suggestions available"]:
            response = reprocess_response(response, image_base64)
            logging.info(f"Image {image_id} reprocessed response:\n{response}")
            parsed = parse_response(response)

        return response
    except Exception as e:
        logging.error(f"Analyze error for {image_id}: {e}")
        st.error(f"Analysis error for {image_id}: {e}")
        return f"Error: {e}"

def process_uploaded_files(files):
    status_messages = []
    for file in files:
        try:
            img = Image.open(file).convert("RGB")
            img_id = os.path.splitext(file.name)[0] + "_" + str(uuid.uuid4())[:8]

            if img_id not in image_data_store:
                raw_response = analyze_image(img, img_id)
                parsed = parse_response(raw_response)

                if parsed["type"] == "Unknown" or parsed["color"] == "Unknown" or not parsed["suggestions"] or parsed["suggestions"] == ["No suggestions available"]:
                    raw_response = reprocess_response(raw_response, encode_image_to_base64(img))
                    parsed = parse_response(raw_response)

                parsed["raw"] = raw_response
                image_data_store[img_id] = parsed
                image_id_map[img_id] = img
                status_messages.append(f"Processed {img_id}")
        except Exception as e:
            logging.error(f"Error processing file {file.name}: {e}")
            status_messages.append(f"Failed to process {file.name}: {e}")

    save_json()
    save_csv()
    rebuild_index()

    return status_messages

def answer_question(question):
    if not image_data_store or not index:
        return "No wardrobe data available. Please upload and analyze images first."

    question_vec = embedder.encode(question).astype("float32")
    _, I = index.search(np.array([question_vec]), k=min(5, len(image_data_store)))
    matched_ids = list(image_data_store.keys())
    output = ""

    wardrobe_texts = {img_id: data['type'].lower() + " " + data['color'].lower() + " " + data['style'].lower() for img_id, data in image_data_store.items()}

    def find_wardrobe_match(suggestion):
        suggestion_vec = embedder.encode(suggestion).astype("float32")
        wardrobe_vecs = list(embedder.encode(list(wardrobe_texts.values())))
        wardrobe_index = faiss.IndexFlatL2(len(suggestion_vec))
        wardrobe_index.add(np.array(wardrobe_vecs).astype("float32"))
        D, I = wardrobe_index.search(np.array([suggestion_vec]), k=1)
        best_match_score = D[0][0]
        best_match_idx = I[0][0]

        if best_match_score < 0.7:
            match_id = list(wardrobe_texts.keys())[best_match_idx]
            matched_data = image_data_store[match_id]
            return f"✅ Found in wardrobe: {matched_data['type']} ({match_id})"
        else:
            return f"🛍️ Not in wardrobe.\n Suggested: {suggestion}\n Options to buy: [Buy on H&M](https://www2.hm.com/en_us/search-results.html?q={suggestion.replace(' ', '+')})"

    for idx in I[0]:
        img_id = matched_ids[idx]
        data = image_data_store[img_id]
        output += f"Wardrobe Image {img_id}\n"
        output += f"- **Type:** {data['type']}\n"
        output += f"- **Color:** {data['color']}\n"
        output += f"- **Style:** {data['style']}\n"
        output += f"- **Occasion:** {data['occasion']}\n"
        output += f"- **Season:** {data['season']}\n"

        suggestions = data.get("suggestions", [])
        output += "- **Suggested Pairing:**\n"
        for s in suggestions:
            output += f"  - {find_wardrobe_match(s)}\n"
        output += "\n"

    return output.strip()

def delete_item(image_id_partial):
    full_id = next((key for key in image_data_store if image_id_partial in key), None)
    if full_id:
        image_data_store.pop(full_id, None)
        image_id_map.pop(full_id, None)
        save_json()
        save_csv()
        rebuild_index()
        return f"✅ Deleted: {full_id}"
    return f"⚠️ {image_id_partial} not found."

# ------------------- Streamlit UI -------------------

# Load existing wardrobe data at startup
load_wardrobe()
rebuild_index()

st.title("🧥 AI Stylist — Wardrobe Advisor with RAG & Vision")

# Debug Info
st.sidebar.header("Debug Info")
st.sidebar.write(f"Wardrobe items: {len(image_data_store)}")
st.sidebar.write(f"Index status: {'Active' if index else 'None'}")

# Image Upload Section
st.header("Upload Clothing Images")
uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if st.button("Analyze Images") and uploaded_files:
    with st.spinner("Analyzing images..."):
        status_messages = process_uploaded_files(uploaded_files)
        for msg in status_messages:
            if "Failed" in msg:
                st.error(msg)
            else:
                st.success(msg)

# Wardrobe Gallery
st.header("Wardrobe Gallery")
if image_id_map:
    cols = st.columns(4)
    for idx, (img_id, img) in enumerate(image_id_map.items()):
        with cols[idx % 4]:
            if img:
                st.image(img, caption=img_id, use_container_width=True)
            else:
                st.text(f"{img_id} (Image not available)")
else:
    st.info("No items in wardrobe yet.")

# Question Section
st.header("Ask Your Stylist")
question = st.text_input("E.g., What should I wear for an interview?")
if st.button("Get Suggestions") and question:
    with st.spinner("Generating suggestions..."):
        response = answer_question(question)
        st.markdown(response)

# Delete Item Section
st.header("Delete Wardrobe Item")
delete_id = st.text_input("Enter Image ID to Delete")
if st.button("Delete Item") and delete_id:
    delete_status = delete_item(delete_id)
    st.write(delete_status)

if __name__ == "__main__":
    st.write("AI Stylist is running...")