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

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# ------------------- Config -------------------
JSON_PATH = "wardrobe_analysis.json"
CSV_PATH = "wardrobe_analysis.csv"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
LLM_MODEL = "llava-hf/llava-1.5-7b-hf"  # fallback to stable model
API_KEY = "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"
# ---------------------------------------------

client = InferenceClient(model=LLM_MODEL, token=API_KEY)
embedder = SentenceTransformer(EMBEDDING_MODEL)

image_data_store = {}
image_id_map = {}
index = None

def resize_image(image: Image.Image, scale=0.5) -> Image.Image:
    width, height = image.size
    return image.resize((int(width * scale), int(height * scale)))

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

def ensure_defaults(parsed):
    return {
        "type": parsed.get("type") or "Shirt",
        "color": parsed.get("color") or "Black",
        "style": parsed.get("style") or "Casual",
        "season": parsed.get("season") or "All Seasons",
        "occasion": parsed.get("occasion") or "General",
        "suggestions": parsed.get("suggestions") or ["Pair with sneakers", "Layer with jacket"]
    }

def parse_response(text):
    return ensure_defaults({
        "type": re.search(r"(?i)type\W*:\W*(.+)", text).group(1).strip() if re.search(r"(?i)type\W*:", text) else None,
        "color": re.search(r"(?i)color\W*:\W*(.+)", text).group(1).strip() if re.search(r"(?i)color\W*:", text) else None,
        "style": re.search(r"(?i)style\W*:\W*(.+)", text).group(1).strip() if re.search(r"(?i)style\W*:", text) else None,
        "season": re.search(r"(?i)season\W*:\W*(.+)", text).group(1).strip() if re.search(r"(?i)season\W*:", text) else None,
        "occasion": re.search(r"(?i)occasion\W*:\W*(.+)", text).group(1).strip() if re.search(r"(?i)occasion\W*:", text) else None,
        "suggestions": re.findall(r"(?i)pair with ([^\.\n]+)[\.\n]", text)
    })

def save_json():
    with open(JSON_PATH, "w") as jf:
        json.dump(image_data_store, jf, indent=2)

def save_csv():
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

def rebuild_index():
    global index
    if not image_data_store:
        index = None
        return
    vectors = [embedder.encode(data['type'] + " " + data['style']) for data in image_data_store.values()]
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors).astype("float32"))

def analyze_image(image: Image.Image, image_id: str):
    image_base64 = encode_image_to_base64(image)

    wardrobe_summary = ""
    for other_id, entry in image_data_store.items():
        wardrobe_summary += f"- {entry['type']} in {entry['color']} ({entry['style']} style)\n"

    prompt = (
        "You are a fashion stylist. Analyze the clothing item shown in the image and return the following:\n"
        "1. Item Type\n"
        "2. Color\n"
        "3. Style (casual, formal, semi-formal)\n"
        "4. Suitable Season\n"
        "5. Occasions to wear\n"
        "6. Suggested pairing items ONLY from the list below:\n\n"
        f"Existing wardrobe items:\n{wardrobe_summary}\n\n"
        "Format your response like:\n"
        "Type: ...\nColor: ...\nStyle: ...\nSeason: ...\nOccasion: ...\nSuggestions: Pair with ..."
    )

    contents = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_base64}},
    ]

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": contents}],
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"API error: {e}")
        return ""

def process_uploaded_files(uploaded_files):
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        img = resize_image(img, scale=0.5)
        img_id = os.path.splitext(file.name)[0] + "_" + str(uuid.uuid4())[:8]

        if img_id not in st.session_state["image_id_map"]:
            st.session_state["image_id_map"][img_id] = img

def analyze_all_uploaded_images():
    for img_id, img in st.session_state["image_id_map"].items():
        if img_id not in image_data_store:
            response = analyze_image(img, img_id)
            if not response:
                continue
            parsed = parse_response(response)
            parsed["raw"] = response
            image_data_store[img_id] = parsed
    save_json()
    save_csv()
    rebuild_index()

def answer_question(question):
    if not image_data_store or not index:
        return "No wardrobe data available."
    question_vec = embedder.encode(question).astype("float32")
    _, I = index.search(np.array([question_vec]), k=min(5, len(image_data_store)))
    matched_ids = list(image_data_store.keys())
    output = ""
    for idx in I[0]:
        if idx >= len(matched_ids):
            continue
        img_id = matched_ids[idx]
        data = image_data_store[img_id]
        output += f"### Image {img_id}\n"
        output += f"- **Type:** {data['type']}\n"
        output += f"- **Color:** {data['color']}\n"
        output += f"- **Style:** {data['style']}\n"
        output += f"- **Occasion:** {data['occasion']}\n"
        output += f"- **Season:** {data['season']}\n"
        output += f"- **Suggested Pairing:** {', '.join(data.get('suggestions', []))}\n\n"
    return output.strip()

def delete_item(image_id):
    if image_id not in image_data_store:
        return f"⚠️ No match found for '{image_id}'."
    image_data_store.pop(image_id, None)
    image_id_map.pop(image_id, None)
    if "image_id_map" in st.session_state:
        st.session_state["image_id_map"].pop(image_id, None)
    save_json()
    save_csv()
    rebuild_index()
    return f"✅ Deleted: {image_id}"

st.set_page_config(page_title="AI Stylist", layout="wide")
st.title("🧥 AI Stylist — Wardrobe Advisor with RAG & Vision")

if "image_id_map" not in st.session_state:
    st.session_state["image_id_map"] = {}

uploaded_files = st.file_uploader("Upload Clothing Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if uploaded_files:
    st.subheader("📂 Uploaded Files (Select to keep)")
    files_to_keep = []
    for file in uploaded_files:
        col1, col2 = st.columns([1, 4])
        with col1:
            keep = st.checkbox("Keep", value=True, key=file.name)
        with col2:
            st.write(file.name)
        if keep:
            files_to_keep.append(file)

    if st.button("Process Selected Files"):
        with st.spinner("Uploading selected wardrobe items..."):
            process_uploaded_files(files_to_keep)
        st.success("Images uploaded!")

if st.session_state["image_id_map"]:
    st.subheader("👗 Wardrobe Gallery")
    image_ids = list(st.session_state["image_id_map"].keys())
    cols = st.columns(4)
    for i, img_id in enumerate(image_ids):
        col = cols[i % 4]
        with col:
            st.image(st.session_state["image_id_map"][img_id], caption=img_id, use_container_width=True)

    st.subheader("🗑️ Delete an item from wardrobe")
    selected_id = st.selectbox("Select an item to delete by Image ID", options=image_ids)
    if st.button("Delete Selected Item"):
        result = delete_item(selected_id)
        if "Deleted" in result:
            st.success(result)
        else:
            st.warning(result)

st.subheader("💬 Ask your stylist")
user_question = st.text_input("What would you like to ask?")
if user_question and st.button("Analyze My Wardrobe"):
    with st.spinner("Analyzing images and generating response..."):
        analyze_all_uploaded_images()
        answer = answer_question(user_question)
    st.markdown(answer)
