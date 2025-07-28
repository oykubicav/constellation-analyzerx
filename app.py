# app.py
import os, re, cv2, numpy as np, streamlit as st
from PIL import Image, ImageOps
from dotenv import load_dotenv
from ultralytics import YOLO

from langchain_community.vectorstores import FAISS

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings







st.title("Constellation Analyzer")

CLASS_LIST = [
    'aquila','bootes','canis_major','canis_minor','cassiopeia','cygnus',
    'gemini','leo','lyra','moon','orion','pleiades','sagittarius',
    'scorpius','taurus','ursa_major'
]
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={"device": "cpu"}  # bazı ortamlarda işe yarayabilir
)


hf_pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256,
    do_sample=False,
    no_repeat_ngram_size=6,
    repetition_penalty=1.2,
    early_stopping=True
)

llm = HuggingFacePipeline(pipeline=hf_pipe)


# ---- YOLO ----
model = YOLO("best.pt")
class_names = model.names

def preprocess_image(pil_image: Image.Image) -> Image.Image:
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm, (3, 3), 0.5)
    return Image.fromarray(blur)

def run_inference(img_path: str):
    results = model(img_path)
    results[0].save(filename="inference_result.jpg")
    return results[0]

def get_yolo_json(results):
    boxes = results.boxes.xywh.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()
    return [
        {
            "class": class_names[c],
            "confidence": float(conf),
            "x": float(b[0]), "y": float(b[1]),
            "width": float(b[2]), "height": float(b[3])
        }
        for b, c, conf in zip(boxes, classes, confs)
    ]

def json_to_text(dets):
    if not dets:
        return "No constellation detected with the current thresholds."
    return "\n".join(
        f"Image contains {d['class']} at ({int(d['x'])}, {int(d['y'])}) with {int(d['confidence']*100)}% confidence."
        for d in dets
    )
SYSTEM_RULES = f"""
You are a constellation QA assistant.
You ONLY know what is in the given detection Context plus brief general astronomy facts.

Detection Context lines look like:
"Image contains <class> at (<x>, <y>) with <confidence>% confidence."

CLASS LIST (use exact spellings): {", ".join(CLASS_LIST)}.

RULES:
1. Answer in the same language as the question.
2. Presence questions ("Was X detected?", "Is there Orion?", "Kaç tane Lyra var?"):
   - First answer ONLY "YES" or "NO" (uppercase).
   - If YES, also give: Coords: [(x,y), ...] and Confidences: [..%]
3. Counting questions ("How many", "Kaç tane"): give an integer and list them.
4. General explanation questions ("Explain Orion", "Bu görüntüde ne var?"):
   - Summarize detections from Context (what, where, confidence).
   - Then add 1–2 factual astronomy sentences for each detected object.
5. If the requested class is NOT in Context, say "NO" and DO NOT invent coordinates.
6. Never hallucinate detections not in Context.
7. If Context is empty, say you don't know or ask the user to run detection.
8. Keep answers concise and fact-based.

### Examples:

Context:
Image contains orion at (240, 310) with 94% confidence.

Question:
Explain Orion

Answer:
The image contains Orion at (240, 310) with 94% confidence.
Orion is a bright constellation named after a hunter in Greek mythology. It includes the stars Betelgeuse and Rigel.

---

Context:
Image contains moon at (211, 79) with 69% confidence.

Question:
Explain Moon

Answer:
The image contains Moon at (211, 79) with 69% confidence.
The Moon is Earth's only natural satellite. It controls ocean tides and reflects sunlight, appearing in phases.
"""

prompt_tmpl = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_RULES + "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)

# ---- Session state ----
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ---- Upload ----
uploaded_file = st.file_uploader("📁 Upload an image of a constellation", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = ImageOps.exif_transpose(img).resize((640, 640))
    st.image(img, caption="Original Image", use_container_width=True)

    os.makedirs("uploads", exist_ok=True)
    img_path = os.path.join("uploads", uploaded_file.name)

    pre_img = preprocess_image(img)
    pre_img.save(img_path)
    st.image(pre_img, caption="Preprocessed Image", use_container_width=True)

    if st.button("🔍 Analyze Constellation"):
        result = run_inference(img_path)
        st.image("inference_result.jpg", caption="YOLO Detection", use_container_width=True)

        dets = get_yolo_json(result)
        summary_text = json_to_text(dets)

        doc = Document(page_content=summary_text, metadata={"image": uploaded_file.name})
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = FAISS.from_documents([doc], embedding_model)
        else:
            st.session_state.vectorstore.add_documents([doc])

        st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})

        st.text_area("Detection Summary", summary_text, height=200)
        st.success("Detection result added.")

# ---- QA ----
# --- QA kısmını değiştir ---
query = st.text_input("Ask a question about the detections:")

if query:
    if st.session_state.vectorstore is None:
        st.warning("Please analyze an image first.")
    else:
        # 1) Docları çek
        docs = st.session_state.vectorstore.as_retriever(search_kwargs={"k":5}).get_relevant_documents(query)
        context = "\n".join(d.page_content for d in docs)
        chain = RetrievalQA.from_chain_type(
            llm=llm
            retriever=st.session_state.retriever,
            chain_type="stuff",
            return_source_documents=False,
            chain_type_kwargs={"prompt: prompt_tmpl}
                               )
        answer = chain.run(query)
        st.write("Answer")
        st.markdown(answer)
 
