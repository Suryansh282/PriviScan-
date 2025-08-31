# PriviScan- scanning with privacy in mind

**PriviScan** is a Streamlit app that detects **persons, cellphones, books, and TVs** in images/videos and performs **privacy-preserving editing** via AI inpainting.  
It combines a **DETR (ONNX)** detector with **FastSAM** segmentation to remove/anonymize content in three modes: **Aware**, **Unaware**, and **Unaware-Lite**.

---

## ✨ Features

- **Multi-object detection** (selected COCO classes): Person, Cellphone, Book, TV  
- **Image modes**
  - **Aware** – you choose which detected objects to keep/remove  
  - **Unaware-Lite** – auto-inpaint **Cellphone(s)** and **Book(s)**  
  - **Unaware** – auto-inpaint **objects inside TV(s)** (e.g., reflections)
- **Video modes** mirroring image modes (Aware uses **CSRT** tracking)
- **AI inpainting**: FastSAM masks → OpenCV inpaint
- **Streamlit UI**: upload, preview, process, and download

---

## 🏗️ Architecture (high-level)

1. **Detect** with DETR (ONNX): boxes + class IDs  
2. **Select** boxes to remove (mode logic)  
3. **Segment** with FastSAM to get pixel-accurate masks  
4. **Inpaint** masked regions with OpenCV  
5. **Render** results and offer download

---

## 📁 Project Structure

```text
.
├── app.py                      # Streamlit app (put your code here)
├── requirements.txt            # Python dependencies
├── README.md
├── assets/
│   └── AURA_logo.png           # referenced in the sidebar
└── models/
    ├── detr_resnet50.onnx      # required by load_detr()
    └── FastSAM-s.pt            # required by load_sam()
```

> If model files are large, see **Deploying models** for Git LFS or runtime download.

---

## 🔧 Requirements

```txt
streamlit>=1.33
ultralytics>=8.2
torch>=2.2
torchvision>=0.17
onnxruntime>=1.17
opencv-contrib-python>=4.9
pillow>=10.2
numpy>=1.26
requests>=2.31  # only needed if you auto-download model files
```

- **opencv-contrib-python** is required for **cv2.legacy.TrackerCSRT_create** (Aware-Video mode).  
- **onnxruntime** is needed for the DETR ONNX session.

---

## 🚀 Quickstart (Local)

1. **Clone & enter repo**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Create & activate venv (recommended)**
   ```bash
   # Windows
   python -m venv env
   env\Scripts\activate

   # macOS / Linux
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install deps**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Place model files**
   ```text
   models/detr_resnet50.onnx
   models/FastSAM-s.pt
   assets/AURA_logo.png
   ```

5. **Run**
   ```bash
   streamlit run app.py
   ```
   Open the local URL (e.g., http://localhost:8501), upload media, choose a mode, process, and download.

---

## ☁️ Deploying to Streamlit Cloud

### A) If each model file is **< 100 MB**
1. Commit models into `models/` and push via **git** (the web uploader blocks files > 25 MB).
   ```bash
   git add .
   git commit -m "Add app and models"
   git push
   ```
2. Deploy via https://share.streamlit.io by selecting your repo and `app.py`.

### B) If any model file is **≥ 100 MB** → **Git LFS**
```bash
git lfs install
git lfs track "*.onnx" "*.pt"
git add .gitattributes models/*.onnx models/*.pt
git commit -m "Add models via LFS"
git push
```
Then deploy on Streamlit Cloud.

### C) **Recommended:** Don’t commit weights; **auto-download at runtime**
1. Host weights on **Hugging Face / GitHub Releases / S3** (direct URLs).
2. Add secrets in **Streamlit Cloud → App → Settings → Secrets** (or locally in `.streamlit/secrets.toml`):
   ```toml
   DETR_URL = "https://<your-host>/detr_resnet50.onnx"
   FASTSAM_URL = "https://<your-host>/FastSAM-s.pt"
   ```
3. In `app.py`, download on first run:
   ```python
   import os, requests, streamlit as st

   def _download(url, dest):
       os.makedirs(os.path.dirname(dest), exist_ok=True)
       if os.path.exists(dest):
           return dest
       with requests.get(url, stream=True, timeout=120) as r:
           r.raise_for_status()
           with open(dest, "wb") as f:
               for chunk in r.iter_content(1<<20):  # 1 MB chunks
                   if chunk: f.write(chunk)
       return dest

   DETR_MODEL_PATH = _download(st.secrets["DETR_URL"], os.path.join("models","detr_resnet50.onnx"))
   FASTSAM_PATH    = _download(st.secrets["FASTSAM_URL"], os.path.join("models","FastSAM-s.pt"))
   ```
4. Ensure `requirements.txt` includes `requests`.

---

## ⚙️ Configuration Notes

- Use **relative paths** in code:
  ```python
  import os
  DETR_MODEL_PATH = os.path.join("models", "detr_resnet50.onnx")
  FASTSAM_PATH    = os.path.join("models", "FastSAM-s.pt")
  LOGO_PATH       = os.path.join("assets", "AURA_logo.png")
  ```
- The DETR ONNX must return outputs named **"boxes"**, **"logits"**, **"classes"** (or adjust `sess.run([...])` accordingly).
- Verify COCO **class indices** used in your model match the hardcoded IDs for Person/Cellphone/Book/TV in the app.

---

## 🧭 Usage

1. **Upload** an image (JPG/PNG) or video (MP4/MOV/AVI).  
2. Select **Media Type**: Image or Video.  
3. Pick a **Mode**:
   - **Aware** – choose detections to keep/remove.
   - **Unaware-Lite** – inpaint **Cellphones** and **Books** automatically.
   - **Unaware** – inpaint objects inside **TV(s)**.
4. Click the action button (e.g., **Inpaint Now** / **Start … Video**).  
5. **Preview** results and **Download**.

---

## 🛠️ Troubleshooting

- `ModuleNotFoundError: No module named 'ultralytics'`  
  → `pip install ultralytics`

- `ModuleNotFoundError: No module named 'onnxruntime'`  
  → `pip install onnxruntime` (or `onnxruntime-gpu` if needed)

- `AttributeError: cv2.legacy.TrackerCSRT_create missing`  
  → Replace plain OpenCV with contrib build:
  ```bash
  pip uninstall -y opencv-python opencv-python-headless
  pip install opencv-contrib-python
  ```

- GitHub **web** upload error: “**Yowza… > 25 MB**”  
  → Use `git push` (limit 100 MB), **Git LFS**, or **runtime download** via secrets.

---

## 🔒 Privacy Notes

- Inpainting obscures content visually; validate adequacy for your use case.  
- For sensitive media, prefer **local processing** and disable logs/telemetry.

---

## 🗺️ Roadmap

- [ ] Batch processing  
- [ ] Face/plate anonymization presets  
- [ ] GPU inference options  
- [ ] Export masks/overlays

---

## 🤝 Contributing

Contributions welcome! Please open an issue to discuss significant changes.

---

## 🙏 Acknowledgements

- **Ultralytics** (FastSAM)  
- **DETR** authors (ONNX export)  
- **OpenCV** (tracking & inpainting)  
- Background image attribution per linked Pexels URL in code
