import streamlit as st
from PIL import Image
import numpy as np
import cv2
import onnxruntime as ort
import torch
import torchvision.ops as ops
from ultralytics import FastSAM
from PIL import Image as PILImage
import tempfile
import os
import io

# COCO class list for mapping
COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic_light", "fire_hydrant", "stop_sign", "parking_meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", "kite",
    "baseball_bat", "baseball_glove", "skateboard", "surfboard", "tennis_racket", "bottle",
    "wine_glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch", "potted_plant",
    "bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell_phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy_bear", "hair_drier", "toothbrush"
]


# Paths & settings
DETR_MODEL_PATH = "detr_resnet50.onnx"
CONF_THRESH = 0.1
AWARE_CLASSES = {"Person": 1, "Cellphone": 77, "Book": 84}
REVERSE_AWARE_CLASSES = {1: "Person", 77: "Cellphone", 84: "Book", 72: "TV"}
UNWARE_CLASS = {"TV": 72}


@st.cache_resource
def load_detr():
    """Loads the DETR model into an ONNX runtime session."""
    sess = ort.InferenceSession(DETR_MODEL_PATH)
    inp = sess.get_inputs()[0]
    return sess, inp.name, inp.shape[2], inp.shape[3]


@st.cache_resource
def load_sam():
    """Loads the FastSAM model."""
    return FastSAM('FastSAM-s.pt')


@st.cache_data
def detect_centers(img: np.ndarray, class_ids: set) -> list:
    """
    Performs object detection using the DETR model.
    """
    sess, inp_name, H, W = load_detr()
    orig_h, orig_w = img.shape[:2]
    
    img_pil = Image.fromarray(img).convert("RGB")
    img_resized = img_pil.resize((W, H), resample=Image.BICUBIC)

    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    batch = np.transpose(img_np, (2, 0, 1))[None, ...]

    boxes_raw, scores_raw, classes_raw = sess.run(
        ["boxes", "logits", "classes"], {inp_name: batch}
    )

    boxes = torch.tensor(boxes_raw[0], dtype=torch.float32)
    scores = torch.tensor(scores_raw[0], dtype=torch.float32)
    labels = torch.tensor(classes_raw[0], dtype=torch.int64)

    mask = scores > CONF_THRESH
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

    keep = ops.nms(boxes, scores, 0.6) # Using a fixed IOU for general detection
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    scale_x, scale_y = orig_w / W, orig_h / H
    boxes = boxes.numpy() * np.array([scale_x, scale_y, scale_x, scale_y])

    dets = []
    idx = 1
    for box, lab in zip(boxes, labels.tolist()):
        if lab not in class_ids:
            continue
        x1, y1, x2, y2 = map(int, box)
        dets.append({
            "id": idx,
            "bbox": [x1, y1, x2, y2],
            "center": [(x1 + x2) // 2, (y1 + y2) // 2],
            "class": REVERSE_AWARE_CLASSES.get(lab, "Unknown")
        })
        idx += 1
    return dets


def inpaint_with_sam(img: np.ndarray, bboxes: list) -> np.ndarray:
    """
    Inpaints the specified bounding boxes in an image using FastSAM.
    """
    if not bboxes:
        return img
    sam = load_sam()
    results = sam.predict(source=img, bboxes=bboxes, device='cpu')
    if not results or results[0].masks is None:
        return img
        
    masks = results[0].masks.data
    binary = (torch.any(masks, dim=0).cpu().numpy().astype(np.uint8) * 255)
    
    if binary.shape != img.shape[:2]:
        binary = cv2.resize(
            binary,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    return cv2.inpaint(img, binary, 3, cv2.INPAINT_NS)


def process_video(video_path, process_frame_func, output_filename):
    """
    A generic video processing loop.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = os.path.join(tempfile.gettempdir(), output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame_rgb = process_frame_func(frame_rgb, i)
        processed_frame_bgr = cv2.cvtColor(processed_frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(processed_frame_bgr)

        progress = (i + 1) / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {i+1}/{total_frames}")

    cap.release()
    out.release()
    status_text.text("Video processing complete!")
    return output_path

def main():
    st.set_page_config(page_title="AURA Editor", layout="wide")

    # Inject custom CSS
    st.markdown("""
        <style>
            .stApp {
                background: url("https://images.pexels.com/photos/1103970/pexels-photo-1103970.jpeg?cs=srgb&dl=pexels-jplenio-1103970.jpg&fm=jpg")
                                 no-repeat center;
                background-size: cover;
            }
            .stSidebar {
                background-color: rgba(255,255,255,0.85);
                padding-top: 1rem;
            }
            .stButton > button {
                background-color: #264653;
                color: white;
                border-radius: 8px;
                padding: 0.6em 1.2em;
            }
            .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
                color: #264653;
            }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.image("AURA_logo.png", width=120)
    st.sidebar.title("📸 AURA")
    
    upload = st.sidebar.file_uploader(
        "Upload Your Media", 
        type=["png", "jpg", "jpeg", "mp4", "mov", "avi"]
    )
    
    if not upload:
        st.sidebar.info("Please upload an image or video file to begin.")
        return

    st.title("📸 AURA - Privacy-preserving Media Editor")
    
    if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file is not upload:
        st.session_state.uploaded_file = upload
        st.session_state.file_bytes = upload.getvalue()

    upload_type = upload.type
    is_image = "image" in upload_type
    
    media_type = st.selectbox(
        "1. Select Media Type",
        ["Image", "Video"],
        index=0 if is_image else 1
    )

    # --- IMAGE ---
    if media_type == "Image":
        mode = st.radio(
            "2. Select an Image Editing Mode",
            ["Aware", "Unaware", "Unaware Lite"],
            horizontal=True
        )
        
        img = np.array(Image.open(io.BytesIO(st.session_state.file_bytes)).convert("RGB"))
        st.image(img, caption="Original Image", use_container_width=True, channels="RGB")
        st.markdown("---")

        if mode == "Aware":
            st.subheader("Aware Mode: Select objects to keep")
            
            with st.spinner("Detecting objects..."):
                all_dets = detect_centers(img, set(AWARE_CLASSES.values()))

            if not all_dets:
                st.warning("No aware objects (Person, Cellphone, Book) detected.")
                return
            
            class_dets = {cls: [] for cls in AWARE_CLASSES.keys()}
            for d in all_dets:
                if d['class'] in class_dets:
                    class_dets[d['class']].append(d)

            to_remove = []
            cols = st.columns(len(AWARE_CLASSES))

            for i, (class_name, dets) in enumerate(class_dets.items()):
                with cols[i]:
                    st.markdown(f"#### {class_name}s")
                    ann_img = img.copy()
                    if dets:
                        for d in dets:
                            x1, y1, x2, y2 = d["bbox"]
                            cv2.rectangle(ann_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(ann_img, f"ID: {d['id']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        st.image(ann_img, caption=f"Detected {class_name}s")
                        
                        for d in dets:
                            keep = st.checkbox(f"Keep {class_name} {d['id']}", value=True, key=f"img_{d['class']}_{d['id']}")
                            if not keep:
                                to_remove.append(d["bbox"])
                    else:
                        st.info(f"No {class_name}s detected.")

            st.markdown("---")
            if st.button("Inpaint Selected"):
                if to_remove:
                    out = inpaint_with_sam(img, to_remove)
                    st.image(out, caption="Result", use_container_width=True)
                    buf = cv2.imencode('.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))[1].tobytes()
                    st.download_button("Download", data=buf, file_name="aura_aware_result.png", mime="image/png")
                else:
                    st.info("No objects selected for removal.")
        
        elif mode == "Unaware Lite":
            st.subheader("Unaware Lite Mode: Auto-inpaint Cellphone(s) and Book(s)")
            with st.spinner("Detecting cellphones and books..."):
                target_classes = {AWARE_CLASSES["Cellphone"], AWARE_CLASSES["Book"]}
                detections = detect_centers(img, target_classes)
            
            if detections:
                cellphones_to_inpaint = [d["bbox"] for d in detections if d["class"] == "Cellphone"]
                books_to_inpaint = [d["bbox"] for d in detections if d["class"] == "Book"]

                st.markdown("---")
                if cellphones_to_inpaint:
                    st.write(f"**Cellphones to be inpainted:** {len(cellphones_to_inpaint)} detected.")
                    for i, bbox in enumerate(cellphones_to_inpaint):
                        st.text(f"  - Cellphone {i+1}: {bbox}")
                else:
                    st.info("No Cellphones detected for inpainting.")

                if books_to_inpaint:
                    st.write(f"**Books to be inpainted:** {len(books_to_inpaint)} detected.")
                    for i, bbox in enumerate(books_to_inpaint):
                        st.text(f"  - Book {i+1}: {bbox}")
                else:
                    st.info("No Books detected for inpainting.")
                st.markdown("---")

                to_inpaint = [d["bbox"] for d in detections]
                if st.button("Inpaint Now"):
                    out = inpaint_with_sam(img, to_inpaint)
                    st.image(out, caption="Result", use_container_width=True)
                    buf = cv2.imencode('.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))[1].tobytes()
                    st.download_button("Download", data=buf, file_name="aura_unaware_lite.png", mime="image/png")
            else:
                st.info("No Cellphones or Books detected.")

        elif mode == "Unaware":
            st.subheader("Unaware Mode: Auto-inpaint objects inside Mirror(s)/TV(s)")
            with st.spinner("Analyzing..."):
                tvs = detect_centers(img, set(UNWARE_CLASS.values()))
                objs = detect_centers(img, set(AWARE_CLASSES.values()))
                inside = []
                for tv in tvs:
                    x1_tv, y1_tv, x2_tv, y2_tv = tv["bbox"]
                    for o in objs:
                        cx, cy = o["center"]
                        if x1_tv <= cx <= x2_tv and y1_tv <= cy <= y2_tv:
                            inside.append(o) # Append the full detection dict to get class info

            if inside:
                persons_in_tv = [d["bbox"] for d in inside if d["class"] == "Person"]
                cellphones_in_tv = [d["bbox"] for d in inside if d["class"] == "Cellphone"]
                books_in_tv = [d["bbox"] for d in inside if d["class"] == "Book"]
                
                st.markdown("---")
                if persons_in_tv:
                    st.write(f"**Persons inside TV(s) to be inpainted:** {len(persons_in_tv)} detected.")
                    for i, bbox in enumerate(persons_in_tv):
                        st.text(f"  - Person {i+1}: {bbox}")
                else:
                    st.info("No Persons detected inside TV(s).")
                
                if cellphones_in_tv:
                    st.write(f"**Cellphones inside TV(s) to be inpainted:** {len(cellphones_in_tv)} detected.")
                    for i, bbox in enumerate(cellphones_in_tv):
                        st.text(f"  - Cellphone {i+1}: {bbox}")
                else:
                    st.info("No Cellphones detected inside TV(s).")

                if books_in_tv:
                    st.write(f"**Books inside TV(s) to be inpainted:** {len(books_in_tv)} detected.")
                    for i, bbox in enumerate(books_in_tv):
                        st.text(f"  - Book {i+1}: {bbox}")
                else:
                    st.info("No Books detected inside TV(s).")
                st.markdown("---")

                bboxes_to_inpaint = [d["bbox"] for d in inside]
                if st.button("Inpaint Now"):
                    out = inpaint_with_sam(img, bboxes_to_inpaint)
                    st.image(out, caption="Result", use_container_width=True)
                    buf = cv2.imencode('.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))[1].tobytes()
                    st.download_button("Download", data=buf, file_name="aura_unaware.png", mime="image/png")
            else:
                st.info("No aware objects detected inside any TV.")
    
    # --- VIDEO ---
    elif media_type == "Video":
        mode = st.radio(
            "2. Select a Video Editing Mode",
            ["Aware-Video", "Unaware-Video", "Unaware-Lite-Video"],
            horizontal=True
        )

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(st.session_state.file_bytes)
        video_path = tfile.name
        st.video(video_path)
        st.markdown("---")

        if mode == "Aware-Video":
            st.subheader("Aware-Video Mode: Select objects in the first frame to track and remove")
            
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                st.error("Could not read the first frame of the video.")
                return

            first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with st.spinner("Detecting objects in the first frame..."):
                all_dets = detect_centers(first_frame, set(AWARE_CLASSES.values()))

            if not all_dets:
                st.warning("No aware objects (Person, Cellphone, Book) detected in the first frame.")
                return
            
            class_dets = {cls: [] for cls in AWARE_CLASSES.keys()}
            for d in all_dets:
                if d['class'] in class_dets:
                    class_dets[d['class']].append(d)

            to_track_initial = []
            cols = st.columns(len(AWARE_CLASSES))

            for i, (class_name, dets) in enumerate(class_dets.items()):
                with cols[i]:
                    st.markdown(f"#### {class_name}s")
                    ann_frame = first_frame.copy()
                    if dets:
                        for d in dets:
                            x1, y1, x2, y2 = d["bbox"]
                            cv2.rectangle(ann_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(ann_frame, f"ID: {d['id']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        st.image(ann_frame, caption=f"Detected {class_name}s")
                        
                        for d in dets:
                            keep = st.checkbox(f"Keep {class_name} {d['id']}", value=True, key=f"vid_{d['class']}_{d['id']}")
                            if not keep:
                                to_track_initial.append(tuple(d["bbox"]))
                    else:
                        st.info(f"No {class_name}s detected.")
            
            st.markdown("---")
            if st.button("Start Tracking and Inpainting"):
                if to_track_initial:
                    try:
                        st.session_state.trackers = [cv2.legacy.TrackerCSRT_create() for _ in to_track_initial]
                    except AttributeError:
                        st.error(
                            "AttributeError: Your OpenCV version is missing the CSRT Tracker.\n\n"
                            "Please upgrade and install the full package:\n\n"
                            "`pip uninstall opencv-python opencv-python-headless`\n\n"
                            "`pip install opencv-contrib-python`"
                        )
                        return
                    
                    for tracker, bbox_x1y1x2y2 in zip(st.session_state.trackers, to_track_initial):
                        x, y, w, h = bbox_x1y1x2y2[0], bbox_x1y1x2y2[1], bbox_x1y1x2y2[2] - bbox_x1y1x2y2[0], bbox_x1y1x2y2[3] - bbox_x1y1x2y2[1]
                        tracker.init(first_frame, (x, y, w, h))

                    def process_aware_frame(frame, frame_idx):
                        # For the first frame, use the initial detections
                        if frame_idx == 0:
                           return inpaint_with_sam(frame, [list(bbox) for bbox in to_track_initial])
                        
                        to_inpaint_current_frame = []
                        for tracker in st.session_state.trackers:
                            success, bbox_xywh = tracker.update(frame)
                            if success:
                                x, y, w, h = [int(v) for v in bbox_xywh]
                                to_inpaint_current_frame.append([x, y, x + w, y + h])
                        return inpaint_with_sam(frame, to_inpaint_current_frame)
                    
                    with st.spinner("Tracking and inpainting video... This may take a while."):
                        output_vid_path = process_video(video_path, process_aware_frame, "aware_video_result.mp4")
                    
                    st.success("Processing complete!")
                    st.video(output_vid_path)
                    with open(output_vid_path, "rb") as f:
                        st.download_button("Download Processed Video", f, file_name="aware_video_result.mp4")
                else:
                    st.info("No objects were selected for removal.")

        elif mode == "Unaware-Lite-Video":
            st.subheader("Unaware-Lite-Video: Auto-inpaint Cellphones and Books in each frame")
            
            # Pre-analysis for initial counts (optional, for display purposes)
            cap_pre = cv2.VideoCapture(video_path)
            ret_pre, frame_pre = cap_pre.read()
            cap_pre.release()
            
            if ret_pre:
                first_frame_rgb_pre = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2RGB)
                target_classes_pre = {AWARE_CLASSES["Cellphone"], AWARE_CLASSES["Book"]}
                detections_pre = detect_centers(first_frame_rgb_pre, target_classes_pre)
                
                cellphones_count = len([d for d in detections_pre if d["class"] == "Cellphone"])
                books_count = len([d for d in detections_pre if d["class"] == "Book"])
                
                st.info(f"The video will automatically inpaint {cellphones_count} Cellphone(s) and {books_count} Book(s) detected in the first frame, and attempt to do so for subsequent frames.")
            
            if st.button("Start Inpainting Video"):
                target_classes = {AWARE_CLASSES["Cellphone"], AWARE_CLASSES["Book"]}
                def process_unaware_lite_frame(frame, _):
                    detections = detect_centers(frame, target_classes)
                    return inpaint_with_sam(frame, [d["bbox"] for d in detections])
                with st.spinner("Detecting and inpainting video..."):
                    output_vid_path = process_video(video_path, process_unaware_lite_frame, "unaware_lite_video_result.mp4")
                st.success("Processing complete!")
                st.video(output_vid_path)
                with open(output_vid_path, "rb") as f:
                    st.download_button("Download Processed Video", f, file_name="unaware_lite_video_result.mp4")

        elif mode == "Unaware-Video":
            st.subheader("Unaware-Video: Auto-inpaint objects inside TVs in each frame")

            # Pre-analysis for initial counts (optional, for display purposes)
            cap_pre = cv2.VideoCapture(video_path)
            ret_pre, frame_pre = cap_pre.read()
            cap_pre.release()

            if ret_pre:
                first_frame_rgb_pre = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2RGB)
                tvs_pre = detect_centers(first_frame_rgb_pre, set(UNWARE_CLASS.values()))
                objs_pre = detect_centers(first_frame_rgb_pre, set(AWARE_CLASSES.values()))
                
                persons_in_tv_count = 0
                cellphones_in_tv_count = 0
                books_in_tv_count = 0

                for tv in tvs_pre:
                    x1_tv, y1_tv, x2_tv, y2_tv = tv["bbox"]
                    for o in objs_pre:
                        cx, cy = o["center"]
                        if x1_tv <= cx <= x2_tv and y1_tv <= cy <= y2_tv:
                            if o["class"] == "Person":
                                persons_in_tv_count += 1
                            elif o["class"] == "Cellphone":
                                cellphones_in_tv_count += 1
                            elif o["class"] == "Book":
                                books_in_tv_count += 1
                
                st.info(f"The video will automatically inpaint {persons_in_tv_count} Person(s), {cellphones_in_tv_count} Cellphone(s), and {books_in_tv_count} Book(s) detected inside TVs in the first frame, and attempt to do so for subsequent frames.")

            if st.button("Start Inpainting Video"):
                def process_unaware_frame(frame, _):
                    tvs = detect_centers(frame, set(UNWARE_CLASS.values()))
                    objs = detect_centers(frame, set(AWARE_CLASSES.values()))
                    inside = []
                    for tv in tvs:
                        x1_tv, y1_tv, x2_tv, y2_tv = tv["bbox"]
                        for o in objs:
                            cx, cy = o["center"]
                            if x1_tv <= cx <= x2_tv and y1_tv <= cy <= y2_tv:
                                inside.append(o["bbox"])
                    return inpaint_with_sam(frame, inside)
                with st.spinner("Analyzing and inpainting video..."):
                    output_vid_path = process_video(video_path, process_unaware_frame, "unaware_video_result.mp4")
                st.success("Processing complete!")
                st.video(output_vid_path)
                with open(output_vid_path, "rb") as f:
                    st.download_button("Download Processed Video", f, file_name="unaware_video_result.mp4")


if __name__ == "__main__":
    main()