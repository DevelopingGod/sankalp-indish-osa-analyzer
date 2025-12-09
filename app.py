import streamlit as st
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
from fpdf import FPDF
import tempfile
import datetime

# Project Imports
from src.pyceph.inference import load_model, process_image
from src.pyceph.Landmarks import Landmarks 

# ==========================================
# 1. CONFIGURATION & CSS
# ==========================================
st.set_page_config(layout="wide", page_title="OSA Analyzer", page_icon="🦷")

# Constants
MAX_CANVAS_SIZE = (2256, 2304) 
DISPLAY_WIDTH = 800          

st.markdown("""
<style>
    /* 1. Global Background */
    .stApp { background-color: #f4f6f9; }

    /* 2. CRITICAL: Stop the "Flash" / Gray-out effect */
    .stApp > div {
        transition: none !important;
        opacity: 1 !important;
        animation: none !important;
    }
    
    /* 3. Hide the small "Running" man/spinner in top right */
    [data-testid="stStatusWidget"] {
        visibility: hidden;
    }
    
    /* 4. Canvas Container Styling */
    .canvas-container {
        border: 2px solid #2d3748;
        border-radius: 4px;
        background-color: #000;
    }

    /* 5. Custom UI Elements */
    .status-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #e6fffa;
        border-left: 5px solid #38b2ac;
        color: #234e52;
        margin-bottom: 10px;
    }
    .prediction-box {
        padding: 15px;
        text-align: center;
        font-weight: bold;
        color: white;
        border-radius: 8px;
        margin-top: 15px;
        font-size: 1.2rem;
    }
    .osa { background-color: #dc3545; }
    .borderline { background-color: #ffc107; color: black; }
    .normal { background-color: #28a745; }
    
    /* 6. Footer Styling */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
        color: #555;
        font-size: 14px;
    }
    .footer a {
        color: #0078D4;
        text-decoration: none;
        font-weight: bold;
        margin: 0 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. STATE MANAGEMENT
# ==========================================
keys = [
    'uploaded_file_bytes', 'high_res_pil', 'display_pil', 'scale_ratio',
    'ai_landmarks', 'px_per_mm', 'measurements', 'click_history', 
    'ruler_points', 'temp_angle_points', 'canvas_key', 'final_result',
    'is_calibrated', 'pdf_bytes', 'original_dims' 
]
for k in keys:
    if k not in st.session_state:
        if k == 'measurements': st.session_state[k] = []
        elif k == 'click_history': st.session_state[k] = []
        elif k == 'ruler_points': st.session_state[k] = []
        elif k == 'temp_angle_points': st.session_state[k] = []
        elif k == 'px_per_mm': st.session_state[k] = 11.53 
        elif k == 'canvas_key': st.session_state[k] = 0
        elif k == 'is_calibrated': st.session_state[k] = False
        else: st.session_state[k] = None

# ==========================================
# 3. CORE LOGIC
# ==========================================
DIAGNOSTIC_NORMS = {
    "Parameter": [
        "Sella to Nasion (S-N)", 
        "Porion to Orbitale (Po-Or)", 
        "Mentale to Gonion (Me-Go)", 
        "Posterior Airway Space (PAS)", 
        "Pharynx Length", 
        "Hyoid Bone Position (MP-H)"
    ],
    "Normal": ["67.9 ± 4.3 mm", "68.5 ± 3.8 mm", "72.4 ± 4.2 mm", "11.5 ± 2.1 mm", "70.3 ± 4.6 mm", "15.2 ± 3.1 mm"],
    "Borderline OSA": ["65.2 ± 4.0 mm", "66.2 ± 3.6 mm", "70.1 ± 4.0 mm", "9.2 ± 1.8 mm", "75.1 ± 5.2 mm", "18.6 ± 3.5 mm"],
    "OSA": ["62.1 ± 3.7 mm", "63.7 ± 3.4 mm", "67.5 ± 3.9 mm", "6.8 ± 1.5 mm", "82.4 ± 6.3 mm", "22.4 ± 4.2 mm"]
}

@st.cache_resource
def get_model():
    return load_model(os.path.join("model", "12-26-22.pkl.gz"))

@st.cache_data
def process_and_cache_images(file_bytes):
    """Scientific Processing with Padding and Labeling."""
    model = get_model()
    if not model: return None, None, None, None, None
    
    # A. Inference
    raw_img, raw_marks = process_image(BytesIO(file_bytes), model)
    h_raw, w_raw = raw_img.shape[:2]
    
    # B. High-Res Layer (Padded)
    img_uint8 = (raw_img * 255).astype(np.uint8)
    pil_raw = Image.fromarray(img_uint8)
    high_res_pil = ImageOps.pad(pil_raw, MAX_CANVAS_SIZE, color="black", centering=(0.5, 0.5))
    
    # Calculate scale/offset
    scale_factor = min(MAX_CANVAS_SIZE[0]/w_raw, MAX_CANVAS_SIZE[1]/h_raw)
    new_w = int(w_raw * scale_factor)
    new_h = int(h_raw * scale_factor)
    offset_x = (MAX_CANVAS_SIZE[0] - new_w) // 2
    offset_y = (MAX_CANVAS_SIZE[1] - new_h) // 2
    
    high_res_marks = []
    for (x, y) in raw_marks:
        nx = int(x * scale_factor) + offset_x
        ny = int(y * scale_factor) + offset_y
        high_res_marks.append((nx, ny))

    # Burn dots for Download/PDF
    draw_hr = ImageDraw.Draw(high_res_pil)
    try: font_hr = ImageFont.truetype("arial.ttf", 40)
    except: font_hr = None
    for i, (hx, hy) in enumerate(high_res_marks):
        r = 10
        draw_hr.ellipse((hx-r, hy-r, hx+r, hy+r), fill="red", outline="white")
        try: name = str(Landmarks(i)).replace("Landmarks.", "")[:3].title()
        except: name = str(i)
        draw_hr.text((hx+15, hy-15), name, fill="yellow", font=font_hr)

    # C. Display Layer
    aspect = MAX_CANVAS_SIZE[1] / MAX_CANVAS_SIZE[0]
    display_h = int(DISPLAY_WIDTH * aspect)
    display_pil = high_res_pil.resize((DISPLAY_WIDTH, display_h), Image.LANCZOS)
    scale_ratio = MAX_CANVAS_SIZE[0] / DISPLAY_WIDTH
    
    return high_res_pil, display_pil, scale_ratio, high_res_marks, (w_raw, h_raw)

def dist_euclidean(p1, p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def calculate_final_prediction(measurements):
    osa_votes = 0
    total_relevant = 0
    
    for m in measurements:
        label = m['Label'].upper()
        try: val = float(m['Value'].split(' ')[0])
        except: continue
        
        if "PAS" in label:
            total_relevant += 1
            if val < 9.2: osa_votes += 1
        elif "PHARYNX" in label:
            total_relevant += 1
            if val > 75.0: osa_votes += 1
        elif "MP-H" in label or "HYOID" in label:
            total_relevant += 1
            if val > 18.0: osa_votes += 1
        elif "S-N" in label:
            total_relevant += 1
            if val < 65.0: osa_votes += 1
            
    if total_relevant == 0: return "Insufficient Data"
    if osa_votes > (total_relevant / 2): return f"OSA DETECTED ({osa_votes}/{total_relevant} criteria)"
    elif osa_votes > 0: return f"BORDERLINE ({osa_votes}/{total_relevant} criteria)"
    else: return "NON-OSA (Normal Anatomy)"

def generate_pdf_report(image_pil, measurements, prediction, user_info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "OSA Cephalometric Analysis Report", ln=True, align="C")
    pdf.ln(5)
    
    # 1. User Info
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"Patient Name: {user_info['name']}", ln=True)
    pdf.cell(0, 6, f"Date: {str(user_info['date'])}", ln=True)
    pdf.cell(0, 6, f"Contact: {user_info['contact']}", ln=True)
    pdf.cell(0, 6, f"Email: {user_info['email']}", ln=True)
    pdf.ln(5)
    
    # 2. Image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        ratio = image_pil.height / image_pil.width
        img_small = image_pil.resize((500, int(500 * ratio)))
        img_small.save(tmp.name)
        pdf.image(tmp.name, x=55, y=pdf.get_y(), w=100)
    
    pdf.ln(100 * ratio + 10)
    
    # 3. Prediction
    pdf.set_font("Arial", "B", 14)
    if "OSA" in prediction and "NON" not in prediction: pdf.set_text_color(220, 53, 69)
    elif "BORDERLINE" in prediction: pdf.set_text_color(255, 193, 7)
    else: pdf.set_text_color(40, 167, 69)
    pdf.cell(0, 10, f"Final Assessment: {prediction}", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    # 4. Norms Table
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 8, "Diagnostic Norms Reference", ln=True)
    pdf.set_font("Arial", "B", 8)
    pdf.cell(50, 6, "Parameter", 1)
    pdf.cell(45, 6, "Normal", 1)
    pdf.cell(45, 6, "Borderline", 1)
    pdf.cell(45, 6, "OSA", 1)
    pdf.ln()
    
    pdf.set_font("Arial", "", 8)
    params = DIAGNOSTIC_NORMS['Parameter']
    n_norm = DIAGNOSTIC_NORMS['Normal']
    n_bord = DIAGNOSTIC_NORMS['Borderline OSA']
    n_osa = DIAGNOSTIC_NORMS['OSA']
    
    for i in range(len(params)):
        pdf.cell(50, 6, params[i], 1)
        pdf.cell(45, 6, n_norm[i], 1)
        pdf.cell(45, 6, n_bord[i], 1)
        pdf.cell(45, 6, n_osa[i], 1)
        pdf.ln()
    pdf.ln(8)

    # 5. Measurements
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 8, "Patient Measurements", ln=True)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(95, 8, "Label", 1)
    pdf.cell(95, 8, "Value", 1)
    pdf.ln()
    
    pdf.set_font("Arial", "", 10)
    for m in measurements:
        pdf.cell(95, 8, str(m['Label']), 1)
        pdf.cell(95, 8, str(m['Value']), 1)
        pdf.ln()
        
    # 6. Footer
    pdf.ln(15)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 10, "💻 Report downloaded from OSA Analyzer [Computer Generated]", ln=True, align="C")
    
    return pdf.output(dest='S').encode('latin-1', errors='replace')

# ==========================================
# 4. MAIN UI LAYOUT
# ==========================================
st.title("🦷 OSA [Obstructive Sleep Apnea] Analyzer")

with st.expander("ℹ️ Instructions & Help"):
    st.markdown("""
    **How to Use:**
    1. **Upload:** Upload a lateral cephalogram X-ray. You can also upload a reference image for guidance.
    2. **Calibrate:** Use the 'Calibrate Ruler' tool to click two points on the image's ruler (usually 10mm). This is auto-calibrated to 11.53 px/mm by default. You can use that and adjust manually as per need.
    3. **Measure:** Use 'Measure Distance' and/or 'Measure Angle' to analyze anatomy. Make sure that you label S-N, Po-Or, PAS, Pharynx Length, MP-H, etc. as per the norms table.
    4. **Report:** Click 'Analyze' to get an OSA prediction and download a PDF report.
    """)

# --- ROW 1: UPLOADS ---
col_u1, col_u2 = st.columns(2)
with col_u1:
    uploaded = st.file_uploader("1. Upload Patient X-Ray", type=['jpg', 'png'])
    if uploaded: st.caption("✅ File Ready")
with col_u2:
    ref_upload = st.file_uploader("2. Upload Reference Image", type=['jpg', 'png'])

# --- PROCESS ---
if uploaded:
    if st.session_state.uploaded_file_bytes != uploaded.getvalue():
        st.session_state.uploaded_file_bytes = uploaded.getvalue()
        st.session_state.measurements = []
        st.session_state.click_history = []
        st.session_state.canvas_key += 1
        st.session_state.final_result = None
        st.session_state.is_calibrated = False
        st.session_state.pdf_bytes = None
        
        with st.spinner("Processing..."):
            hr, disp, ratio, lms, dims = process_and_cache_images(uploaded.getvalue())
            st.session_state.high_res_pil = hr
            st.session_state.display_pil = disp
            st.session_state.scale_ratio = ratio
            st.session_state.ai_landmarks = lms
            st.session_state.original_dims = dims
            st.rerun()

# --- WORKSPACE ---
if st.session_state.display_pil:
    w, h = st.session_state.original_dims
    st.markdown(f"""
    <div class="status-box">
        ✅ <b>Auto-Marking Complete</b><br>
        Scientific Resize: Anatomy preserved. [{w}px x {h}px]<br>
    </div>
    """, unsafe_allow_html=True)
    
    buf = BytesIO()
    st.session_state.high_res_pil.save(buf, format="PNG")
    st.download_button("📥 Download Auto-Marked Image", data=buf.getvalue(), file_name="auto_marked.png", mime="image/png")
    
    st.divider()

    col_canvas, col_tools = st.columns([1.8, 1]) 
    
    # === TOOLS ===
    with col_tools:
        if ref_upload:
            st.image(ref_upload, caption="Reference", use_container_width=True)
            st.divider()

        st.markdown("#### 🛠 Controls")
        
        if not st.session_state.is_calibrated:
            st.markdown("""<div class="warning-box">⚠️ <b>Uncalibrated</b>: Measurements may be inaccurate.</div>""", unsafe_allow_html=True)
        
        mode = st.radio("Active Tool", ["📏 Calibrate Ruler", "📐 Measure Distance", "🔄 Measure Angle"])
        label_input = st.text_input("Label:", value="MPH")
        
        st.markdown("##### Calibration")
        new_calib = st.number_input("Pixels per MM (px/mm):", value=float(st.session_state.px_per_mm), format="%.2f", step=0.1)
        if new_calib != st.session_state.px_per_mm:
            st.session_state.px_per_mm = new_calib
            st.toast("Calibration Updated")
            
        st.divider()
        
        # Tabs
        tab_measure, tab_dist_norms, tab_angle_norms = st.tabs(["Measurements", "Distance Norms", "Angle Norms"])
        
        with tab_dist_norms:
            st.markdown("**Diagnostic Norms**")
            st.dataframe(pd.DataFrame(DIAGNOSTIC_NORMS), hide_index=True)
        with tab_angle_norms:
            st.info("Angle Norms Reference (Empty)")
        with tab_measure:
            if st.session_state.measurements:
                st.dataframe(pd.DataFrame(st.session_state.measurements)[["Label", "Value"]], hide_index=True, use_container_width=True)
                if st.button("Clear Log"):
                    st.session_state.measurements = []
                    st.session_state.click_history = []
                    st.session_state.final_result = None
                    st.session_state.pdf_bytes = None
                    st.rerun()
            else:
                st.info("No measurements yet.")

        # Final Analysis
        st.markdown("### Analysis")
        if st.button("🔮 Analyze", type="primary", use_container_width=True):
            res = calculate_final_prediction(st.session_state.measurements)
            st.session_state.final_result = res
            
        if st.session_state.final_result:
            r = st.session_state.final_result
            c = "osa" if "OSA" in r else "borderline" if "BORDERLINE" in r else "normal"
            st.markdown(f'<div class="prediction-box {c}">{r}</div>', unsafe_allow_html=True)
            
            st.markdown("#### Generate Report")
            with st.form("pdf_form"):
                n = st.text_input("Name")
                e = st.text_input("Email")
                c = st.text_input("Contact")
                d = st.date_input("Date", datetime.date.today())
                submitted = st.form_submit_button("Generate PDF")
                
                if submitted:
                    u_info = {"name": n, "email": e, "contact": c, "date": d}
                    pdf_data = generate_pdf_report(
                        st.session_state.high_res_pil, 
                        st.session_state.measurements, 
                        r, 
                        u_info
                    )
                    st.session_state.pdf_bytes = pdf_data
            
            if st.session_state.pdf_bytes:
                st.download_button("📄 Download PDF Report", data=st.session_state.pdf_bytes, file_name="osa_report.pdf", mime="application/pdf", use_container_width=True)

    # === CANVAS ===
    with col_canvas:
        st.markdown(f"**Interactive Canvas**")
        if "Calibrate" in mode: st.info("Click 2 points on the ruler.")
        elif "Distance" in mode: st.info(f"Click 2 points to measure '{label_input}'.")
        
        # --- FIXED: Use PIL Image (Works with Streamlit 1.35.0) ---
        canvas_result = st_canvas(
            fill_color="lime", stroke_width=2,
            background_image=st.session_state.display_pil, # Passing PIL Image Object
            update_streamlit=True,
            height=int(st.session_state.display_pil.height),
            width=DISPLAY_WIDTH,
            drawing_mode="point",
            point_display_radius=4,
            key=f"canvas_{st.session_state.canvas_key}"
        )
        
        if canvas_result.json_data:
            objects = canvas_result.json_data["objects"]
            current_clicks = [(obj["left"]+obj["radius"], obj["top"]+obj["radius"]) for obj in objects if obj["type"] == "circle"]
            
            if len(current_clicks) > len(st.session_state.click_history):
                click = current_clicks[-1]
                st.session_state.click_history.append(click)
                ratio = st.session_state.scale_ratio
                pt_real = (click[0] * ratio, click[1] * ratio)
                
                if "Calibrate" in mode:
                    st.session_state.ruler_points.append(pt_real)
                    if len(st.session_state.ruler_points) == 2:
                        d_px = dist_euclidean(st.session_state.ruler_points[0], st.session_state.ruler_points[1])
                        st.session_state.px_per_mm = d_px / 10.0 
                        st.session_state.ruler_points = []
                        st.session_state.is_calibrated = True
                        st.session_state.canvas_key += 1
                        st.toast(f"Calibrated: {st.session_state.px_per_mm:.2f} px/mm")
                        st.rerun()
                elif "Distance" in mode:
                    if "t_dist" not in st.session_state: st.session_state.t_dist = []
                    st.session_state.t_dist.append(pt_real)
                    if len(st.session_state.t_dist) == 2:
                        d_mm = dist_euclidean(st.session_state.t_dist[0], st.session_state.t_dist[1]) / st.session_state.px_per_mm
                        st.session_state.measurements.append({"Label": label_input, "Value": f"{d_mm:.2f} mm"})
                        st.session_state.t_dist = []
                        st.session_state.canvas_key += 1
                        st.rerun()
                elif "Angle" in mode:
                    st.session_state.temp_angle_points.append(pt_real)
                    if len(st.session_state.temp_angle_points) == 3:
                        pts = st.session_state.temp_angle_points
                        a, v, b = np.array(pts[0]), np.array(pts[1]), np.array(pts[2])
                        va, vb = a-v, b-v
                        cos = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
                        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
                        st.session_state.measurements.append({"Label": label_input, "Value": f"{ang:.1f}°"})
                        st.session_state.temp_angle_points = []
                        st.session_state.canvas_key += 1
                        st.rerun()
            st.session_state.click_history = current_clicks
else:
    st.info("🩻 Please upload the cephalogram to begin.")

# === FOOTER ===
st.markdown("""
<div class="footer">
    <p>© Copyright 2025</p>
    <p>Made with ❤️ by <b>Sankalp Indish</b> and medically assisted by <b>Vaidehi Mahod</p>
    <p>
    <a href="https://www.linkedin.com/in/sankalp-indish" target="_blank">LinkedIn [Sankalp]</a> | <a href="https://github.com/DevelopingGod" target="_blank">GitHub [Sankalp] </a> | <a href="https://www.linkedin.com/in/vaidehi-mohod-0116932b1/?originalSubdomain=in" target="_blank">LinkedIn [Vaidehi]</a> 
    </p>
</div>
""", unsafe_allow_html=True)
