# 🦷 OSA Analyzer: AI-Powered Cephalometric Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

**OSA Analyzer** is a state-of-the-art web application designed to assist clinicians in screening for **Obstructive Sleep Apnea (OSA)** using routine lateral cephalograms. It uses **Deep Learning (ResNet)** for automated landmark detection with computer vision for precise geometric analysis.

---

## 🚀 Features

- **🤖 AI Auto-Marking**: Automatically detects 19 cephalometric landmarks in <5 seconds.
- **📏 Precision Calibration**: "Human-in-the-Loop" ruler calibration ensures 100% accurate measurements regardless of image resolution.
- **📐 Automated Analysis**: Calculates critical diagnostic metrics
- **🔮 Intelligent Prediction**: Voting-based algorithm classifies patients as *Normal*, *Borderline*, or *OSA Likely*.
- **📄 Professional Reporting**: Generates downloadable PDF reports with patient details, annotated X-rays, and diagnostic norms.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/osa-analyzer.git](https://github.com/your-username/osa-analyzer.git)
cd osa-analyzer

2. Install Dependencies
Bash

pip install -r requirements.txt

3. Run the App
Bash

streamlit run app.py


🖥️ Usage Workflow
Upload: Drag & drop a lateral cephalogram (JPG/PNG).

Verify: The AI will auto-mark landmarks. Review the red dots.

Calibrate: Select the "Calibrate Ruler" tool and click two points on the X-ray's ruler (10mm).

Analyze:

Click "Measure Distance" to check Airway Space (PAS) or Hyoid Position (MPH).

Click "Final Analyze" to get the OSA prediction.

Report: Enter patient details and download the PDF Report.

🏗️ Tech Stack
Frontend: Streamlit

Deep Learning: PyTorch (Fusion ResNet/HRNet)

Image Processing: OpenCV, Pillow, Scikit-Image

Reporting: FPDF

👥 Credits
Developed with ❤️ by Sankalp Indish

LinkedIn: Sankalp Indish

Research Partner: Vaidehi Mohod