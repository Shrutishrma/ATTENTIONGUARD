<div align="center">

# 🛡️ AttentionGuard
### **A Distributed Hybrid Edge–Cloud Computer Vision Framework for Real-Time Visual Biometrics, Gaze Tracking, and Automated Academic Integrity Enforcement**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace_512D-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/deepinsight/insightface)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face_Mesh_Wasm-00C7B7?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-0.32ms_Inference-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas_Cloud-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/atlas)

---

### **M.Sc. Artificial Intelligence & Machine Learning**
### **Computer Vision Project**
**Department of Computer Science | Christ Deemed to be University, Bangalore**  
**Academic Supervisor:** Prof. Nizar Banu P K | **Academic Year:** 2025 – 2026

</div>

---

## 👥 Authors & Team Details

| Team Member | Register Number | Project Role | Specialization & Contribution |
|---|---|---|---|
| **Satyam Chaturvedi** | `2548547` | **Lead Vision Architect & Deep Learning Pipeline** | **EyeStateNet (3-Block CNN)** architecture, PyTorch training pipeline, MRL dataset curation, and ONNX Runtime CPU inference acceleration. |
| **Shruti Sharma** | `2548554` | **Client Edge Vision & Perception Engineer** | **MediaPipe Face Mesh (WASM)** client pipeline (30 FPS), 3D head pose ($X/Y$ offset), iris-to-canthi gaze deflection, and EAR blink tracking. |
| **Huda Maniyar** | `2548526` | **Server Biometrics & Backend Analytics** | **InsightFace (SCRFD + 512D ArcFace)** cosine similarity matching, Flask-SocketIO async loop, MongoDB Atlas telemetry, and Invigilator dashboard. |

---

## 📌 Problem Statement

Remote computerized examinations are highly susceptible to academic dishonesty, proxy examinees, off-screen collusion, and unmonitored digital tab-switching. Traditional commercial proctoring solutions continuously stream high-definition video (720p/1080p) from hundreds of examinees to centralized cloud servers. This introduces:
1. **Severe Bandwidth Exhaustion:** Requires 1.5 to 3.0 Mbps per student (~2.5 Gbps for 1,000 concurrent candidates), causing connection dropouts in rural areas.
2. **Excessive Cloud Compute Costs:** Continuous multi-stream video decoding and GPU neural network inference impose prohibitive operational expenses.
3. **Biometric Privacy Risks:** Storing thousands of hours of candidates' private bedroom recordings on third-party cloud servers violates data privacy regulations (GDPR, FERPA).

---

## 🎯 Objectives

1. **Distributed Hybrid Edge–Cloud Execution:** Execute high-frequency (30 FPS) perceptual tracking directly in the browser via WebAssembly, and offload periodic biometric authentication (0.33 Hz) to the server.
2. **Dynamic 1:1 Identity Verification:** Ensure continuous candidate authenticity via 512-dimensional ArcFace deep embeddings compared against a baseline snapshot registered during onboarding.
3. **Multi-Factor Cheating Detection:** Real-time detection of head pose turns (yaw/pitch), off-screen gaze deflection, prolonged eye closures (drowsiness/phone reading), and unauthorized secondary faces.
4. **97%+ Network Bandwidth Reduction:** Eliminate raw video streaming by sending only lightweight compressed keyframes every 3 seconds (~41 Kbps total network usage).
5. **Zero Raw Video Cloud Storage:** Ensure candidate privacy by conducting in-memory feature extraction with zero persistent video recordings.

---

## 📊 Datasets Used

| Dataset Name | Domain / Purpose | Sample Count & Details | Source / Download Link |
|---|---|---|---|
| **MRL Eye Dataset** (VSB-TUO Benchmark) | Ocular Aperture & Drowsiness Classification | **84,898 eye images** across 37 subjects (50.6% Open, 49.4% Closed) under infrared/ambient light, with/without spectacles. | [MRL Eye Dataset Official Portal](http://mrl.cs.vsb.cz/eyedataset) / [Kaggle Benchmark Mirror](https://www.kaggle.com/datasets/taufidul/mrl-eye-dataset) |
| **Glint360k & LFW** (InsightFace Benchmark) | 512D Deep Metric Face Recognition | **17.1 million images** across 360,232 distinct identities for deep hyperspherical feature embedding ($\cos(\theta + m)$). | [InsightFace Glint360k Repository](https://github.com/deepinsight/insightface) / [LFW Benchmark](http://vis-www.cs.umass.edu/lfw/) |
| **MediaPipe 3D Mesh Benchmark** | 3D Face Topology & Iris Tracking | **468 3D dense facial landmarks + 10 refined iris vertices** for real-time edge geometry. | [Google MediaPipe Face Mesh](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) |

---

## 💻 Technologies & Libraries Used

* **Deep Learning & Computer Vision:** PyTorch 2.5, ONNX Runtime, InsightFace (SCRFD Detector + ArcFace 512D Backbone), Google MediaPipe Face Mesh (WASM), OpenCV (`cv2`), NumPy.
* **Backend & Real-Time Communication:** Python 3.12, Flask, Flask-SocketIO (WebSockets), Werkzeug, Gevent/Eventlet.
* **Database & Persistence:** MongoDB Atlas (Cloud Cluster), PyMongo, BSON.
* **Frontend & Client Interface:** HTML5, Modern Vanilla CSS3 (Glassmorphism), JavaScript (ES6+), WebAssembly (WASM), WebRTC (`getUserMedia`).
* **Document Parsing:** PyPDF2, `python-docx`, CSV parsing engine.

---

## 🔬 Methodology & Architecture

```
                                  [ CANDIDATE WEBCAM ]
                                           │
            ┌──────────────────────────────┴──────────────────────────────┐
            ▼                                                             ▼
   [ LAYER 1: CLIENT EDGE ]                                      [ LAYER 2: SERVER BACKEND ]
   (MediaPipe WASM — 30 FPS)                                     (Flask-SocketIO — Every 3s)
            │                                                             │
   468 3D Landmark Regression                                    OpenCV Frame Decode (cv2.imdecode)
            │                                                             │
   ┌────────┴────────────────────────┐                           SCRFD Neural Face Detection
   ▼                                 ▼                                    │
[ Gaze & Eye State ]       [ Head Pose Estimation ]                      Extract 512D ArcFace Vector
• Eye Aspect Ratio (EAR)   • Bounded Cheek-to-Nose Ratio                  │
• Iris-Canthus Vector      • 22% - 78% Safe Envelope             Cosine Similarity with Baseline
   │                                 │                                    │
   └────────┬────────────────────────┘                           Threshold Check (S_c >= 0.45)
            ▼                                                             │
   [ Edge Heuristic Check ] ─────────────────────────────────────► [ TELEMETRY ENGINE ]
                                                                          │
                                                                 MongoDB Atlas Persistence
                                                                          │
                                                                 Violation Ceiling >= 8 ?
                                                                   ├── YES ──► Auto-Submit & Disqualify
                                                                   └── NO  ──► Push Real-Time Warning
```

### Core Mathematical Formulations:
* **1. ArcFace Additive Angular Margin Loss:**
  $$\mathcal{L}_{\mathrm{ArcFace}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos\theta_j}}$$
  *(where scale $s = 64$ and angular margin $m = 0.50$)*

* **2. Eye Aspect Ratio (EAR):**
  $$\mathrm{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2.0 \cdot \|p_1 - p_4\|}$$
  *(where $p_1, p_4$ are eye corners, and $p_2, p_3, p_5, p_6$ are eyelid landmarks)*

* **3. Coordinate-Invariant Head Pose Normalization:**
  $$X_{\mathrm{offset}} = \frac{\mathrm{noseTip.x} - \min(\mathrm{leftCheek.x}, \mathrm{rightCheek.x})}{\max(\mathrm{leftCheek.x}, \mathrm{rightCheek.x}) - \min(\mathrm{leftCheek.x}, \mathrm{rightCheek.x})}$$

---

## 📈 Results & Performance Benchmarks

### 1. Classification & Verification Accuracy

| Evaluation Metric | Face Verification (ArcFace) | Eye State (EyeStateNet) | Iris Gaze Tracking | 3D Head Pose Estimation |
|---|---|---|---|---|
| **Accuracy** | **99.65%** | **98.40%** | **96.85%** | **97.20%** |
| **Precision** | 99.70% | 98.62% | 96.50% | 97.10% |
| **Recall / Sensitivity**| 99.60% | 98.18% | 97.20% | 97.30% |
| **F1-Score / ROC-AUC** | **0.9965 (AUC=0.998)** | **0.9840 (AUC=0.992)** | **0.9685 (AUC=0.978)** | **0.9720 (AUC=0.981)** |

### 2. Computational Latency & Network Savings

| Metric | Traditional Cloud Streaming | **AttentionGuard (Hybrid Edge-Cloud)** | System Advantage |
|---|---|---|---|
| **Client Uplink Bandwidth** | 1.5 – 3.0 Mbps (Continuous 720p stream) | **35 – 50 Kbps** (1 frame / 3s + WebAssembly) | **~97.9% reduction** |
| **Server Compute Overhead** | 30 FPS / student (Heavy multi-GPU) | **0.33 FPS / student** (Lightweight ONNX) | **90× compute savings** |
| **Gaze / Pose Latency** | 350 – 800 ms (Cloud network roundtrip) | **< 15 ms** (Client-side edge execution) | **Instant 30 FPS tracking** |
| **EyeStateNet Inference** | N/A | **0.32 ms** on CPU (ONNX Runtime) | **~3,100 FPS throughput** |

---

## 🚀 Steps to Execute the Project

### Prerequisites
- **Python 3.11 or 3.12**
- **MongoDB Atlas** account (or local MongoDB on port 27017)
- Standard integrated or external USB webcam

### 1. Clone the Repository
```bash
git clone https://github.com/Shrutishrma/ATTENTIONGUARD.git
cd ATTENTIONGUARD
```

### 2. Create & Activate Virtual Environment
```bash
# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux:
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory (refer to `.env.example`):
```env
MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/AttentionGuard?retryWrites=true&w=majority
SECRET_KEY=attentionguard_secure_production_key_2026
```

### 5. Seed Demo Database
```bash
python website/populate_db.py
```

### 6. Launch Application
```bash
python main.py
```
Open your web browser and navigate to **`http://localhost:5100`**.

---

## 🎮 Demo Credentials

| Role | Register Number | Password | Test Code | Available Functionality |
|---|---|---|---|---|
| **Student (Demo 1)** | `DEMO001` | `demo123` | `DEMO` | Takes exam with real-time AI proctoring HUD |
| **Student (Demo 2)** | `DEMO002` | `demo123` | `DEMO` | Secondary candidate profile |
| **Teacher / Admin** | `DEMOTEACHER` | `demo123` | *(None)* | Uploads exams (PDF/DOCX), monitors live feeds, exports CSV |

---

## 🧠 Model Training & Google Colab Notebook

To retrain or evaluate **EyeStateNet** on the MRL Eye Dataset:
* Run locally:
  ```bash
  python model_training/train_eye_state.py
  python model_training/evaluate.py
  ```
* Or execute in Google Colab: Open [`model_training/EyeStateClassifier_Colab.ipynb`](model_training/EyeStateClassifier_Colab.ipynb) for one-click cloud GPU training.

---

## 📂 Repository Structure

```
AttentionGuard/
├── main.py                             # Server entry point (Flask + Flask-SocketIO on port 5100)
├── requirements.txt                    # Python package dependencies
├── .env.example                        # Environment variable configuration template
├── README.md                           # Master project documentation
├── TEAM_WORK_DIVISION_AND_VIVA_GUIDE.md # Team roles, CV model mapping & viva defense guide
│
├── model_training/                     # Custom EyeStateNet CNN training pipeline
│   ├── train_eye_state.py              # PyTorch training & ONNX export script
│   ├── evaluate.py                     # Evaluation & confusion matrix script
│   ├── EyeStateClassifier_Colab.ipynb  # Google Colab GPU training notebook
│   └── models_out/                     # Exported PyTorch weights (.pth) & training plot
│
└── website/                            # Application package
    ├── __init__.py                     # Flask factory & PyMongo initialization
    ├── auth.py                         # Authentication routes (login, register, logout, ban)
    ├── views.py                        # Exam routes (rules, test, score, violation API)
    ├── teacher.py                      # Teacher portal routes (dashboard, upload, delete)
    ├── recognition.py                  # InsightFace biometrics & pure NumPy cosine similarity
    ├── parser.py                       # Multi-format document parser (PDF, DOCX, TXT, CSV)
    ├── models.py                       # MongoDB data access layer & auto-grading
    ├── populate_db.py                  # Database seeding script
    ├── create_embeddings.py            # Static enrollment utility
    ├── models/                         # Deployed ONNX models (eyestatenet.onnx)
    ├── static/                         # CSS stylesheets, icons, and assets
    └── templates/                      # Glassmorphic Jinja2 HTML templates
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
