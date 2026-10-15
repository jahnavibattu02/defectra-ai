# Defectra AI

**Intelligent Visual Inspection for Industrial Quality Control**

Defectra AI is a computer vision application that explores automated defect detection in industrial products using deep learning. Built around the MVTec Anomaly Detection (AD) benchmark, it brings image-based inspection and model inference into an application designed for practical experimentation and deployment.

The project combines computer vision, model evaluation, and application development to explore how AI can help identify visual irregularities in manufacturing workflows.

## What It Does

- 🔎 **Inspect product images** for potential visual anomalies.
- 🧠 **Run anomaly detection models** through a reusable inference workflow.
- 📊 **Compare detection approaches** to understand their strengths and limitations.
- ⚡ **Serve predictions through an API** using FastAPI.
- 🖥️ **Explore results through a user interface** for a more accessible inspection workflow.
- 🐳 **Support containerized execution** with Docker.

## Built With

`Python` · `Computer Vision` · `Deep Learning` · `FastAPI` · `Streamlit` · `Docker` · `MVTec AD`

## Quick Start

Clone the repository:

```bash
git clone https://github.com/jahnavibattu02/defectra-ai.git
cd defectra-ai
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Start the application using the entry point configured in the repository. For example, if the API is exposed as `app` in `main.py`:

```bash
uvicorn main:app --reload
```

API documentation, when enabled, is available at:

`http://127.0.0.1:8000/docs`

## Dataset

This project uses the [MVTec Anomaly Detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad), a benchmark for research into industrial anomaly detection.

Download the dataset separately and configure its location according to the project setup. Dataset files and large model artifacts should not be committed directly to the repository.

## Documentation

For the detailed system architecture, model methodology, evaluation strategy, API design, and deployment details, see the project's dedicated technical documentation.

## Project Status

An evolving computer vision project focused on building a practical workflow for AI-assisted industrial inspection.

## Author

**Jahnavi Pravaleeka Battu**

---

*Defectra AI — Detect anomalies. Improve inspection. Build smarter quality-control workflows.*
