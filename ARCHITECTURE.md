# Defectra AI

**Project Type:** AI-Powered Industrial Visual Anomaly Detection  
**Primary Domain:** Computer Vision and Industrial Quality Inspection  
**Dataset:** MVTec Anomaly Detection (MVTec AD)  
**Application Stack:** Streamlit, FastAPI, PyTorch, CNN Autoencoder, ResNet18

---

## 1. Project Overview

Defectra AI is an AI-powered visual inspection application designed to identify potential defects and abnormal patterns in industrial products using computer vision.

Traditional quality inspection often relies on manual inspection or predefined rules. These approaches can be time-consuming and may struggle to identify subtle or previously unseen defects.

Defectra AI explores deep-learning-based anomaly detection to learn the visual characteristics of normal products and identify deviations from those patterns. The application aims to provide visual feedback through anomaly heatmaps, helping users understand which regions of an image may require further inspection.

The initial implementation focuses on the **bottle category** from the MVTec AD dataset, with the architecture intended to support expansion to additional product categories.

### Key Objectives

- Automate visual inspection using deep learning.
- Identify abnormal regions in product images.
- Generate anomaly heatmaps for visual interpretation.
- Separate the user interface, API layer, and inference logic.
- Establish an extensible foundation for multi-category industrial inspection.

## 2. Problem Statement

Industrial manufacturers need reliable methods to detect surface defects, structural irregularities, and other visual abnormalities.

Defectra AI addresses this problem by exploring an anomaly-detection approach in which models learn representations of normal product appearance and evaluate new images for deviations.

The system is intended to support inspection workflows rather than replace final quality-control decisions without human verification.

## 3. High-Level System Architecture

The application follows a layered architecture that separates user interaction from model inference.

```text
                 USER
                   |
                   v
          Streamlit Web Interface
                   |
                   v
            FastAPI Backend
                   |
                   v
          Image Processing Layer
                   |
                   v
          Model Inference Layer
             /           \
            v             v
      CNN Autoencoder   ResNet18
            \             /
             v           v
           Anomaly Analysis
                   |
                   v
         Anomaly Map / Results
                   |
                   v
          Streamlit Interface
```

*Note: This diagram represents the intended high-level architecture. The exact inference path and integration between the models should match the implementation in the repository.*

### 3.1 Presentation Layer — Streamlit

The Streamlit application provides the user-facing interface.

Responsibilities include:
- Allowing users to upload product images.
- Sending images to the backend for analysis.
- Displaying inspection results.
- Presenting anomaly heatmaps or other visual outputs.
- Showing relevant prediction details when available.

### 3.2 API Layer — FastAPI

FastAPI provides the communication layer between the user interface and the machine-learning components.

Responsibilities include:
- Accepting image-analysis requests.
- Validating incoming inputs.
- Coordinating image preprocessing and model inference.
- Returning prediction results in a structured response.
- Providing a foundation for future integrations with other applications.

Specific endpoint names and response schemas should be documented once verified against the implemented API routes.

### 3.3 Image Processing Layer

The image-processing component prepares input images for model inference.

Typical responsibilities include:
- Loading and validating images.
- Converting images into the expected format.
- Resizing images to the model's input dimensions.
- Applying the appropriate normalization and tensor transformations.
- Preparing outputs for visualization.

The preprocessing configuration must remain consistent with the configuration used during model training.

### 3.4 Model Inference Layer

Defectra AI explores two complementary computer-vision approaches.

**CNN Autoencoder**

An autoencoder learns to reconstruct images from a compressed representation. When trained primarily on normal product images, it may reconstruct familiar patterns more accurately than abnormal ones.

The difference between an input image and its reconstruction can be used as an anomaly signal. This signal may help identify regions that deviate from the learned normal appearance.

**Pretrained ResNet18**

ResNet18 is a convolutional neural network that can extract hierarchical visual features from images.

Its pretrained feature representations can be used to compare product images or image regions against reference representations of normal samples. The resulting feature differences can help identify potentially anomalous patterns.

The two approaches provide different perspectives on visual abnormalities. Their relative performance should be evaluated experimentally before selecting a production inference strategy.

## 4. End-to-End Workflow

The intended inspection workflow is as follows:

1. **Image input:** The user uploads an image through the Streamlit interface.
2. **Request validation:** The backend receives the image and checks that it can be processed.
3. **Preprocessing:** The image is resized, normalized, and converted into the expected model input format.
4. **Model inference:** The configured anomaly-detection model processes the image.
5. **Anomaly analysis:** Reconstruction differences or feature deviations are analyzed to identify suspicious regions.
6. **Result generation:** The system produces the available anomaly scores, maps, or classifications.
7. **Visualization:** The interface displays the original image alongside the generated results.
8. **Inspection:** The user reviews the highlighted regions for potential defects.

The exact outputs depend on the model implementation and the postprocessing logic.

## 5. Dataset and Model Development

### MVTec AD Dataset

MVTec AD is a benchmark dataset for industrial anomaly detection. It contains images of different object and texture categories, including normal samples and defective samples.

The dataset supports evaluating whether a model can distinguish normal appearance from a range of anomalous patterns.

### Current Development Focus

The initial project focuses on the bottle category to establish and evaluate the core pipeline.

The development process includes:
- Preparing normal training images.
- Applying consistent image preprocessing.
- Training or loading the selected model.
- Evaluating performance on held-out normal and anomalous images.
- Examining anomaly maps and prediction behavior.

### Planned Expansion

The broader goal is to extend support to all 15 MVTec AD categories. This requires category-aware data handling, suitable model artifacts or training configurations, and consistent evaluation across object and texture categories.

## 6. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| PyTorch | Deep-learning model development and inference |
| CNN Autoencoder | Reconstruction-based anomaly detection |
| ResNet18 | Pretrained visual feature extraction |
| FastAPI | Backend API and inference coordination |
| Streamlit | Interactive user interface |
| MVTec AD | Industrial anomaly-detection benchmark |
| Git and GitHub | Version control and source-code collaboration |

Additional deployment technologies should be listed when they are present in the implementation.

## 7. Expected Outputs

Depending on the configured model and implementation, the application can provide:

- An uploaded product image.
- An anomaly score or other quantitative output, if implemented.
- A reconstruction-error map or feature-based anomaly map.
- A heatmap highlighting suspicious image regions.
- A visual comparison between the original image and the anomaly visualization.

Anomaly scores should not automatically be interpreted as calibrated defect probabilities unless the model has been explicitly calibrated and validated for that purpose.

## 8. Evaluation Strategy

The system should be evaluated using separate image-level and pixel-level metrics where ground-truth annotations are available.

Potential metrics include:

- **Image-level AUROC:** Measures how well normal and anomalous images are ranked.
- **Pixel-level AUROC:** Measures discrimination between normal and anomalous pixels.
- **Precision and recall:** Measures the quality of anomaly decisions at a defined threshold.
- **Inference latency:** Measures the time required to analyze an image.
- **Resource usage:** Measures memory and compute requirements.

Evaluation should use held-out test data and consistent preprocessing. Performance results should be added to the documentation only after they have been measured.

## 9. Repository Organization

The repository should keep application code, model logic, configuration, and documentation clearly separated.

An example structure is:

```text
AI_Vision/
├── api/
│   └── Backend API and request handling
├── ui/
│   └── Streamlit user interface
├── models/
│   └── Model definitions and inference logic
├── preprocessing/
│   └── Image preparation utilities
├── tests/
│   └── Automated tests
├── autoe.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

This is a suggested logical structure, not a claim that every directory already exists in the current repository. The documentation should be adjusted to match the actual files.

## 10. Deployment and Scalability

The separation between the user interface and API allows each component to evolve independently.

Potential future improvements include:

- Containerizing the API and interface.
- Adding automated tests and CI/CD.
- Supporting multiple industrial product categories.
- Managing model versions and inference configurations.
- Adding structured logging and request validation.
- Monitoring inference latency and prediction quality.
- Introducing model-artifact storage separate from source code.

Large datasets, trained model weights, virtual environments, and generated artifacts should generally be stored outside the ordinary Git source history unless they are intentionally managed using an appropriate large-file storage solution.

## 11. Current Scope and Future Roadmap

**Current focus**
- Establishing the industrial anomaly-detection workflow.
- Evaluating the bottle category.
- Exploring reconstruction-based and feature-based detection.
- Building the interface-to-inference workflow.

**Future roadmap**
- Expand to all 15 MVTec AD categories.
- Improve anomaly localization and result visualization.
- Compare models using consistent evaluation metrics.
- Improve error handling, testing, and deployment.
- Optimize inference performance and resource consumption.

## 12. Conclusion

Defectra AI provides a foundation for AI-assisted industrial visual inspection using deep-learning-based anomaly detection.

Its layered architecture separates user interaction, API orchestration, image preprocessing, and model inference. By combining experimentation with reconstruction-based and feature-based approaches, the project can evaluate different strategies for identifying visual abnormalities.

The longer-term objective is to develop an extensible inspection application that supports multiple industrial categories and produces interpretable results for human review.

---

**Documentation status:** High-level architectural overview. Implementation details, API endpoints, measured performance, and deployment instructions should be verified against the current source code before being treated as finalized.
