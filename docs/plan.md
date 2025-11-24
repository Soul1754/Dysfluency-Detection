## Plan: Build the Intelligent Speech Dysfluency Platform

### 1. Data & Labeling Pipeline

- **Parquet ingestion utilities**: Scripts/notebooks to load Parquet datasets (audio + metadata) using `pandas`/`pyarrow` and validate schema.
- **Audio standardization module**: Functions to resample audio to 16kHz, normalize loudness, and handle mono conversion.
- **Frame generator**: Utility to chunk waveforms into fixed-duration frames (e.g., 20ms) and map sample indices ↔ time.
- **Frame-level label builder**: Module that converts `start_time` / `end_time` / `stutter_type` into frame-level class IDs (0=Fluent, 1=Block, 2=Prolongation, 3=Repetition) using a BIO-like or simple tagging scheme.
- **Dataset serialization**: Save processed examples to disk (e.g., Arrow/Parquet/JSONL) in a format directly consumable by Hugging Face `datasets`.
- **Dataset loader**: Reusable code to load the processed dataset, perform train/val/test splits, and apply basic filtering (min/max duration, corrupted audio, etc.).

### 2. Core Detection & Localization Model

- **Model configuration**: Config files/classes specifying Wav2Vec2 backbone choice (`facebook/wav2vec2-base` or `wav2vec2-xls-r-300m`), number of labels, label mapping, frame hop size, and training hyperparameters.
- **Token classification model wrapper**: Code to instantiate `Wav2Vec2ForTokenClassification` with custom label mappings and optional freezing of lower layers.
- **Collator & preprocessing**: Data collator to batch variable-length audio, apply padding, and align frame-level labels with model output frames.
- **Class-imbalance handling**: Logic to compute class frequencies and pass class weights into `CrossEntropyLoss` (or use sampling strategies).
- **Training script**: End-to-end training entry point (e.g., `train_token_classifier.py`) using `transformers.Trainer` or a custom loop, with logging (TensorBoard/W&B) and checkpointing.
- **Evaluation metrics**: Functions to compute per-frame and per-event metrics (precision/recall/F1 for each stutter type, confusion matrices, overall accuracy) and to aggregate over utterances.
- **Post-processing for events**: Logic to convert per-frame predictions into contiguous time segments with labels (merge, thresholding, minimum duration filters).

### 3. Experimentation & Analysis

- **Exploratory notebooks**: Jupyter notebooks for inspecting raw and processed data, label distributions, and sample waveforms with overlaid labels.
- **Error analysis tools**: Scripts to visualize detection errors (false positives/negatives) on timelines and to inspect representative audio segments.
- **Model comparison utilities**: Simple framework to compare different backbones, hyperparameters, and frame sizes with consistent metrics.

### 4. Rehabilitation (ASR + TTS) Module

- **ASR integration**: Wrapper around Whisper (via `openai-whisper` or Hugging Face) to transcribe audio segments, with configuration for model size and decoding parameters.
- **Text cleaning component**: Lightweight LLM or rule-based module to normalize dysfluent text (e.g., remove repetitions/prolongations) into fluent sentences.
- **TTS integration**: Wrapper for Coqui TTS or VITS to generate fluent speech audio from cleaned text, with configurable voice and sampling rate.
- **Segment routing logic**: Functions that, given detected stutter segments + context, extract corresponding audio chunks, pass them through ASR → text cleaner → TTS, and return synthesized audio plus metadata.
- **Quality checks**: Simple heuristics or user-configurable thresholds to avoid triggering rehab on low-confidence detections.

### 5. Backend API & Services

- **Backend framework setup**: FastAPI (or similar) application skeleton with structured routing and config management.
- **Model serving layer**: Startup hooks to load the Wav2Vec2 model, ASR, and TTS models into memory and expose them through dependency-injected services.
- **Detection API endpoints**:
- Endpoint to accept an audio file/stream, run detection, and return frame-level and segment-level predictions with timestamps.
- **Rehabilitation API endpoints**:
- Endpoint to trigger the rehab pipeline for a given audio file or detected segment list, returning fluent audio and alignment metadata.
- **Combined pipeline endpoint**: High-level endpoint that performs input → detection → optional rehab and returns both raw predictions and corrected audio.
- **Monitoring & logging**: Basic structured logging, latency tracking, and error handling for all endpoints.

### 6. Frontend / User Interface

- **Web client setup**: SPA (e.g., React/Vue) or simple web frontend scaffold to interact with the backend.
- **Audio input UI**: Components for microphone recording and/or file upload, with basic controls (start/stop, re-record, progress bars).
- **Timeline visualization**: Visualization of the audio timeline with color-coded segments for Fluent vs different stutter types and hover tooltips showing timestamps and labels.
- **Real-time-ish feedback**: UI hooks to display incoming detection results progressively (if supported) and highlight stutter regions.
- **Rehab interaction UI**: Controls for the user to play back the fluent audio, compare original vs corrected segments, and optionally replay or discard suggestions.
- **Session management**: Minimal persistence of recent recordings, results, and metadata to allow side-by-side comparison and history.

### 7. Infrastructure, Packaging, and Deployment

- **Environment definition**: `requirements.txt` or `pyproject.toml` specifying core dependencies (PyTorch, transformers, datasets, FastAPI, Whisper, TTS library, etc.).
- **Config management**: Centralized configuration (YAML/env-based) for model paths, thresholds, and hardware options (CPU/GPU).
- **Model artifact handling**: Scripts or utilities to download, cache, and version pre-trained and fine-tuned models.
- **Containerization**: Dockerfile to package the backend with models and expose HTTP endpoints.
- **Basic deployment setup**: Instructions/manifests for running the service locally, on a server, or in the cloud (e.g., gunicorn/uvicorn + reverse proxy).

### 8. Documentation & UX Guidelines

- **Developer docs**: README sections or `docs/` pages describing data preprocessing, training, evaluation, and inference pipelines.
- **API docs**: Auto-generated OpenAPI/Swagger docs plus examples for each backend endpoint.
- **User-facing guide**: Simple documentation describing how a speech therapist or user interacts with the system, what the visualizations mean, and how to interpret feedback.
- **Experiment logs**: A structured changelog or experiment tracking notes linking model versions to datasets and metrics.