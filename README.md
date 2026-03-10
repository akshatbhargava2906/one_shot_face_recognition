# One-Shot Face Recognition using Siamese Neural Networks

A deep learning pipeline for real-time, one-shot face verification built with **TensorFlow/Keras** and **OpenCV**. The system learns to compare pairs of face images using a shared CNN embedding and a custom L1 distance layer — enabling identity verification from as little as one reference image.

---

## Project Overview

| Detail | Value |
|---|---|
| **Task** | One-shot face verification (same/different person) |
| **Architecture** | Siamese Neural Network |
| **Embedding Model** | 4-block CNN → Dense(4096, sigmoid) |
| **Distance Metric** | Custom L1 Distance Layer |
| **Loss Function** | Binary Cross-Entropy |
| **Optimizer** | Adam (lr = 1e-4) |
| **Epochs** | 50 |
| **Framework** | TensorFlow / Keras |

---

## Architecture

```
        Input A (Anchor)          Input B (Validation)
              │                          │
              ▼                          ▼
    ┌─────────────────┐       ┌─────────────────┐
    │  Shared CNN     │       │  Shared CNN     │   ← same weights
    │  Embedding      │       │  Embedding      │
    └────────┬────────┘       └────────┬────────┘
             │                         │
             └──────────┬──────────────┘
                        ▼
               L1 Distance Layer
               |embedding_A - embedding_B|
                        │
                        ▼
               Dense(1, sigmoid)
                        │
                        ▼
               Similarity Score [0, 1]
```

---

## Features

- **One-shot learning** — verifies identity from a single reference image per person
- **Custom L1Dist Layer** — computes absolute element-wise distance between embeddings
- **Shared CNN weights** — both branches of the Siamese network share identical parameters
- **Webcam data collection** — real-time anchor and positive image capture using OpenCV
- **LFW negatives** — leverages the Labeled Faces in the Wild dataset for negative pairs
- **GPU memory growth** — configures TensorFlow to prevent OOM errors during training
- **Checkpoint saving** — model weights saved every 10 epochs for recovery
- **Precision & Recall** evaluation on held-out test set

---

## Dataset

### Negative Images
- **[LFW — Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)**
- Diverse face images from different identities → used as *non-matching* pairs

### Anchor & Positive Images
- Collected via **live webcam** using OpenCV
- Press `a` to capture an **anchor** image
- Press `p` to capture a **positive** image
- Press `q` to quit

```
data/
├── anchor/     ← Reference face images (webcam)
├── positive/   ← Same-person validation images (webcam)
└── negative/   ← Different-person images (LFW)
```

300 images sampled from each class.

---

## Preprocessing

1. **Read** — raw bytes from file path via `tf.io.read_file`
2. **Decode** — JPEG bytes to image tensor via `tf.io.decode_jpeg`
3. **Resize** — to **100 × 100** pixels
4. **Normalize** — pixel values scaled from `[0, 255]` → `[0, 1]`
5. **Label** — `(anchor, positive) → 1` / `(anchor, negative) → 0`
6. **Cache + Shuffle** — buffer_size=1024 for training efficiency
7. **Split** — 70% train / 30% test, batch size=16, prefetch=8

---

## Model Details

### Embedding CNN (`make_embedding`)

| Block | Layer | Filters | Kernel | Activation |
|---|---|---|---|---|
| 1 | Conv2D + MaxPool | 64 | 10×10 | ReLU |
| 2 | Conv2D + MaxPool | 128 | 7×7 | ReLU |
| 3 | Conv2D + MaxPool | 128 | 4×4 | ReLU |
| 4 | Conv2D | 256 | 4×4 | ReLU |
| — | Flatten + Dense | 4096 | — | Sigmoid |

### L1 Distance Layer

```python
class L1Dist(Layer):
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
```

### Siamese Network

```
(anchor_img, validation_img)
        → shared embedding
        → L1 distance
        → Dense(1, sigmoid)
        → similarity score ∈ [0, 1]
```

Threshold: `> 0.5` → same person, `≤ 0.5` → different person

---

## Training

```python
EPOCHS = 50
optimizer = Adam(lr=1e-4)
loss = BinaryCrossentropy()
```

- Custom `@tf.function` training loop for performance
- Checkpoint saved every 10 epochs to `./training_checkpoints/`
- GPU memory growth enabled via `tf.config.experimental.set_memory_growth`

---

## Evaluation

```python
from tensorflow.keras.metrics import Precision, Recall

# Post-processing: threshold at 0.5
predictions = [1 if p > 0.5 else 0 for p in y_hat]

# Metrics
Recall:    m.result().numpy()
Precision: p.result().numpy()
```

---

## Model Saving & Loading

```python
# Save
model.save('siamese_model.h5')

# Load (with custom objects)
model = tf.keras.models.load_model(
    'siamese_model.h5',
    custom_objects={
        'L1Dist': L1Dist,
        'BinaryCrossentropy': tf.losses.BinaryCrossentropy
    }
)
```

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **Deep Learning** | TensorFlow 2.x, Keras Functional API |
| **Computer Vision** | OpenCV (cv2) |
| **Data Pipeline** | `tf.data` (map, cache, shuffle, batch, prefetch) |
| **Visualization** | Matplotlib |
| **Utilities** | UUID, NumPy, OS |

---

## Repository Structure

```
├── face_rec_siamese.ipynb      # Main notebook
├── siamese_model.h5            # Saved model
├── training_checkpoints/       # Model checkpoints (every 10 epochs)
├── data/
│   ├── anchor/                 # Webcam anchor images
│   ├── positive/               # Webcam positive images
│   └── negative/               # LFW negative images
└── README.md
```

---

## Setup & Usage

### 1. Install Dependencies
```bash
pip install tensorflow opencv-python numpy matplotlib
```

### 2. Download LFW Dataset
Download from [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/) and extract into `lfw/`.

### 3. Collect Face Images
Run the webcam cell (set `cv2.VideoCapture(0)`):
- Press `a` → save anchor image
- Press `p` → save positive image
- Press `q` → quit

### 4. Train the Model
Run all cells sequentially in `face_rec_siamese.ipynb`.

### 5. Verify a Face
```python
model = tf.keras.models.load_model('siamese_model.h5', custom_objects={'L1Dist': L1Dist, ...})
result = model.predict([anchor_img, test_img])
print("Match" if result > 0.5 else "No Match")
```

---

## References

1. Koch, G., Zemel, R., & Salakhutdinov, R. (2015). *Siamese Neural Networks for One-Shot Image Recognition.* ICML Deep Learning Workshop.
2. [LFW — Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)
3. [TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)

---
