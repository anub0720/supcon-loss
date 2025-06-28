
# Balanced Image Classification with Enhanced Supervised Contrastive Learning on an Imbalanced dataset

This repository implements a full image classification pipeline for imbalanced datasets. It includes:

- Dataset balancing through augmentation  
- Representation learning using enhanced supervised contrastive learning  
- Classification with transfer learning and focal loss  
- A complete inference pipeline for deployment  

---

## 📊 Dataset Balancing

**Goal**: Ensure 7 image classes (`bus`, `car`, `cat`, `dog`, `cricket`, `football`, `product`) each have 800 samples.

### Steps:

- **Resize** all images to `256 × 256` pixels.
- **Downsample** classes with ≥800 images (randomly select 800).
- **Upsample** classes with <800 images:
  - Duplicate existing images
  - Apply simple augmentations:
    - Horizontal flip (50% chance)
    - ±20° rotation
    - Color jitter (±20% brightness, contrast, saturation)

### Final Class Distribution:

| Class    | Original → Final |
|----------|------------------|
| Bus      | 542 → 800        |
| Car      | 469 → 800        |
| Cat      | 500 → 800        |
| Dog      | 542 → 800        |
| Cricket  | 90 → 800         |
| Football | 100 → 800        |
| Product  | 800 → 800        |

---

## 🔍 Embedding Generation with Enhanced Supervised Contrastive Learning

Learn robust and balanced image embeddings using PyTorch Lightning with custom loss and data handling.

### Key Components:

#### 1. `EnhancedSupConLoss`

- Temperature scaling  
- Class re-weighting:  
  \( w_c = \frac{\max_{c'} N_{c'}}{N_c} \)  
- Margin-augmented hard negative mining  
- Core loss function includes top-*k* hard negatives

#### 2. `AdvancedSupConDataset`

- Strong augmentations (crop, color jitter, blur, etc.)
- Generates multiple views per image
- Computes class weights for sampler and loss

#### 3. `EnhancedEncoder`

- EfficientNet as the backbone (outperforms ResNet for this task)
- Optional attention mechanism:
  \( a = \sigma(W_2 \text{ReLU}(W_1 f)) \)
- MLP projection head with ℓ2 normalization

#### 4. `AdvancedSupConModule`

- Handles contrastive loss training
- Logs metrics and manages optimizer (AdamW + CosineAnnealingWarmRestarts)

#### 5. `AdvancedSupConDataModule`

- Stratified train/val split
- WeightedRandomSampler for balanced training batches

#### 6. `train_contrastive_model()`

- Orchestrates data, model, logging, checkpointing, and training

---

## 🧠 Classification with MLP and Focal Loss

Train an MLP classifier using frozen SupCon embeddings.

### Features:

#### 1. Focal Loss

- Designed for imbalanced datasets  
- Formula:  
  \( FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) \)  
- Focuses training on hard samples

#### 2. Classifier Architecture

- Frozen SupCon encoder → MLP Head:
## This implementation achieved an F1 score of 0.9917 over 1400 images of the hackathon dataset and our team became First Runner up.
