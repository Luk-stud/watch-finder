# DINO ViT Watch Embedding Generator

This system generates embeddings for watch images using the **DINO ViT (Vision Transformer)** model, replacing the current CLIP + text embedding system with a single, more powerful visual model.

## 🎯 Overview

### **Why DINO ViT?**

**DINO (Self-Distillation with No Labels)** offers several advantages over CLIP:

- **🦕 Self-supervised learning** - No labels needed, learns from image structure
- **🎯 Better visual features** - Excellent at capturing fine-grained visual details
- **🔄 More robust** - Better handling of image variations and lighting
- **⚡ Single model** - No need for text + image combination
- **📏 Direct 768D output** - Matches current system dimensionality

### **Current vs DINO System:**

| **Aspect** | **Current System** | **DINO System** |
|------------|-------------------|-----------------|
| **Models** | CLIP ViT-B/32 + OpenAI Text | DINO ViT-B/14 only |
| **Dimensions** | 512D + 1536D → PCA → 768D | 768D direct |
| **Complexity** | Two models + PCA reduction | Single model |
| **Visual Quality** | Good | Excellent |
| **Text Understanding** | Yes | No (visual only) |

## 🚀 Quick Start

### **1. Install Dependencies**

```bash
pip install -r dino_requirements.txt
```

### **2. Generate DINO Embeddings**

**Test with 50 watches:**
```bash
python generate_dino_embeddings.py 50
```

**Generate for all watches:**
```bash
python generate_dino_embeddings.py
```

### **3. Update Backend**

```bash
python update_backend_for_dino.py
```

## 📊 Current Setup Analysis

### **Data Structure:**

```
production_scrape_20250601_175426/
├── data/
│   └── final_scrape.csv          # 2,028 watches metadata
└── images/
    ├── Aevig_Huldra_main.jpg     # 1,452 watch images
    ├── Aevig_Thor_main.jpg
    └── ...
```

### **Current Backend:**
- **500 watches** (limited by text embedding generation)
- **768D embeddings** (CLIP 384D + Text 384D)
- **SimpleSgdEngine** for recommendations

### **DINO Target:**
- **2,028 watches** (full production scrape)
- **768D embeddings** (DINO ViT-B/14 direct)
- **Same SimpleSgdEngine** compatibility

## 🔧 Technical Details

### **DINO Model Configuration:**

```python
model_name = "dinov2_vitb14"    # DINO v2 ViT-B/14
output_dim = 768               # Match current system
image_size = 224               # Standard ViT input size
```

### **Image Processing:**

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### **Embedding Generation:**

```python
# Use CLS token as image representation
features = model.forward_features(image_tensor)
embedding = features['x_norm_clstoken'].cpu().numpy().flatten()
```

## 📁 File Structure

### **Generated Files:**

```
dino_embeddings/
├── watch_dino_embeddings.pkl      # DINO embedding matrix
├── watch_dino_metadata.pkl        # Enhanced watch metadata
└── dino_embedding_summary.pkl     # Generation summary

watch_finder_v2/backend/data/
├── watch_dino_embeddings.pkl      # Backend-ready embeddings
├── watch_dino_metadata.pkl        # Backend-ready metadata
├── precomputed_embeddings_dino.pkl # DINO precomputed file
└── precomputed_embeddings.pkl     # Active embeddings (after update)
```

## 🔄 Migration Process

### **Step 1: Generate DINO Embeddings**
```bash
# Test with small subset
python generate_dino_embeddings.py 50

# Generate for all watches
python generate_dino_embeddings.py
```

### **Step 2: Update Backend**
```bash
python update_backend_for_dino.py
```

### **Step 3: Restart Backend**
```bash
# Railway will automatically restart with new embeddings
git add .
git commit -m "feat: switch to DINO ViT embeddings"
git push origin feature/linucb-experts
```

## 📈 Performance Comparison

### **Expected Improvements:**

| **Metric** | **Current (CLIP+Text)** | **DINO ViT** | **Improvement** |
|------------|------------------------|--------------|-----------------|
| **Watches** | 500 | 2,028 | **4x more** |
| **Models** | 2 (CLIP + OpenAI) | 1 (DINO) | **50% simpler** |
| **Visual Quality** | Good | Excellent | **Better features** |
| **Processing Time** | ~45min | ~15min | **3x faster** |
| **Memory Usage** | High | Lower | **More efficient** |

### **Quality Improvements:**

- **🎨 Better visual similarity** - DINO excels at visual feature matching
- **🔍 Fine-grained details** - Better at distinguishing similar watches
- **💡 Self-supervised** - Learns from image structure, not labels
- **🔄 Robust to variations** - Better handling of lighting, angles, etc.

## 🛠️ Customization

### **Different DINO Models:**

```python
# Available DINO models
"dinov2_vitb14"     # ViT-B/14 (768D) - Recommended
"dinov2_vitl14"     # ViT-L/14 (1024D) - Larger, better
"dinov2_vits14"     # ViT-S/14 (384D) - Smaller, faster
```

### **Custom Image Size:**

```python
generator = DINOEmbeddingGenerator(
    model_name="dinov2_vitb14",
    output_dim=768,
    image_size=224  # Can be 224, 384, or 512
)
```

## 🔍 Troubleshooting

### **Common Issues:**

**1. DINO Model Not Found:**
```bash
pip install torch torchvision
# DINO is loaded from torch hub automatically
```

**2. Out of Memory:**
```python
# Use smaller batch or model
generator = DINOEmbeddingGenerator(
    model_name="dinov2_vits14",  # Smaller model
    output_dim=384
)
```

**3. Missing Images:**
```bash
# Check image paths
ls production_scrape_20250601_175426/images/ | head -10
```

### **Debug Mode:**

```python
# Add debug logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Next Steps

### **After DINO Migration:**

1. **Test with full dataset** - 2,028 watches instead of 500
2. **Fine-tune parameters** - Optimize for watch similarity
3. **Add data augmentation** - Improve robustness
4. **Consider ensemble** - Combine DINO with other models

### **Advanced Features:**

- **Multi-scale DINO** - Use different image sizes
- **DINO + CLIP hybrid** - Best of both worlds
- **Fine-tuned DINO** - Train on watch-specific data
- **DINO for text** - Use DINO for text embeddings too

## 📚 References

- **DINO Paper**: [Self-supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- **DINO v2**: [Improved DINO for Visual Features](https://arxiv.org/abs/2304.07193)
- **ViT Architecture**: [Vision Transformer](https://arxiv.org/abs/2010.11929)

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Verify data paths and dependencies
4. Test with a small subset first

---

**🎉 DINO ViT provides a significant upgrade to your watch recommendation system!** 