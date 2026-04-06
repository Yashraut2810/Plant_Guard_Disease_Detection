# PlantGuard - Plant Disease Detection Project

## 📋 Project Overview

**PlantGuard** is a deep learning-based plant disease detection system that uses computer vision to identify diseases in plant leaves. The system can classify 38 different plant conditions (including healthy states) across 14 different plant species.

### Key Features:
- **38 disease classes** detection
- **96% accuracy** on validation data
- **Multi-language support** (English, Hindi, Marathi)
- **Web-based interface** using Streamlit
- **AI-powered treatment recommendations** using Google Gemini API

---

## 🗂️ Project Structure

```
PlantDiseaseProject/
├── Kaggle/
│   ├── New Plant Diseases Dataset(Augmented)/
│   │   ├── train/          # Training images (70,295 images)
│   │   └── valid/          # Validation images (~9,673 images)
│   └── test/               # Test images for evaluation
├── plant_disease_detection.ipynb  # Main training notebook
├── model_test.ipynb               # Model evaluation notebook
├── PlantGuard.py                  # Streamlit web application
├── plant-disease-model.pth        # Trained model weights
├── plant-disease-model-complete.pth # Complete model file
├── model.onnx                     # ONNX format model
├── class_labels.csv               # Class index to disease name mapping
└── README.md                      # Project description
```

---

## 🔍 Detailed Code Explanation

### 1. **Data Preparation & Exploration** (`plant_disease_detection.ipynb`)

#### **Import Libraries** (Cells 1-2)
```python
import torch, torchvision
import numpy as np, pandas as pd
from PIL import Image
```
- **Purpose**: Imports essential libraries for deep learning, data processing, and visualization
- **Key Libraries**:
  - `torch`: PyTorch for neural networks
  - `torchvision`: Pre-built datasets and transforms
  - `PIL`: Image processing
  - `matplotlib`: Data visualization

#### **Dataset Loading** (Cells 5-13)
```python
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
diseases = os.listdir(train_dir)
```
- **Purpose**: Loads and explores the dataset structure
- **Dataset Statistics**:
  - **38 disease classes** (12 healthy + 26 diseases)
  - **14 unique plant species** (Apple, Corn, Tomato, etc.)
  - **70,295 training images**
  - **~9,673 validation images**

#### **Data Visualization** (Cells 14-17)
- Creates bar charts showing image distribution per class
- Helps identify class imbalance issues
- Visualizes sample images from different disease categories

---

### 2. **Model Architecture** (`plant_disease_detection.ipynb`)

#### **Base Class: ImageClassificationBase** (Cell 43)
```python
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        # Calculates loss and accuracy
```
- **Purpose**: Provides common functionality for training and validation
- **Key Methods**:
  - `training_step()`: Calculates loss during training
  - `validation_step()`: Evaluates model on validation data
  - `epoch_end()`: Prints training progress

#### **Convolution Block** (Cell 45)
```python
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)
```
- **Purpose**: Creates reusable convolution blocks
- **Components**:
  - **Conv2d**: 2D convolution (3x3 kernel)
  - **BatchNorm2d**: Normalizes activations (speeds up training)
  - **ReLU**: Activation function
  - **MaxPool2d**: Optional downsampling (reduces image size by 4x)

#### **ResNet9 Architecture** (Cell 45)
```python
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases)
        )
```
- **Purpose**: Custom ResNet-like architecture optimized for plant disease detection
- **Architecture Flow**:
  1. **conv1**: Input (3 channels) → 64 feature maps (256×256)
  2. **conv2**: 64 → 128 feature maps, then pool to 64×64
  3. **res1**: Residual block (128→128) with skip connection
  4. **conv3**: 128 → 256 feature maps, pool to 16×16
  5. **conv4**: 256 → 512 feature maps, pool to 4×4
  6. **res2**: Residual block (512→512) with skip connection
  7. **classifier**: Final pooling, flatten, and fully connected layer (512 → 38 classes)

- **Key Features**:
  - **Residual Connections**: `out = self.res1(out) + out` helps with gradient flow
  - **Progressive Downsampling**: Reduces spatial dimensions while increasing channels
  - **Total Parameters**: ~6.5 million parameters

---

### 3. **Training Process** (`plant_disease_detection.ipynb`)

#### **Device Configuration** (Cells 36-39)
```python
device = get_default_device()  # GPU if available, else CPU
train_dl = DeviceDataLoader(train_dl, device)
```
- **Purpose**: Automatically uses GPU for faster training if available

#### **Data Loading** (Cells 19-32)
```python
train = ImageFolder(train_dir, transform=transforms.ToTensor())
train_dl = DataLoader(train, batch_size=32, shuffle=True)
```
- **Purpose**: Converts images to tensors and batches them
- **Image Preprocessing**:
  - Resize to 256×256 pixels
  - Convert to tensor format (values 0-1)
  - Batch size: 32 images per batch

#### **Training Function: OneCycleLR** (Cell 51)
```python
def fit_OneCycle(epochs, max_lr, model, train_loader, val_loader, ...):
    optimizer = torch.optim.Adam(model.parameters(), max_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(...)
    
    for epoch in range(epochs):
        # Training phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
            scheduler.step()  # Update learning rate
        
        # Validation phase
        result = evaluate(model, val_loader)
```
- **Purpose**: Trains the model using OneCycle learning rate policy
- **Key Components**:
  - **Optimizer**: Adam (adaptive learning rate)
  - **Learning Rate**: Starts low, peaks mid-training, ends low
  - **Gradient Clipping**: Prevents exploding gradients
  - **Weight Decay**: L2 regularization (prevents overfitting)

#### **Training Configuration** (Cell 54)
```python
epochs = 2
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
```
- **Training Results**:
  - **Epoch 0**: 86.36% accuracy, loss: 0.44
  - **Epoch 1**: 99.15% accuracy, loss: 0.027

---

### 4. **Model Evaluation** (`model_test.ipynb`)

#### **Performance Metrics** (Cells 4-10)
```python
# Calculate accuracy per batch
acc = (outputs.argmax(1) == labels).float().mean()

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
```
- **Evaluation Results**:
  - **Average Accuracy**: 99.19%
  - **Average Loss**: 0.0284
- **Analysis Tools**:
  - Confusion matrix (shows classification errors)
  - Classification report (precision, recall, F1-score per class)
  - Smoothed loss/accuracy plots

#### **Visualization** (Cells 5-8)
- **Loss plots**: Monitor training stability
- **Confusion matrix**: Identify misclassified disease pairs
- **Normalized confusion matrix**: See error rates as percentages

---

### 5. **Web Application** (`PlantGuard.py`)

#### **Model Loading** (Lines 109-113)
```python
model = ResNet9(in_channels=3, num_diseases=len(class_names))
model.load_state_dict(torch.load("plant-disease-model.pth"))
model.eval()  # Set to evaluation mode
```
- **Purpose**: Loads the pre-trained model for inference
- **Note**: Model is set to `eval()` mode to disable dropout/batch norm updates

#### **Prediction Function** (Lines 98-104)
```python
def predict_image(img, model):
    img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]
```
- **Purpose**: Classifies a single image
- **Process**:
  1. Convert image to RGB format
  2. Resize to 256×256 and convert to tensor
  3. Add batch dimension (`unsqueeze(0)`)
  4. Get predictions (disable gradient computation for efficiency)
  5. Return class name with highest probability

#### **Streamlit UI Components** (Lines 115-318)

**1. Page Configuration** (Lines 116-121)
```python
st.set_page_config(
    page_title="PlantGuard",
    page_icon="🌿",
    layout="wide"
)
```

**2. Multi-language Support** (Lines 186-238)
- **Languages**: English, Hindi, Marathi
- **Implementation**: Dictionary-based translation system
- **Session State**: Stores selected language across page reloads

**3. Sidebar** (Lines 241-263)
- Language selector
- Statistics (38 diseases, 96% accuracy)
- Quick tips for better image quality

**4. Main Interface** (Lines 269-311)
```python
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    label = predict_image(image, model)
    # Generate treatment recommendations using Gemini AI
```
- **Image Upload**: Accepts JPG, JPEG, PNG formats
- **Real-time Prediction**: Processes image immediately
- **AI Treatment Guide**: Uses Google Gemini API to generate disease explanations and solutions

#### **Google Gemini Integration** (Lines 7-11, 294-303)
```python
genai.configure(api_key="...")
model_gemini = genai.GenerativeModel('gemini-2.5-flash')

# Generate treatment guide
response = model_gemini.generate_content(f"Explain: {label}")

# Translate if needed
if lang in ["Hindi", "Marathi"]:
    translated = model_gemini.generate_content(f"Translate: {response.text}")
```
- **Purpose**: Provides AI-generated treatment recommendations
- **Features**:
  - Explains disease symptoms
  - Suggests treatment methods
  - Translates content to user's preferred language

---

## 🔧 Technical Details

### **Image Preprocessing**
- **Input Size**: 256×256 pixels (RGB)
- **Normalization**: Values scaled to [0, 1]
- **Format**: PyTorch tensor (batch_size × 3 × 256 × 256)

### **Model Architecture Summary**
```
Input: 3 × 256 × 256
  ↓
Conv1: 64 × 256 × 256
  ↓
Conv2 + Pool: 128 × 64 × 64
  ↓
Res1 (skip): 128 × 64 × 64
  ↓
Conv3 + Pool: 256 × 16 × 16
  ↓
Conv4 + Pool: 512 × 4 × 4
  ↓
Res2 (skip): 512 × 4 × 4
  ↓
MaxPool + Flatten: 512
  ↓
Linear: 38 classes
```

### **Training Hyperparameters**
- **Batch Size**: 32
- **Learning Rate**: 0.01 (OneCycle scheduler)
- **Optimizer**: Adam
- **Weight Decay**: 1e-4
- **Gradient Clipping**: 0.1
- **Epochs**: 2 (quick training, high accuracy)

---

## 📊 Dataset Information

### **Plant Species Covered**
1. Apple (4 classes: scab, black rot, rust, healthy)
2. Corn/Maize (4 classes: rust, blight, leaf spot, healthy)
3. Tomato (10 classes: various diseases + healthy)
4. Grape (4 classes)
5. Potato (3 classes)
6. Pepper (2 classes)
7. Cherry (2 classes)
8. Peach (2 classes)
9. Strawberry (2 classes)
10. Orange (1 disease: Citrus greening)
11. Blueberry, Raspberry, Soybean, Squash (healthy + some diseases)

### **Class Distribution**
- **Healthy classes**: 12
- **Disease classes**: 26
- **Total classes**: 38
- **Training images**: ~70,295 (varying per class)
- **Validation images**: ~9,673

---

## 🚀 Usage

### **Running the Web Application**
```bash
streamlit run PlantGuard.py
```

### **Using the Model Programmatically**
```python
from PIL import Image
from PlantGuard import predict_image, model

image = Image.open("plant_leaf.jpg")
disease = predict_image(image, model)
print(f"Detected: {disease}")
```

---

## 🔑 Key Concepts Explained

### **1. Residual Blocks**
- **Problem**: Deep networks suffer from vanishing gradients
- **Solution**: Skip connections allow gradients to flow directly
- **Formula**: `output = F(x) + x` (where F is the convolutional block)

### **2. Batch Normalization**
- **Purpose**: Normalizes activations within each batch
- **Benefits**: Faster training, more stable gradients, allows higher learning rates

### **3. OneCycle Learning Rate**
- **Strategy**: Learning rate starts low, increases to peak, then decreases
- **Benefits**: Better generalization, faster convergence

### **4. Cross-Entropy Loss**
- **Purpose**: Measures difference between predicted and actual class probabilities
- **Use Case**: Multi-class classification (38 classes)

### **5. Transfer Learning vs Custom Architecture**
- **Choice**: Custom ResNet9 (not pre-trained)
- **Reason**: Dataset is specific to plant diseases, custom architecture fits better

---

## 📈 Performance Metrics

- **Training Accuracy**: 99.15% (after 2 epochs)
- **Validation Accuracy**: 99.19%
- **Average Loss**: 0.0284
- **Model Size**: ~6.5M parameters
- **Inference Speed**: Fast (GPU) / Moderate (CPU)

---

## 🔮 Future Improvements

1. **Data Augmentation**: More image transformations during training
2. **Mobile Deployment**: Convert to ONNX/Mobile format
3. **Confidence Scores**: Display prediction confidence
4. **Batch Processing**: Process multiple images at once
5. **More Languages**: Expand translation support
6. **Historical Tracking**: Save prediction history

---

## 📚 Dependencies

```
torch >= 1.0.0
torchvision >= 0.8.0
Pillow >= 8.0.0
streamlit >= 1.0.0
google-generativeai
numpy
pandas
matplotlib
scikit-learn
```

---

This project demonstrates end-to-end machine learning workflow: data exploration, model design, training, evaluation, and deployment in a user-friendly web application.

