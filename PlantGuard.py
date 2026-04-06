import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyDKqEmYY8y3wOC82Yv4WDzneWUPcCxl94c")
model_gemini = genai.GenerativeModel('gemini-2.5-flash')

# ImageClassificationBase class
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))

# Conv Block
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# ResNet9 Architecture
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
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

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Prediction
def predict_image(img, model):
    img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = ResNet9(in_channels=3, num_diseases=len(class_names))
model.load_state_dict(torch.load("plant-disease-model.pth", map_location=device))
model.to(device)
model.eval()

# ===== Streamlit UI =====
st.set_page_config(
    page_title="PlantGuard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        border-radius: 10px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
    }
    .upload-text {
        font-size: 1.2rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1.5rem;
        border-radius: 5px;
        background-color: #e8f5e9;
        border-left: 5px solid #4CAF50;
        font-size: 1.1rem;
        color: #1b5e20;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1.5rem;
        border-radius: 5px;
        background-color: #e3f2fd;
        border-left: 5px solid #2196F3;
        font-size: 1.1rem;
        color: #0d47a1;
        margin: 1rem 0;
        line-height: 1.6;
    }
    h3 {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
    .result-text {
        font-size: 1.2rem;
        font-weight: 500;
        color: #1b5e20;
    }
    .treatment-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #0d47a1;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize default language
if 'language' not in st.session_state:
    st.session_state.language = 'English'
lang = st.session_state.language

# Translation UI strings
UI_TRANSLATIONS = {
    "English": {
        "title": "🌿 PlantGuard",
        "upload_text": "Upload a plant leaf image to detect diseases and get instant care recommendations.",
        "language_settings": "Language Settings",
        "choose_language": "Choose your preferred language",
        "statistics": "Statistics",
        "diseases_detectable": "Diseases Detectable",
        "accuracy": "Accuracy",
        "quick_tips": "Quick Tips",
        "tips_content": "• Ensure good lighting\n• Center the leaf in image\n• Use clear, focused shots",
        "upload_image": "Upload Image",
        "analysis_results": "Analysis Results",
        "analyzing": "Analyzing your plant...",
        "detected_condition": "Detected Condition:",
        "treatment_guide": "Treatment & Care Guide",
        "footer": "Powered by PlantGuard AI • Protecting Plants, Empowering Farmers"
    },
    "Hindi": {
        "title": "🌿 प्लांटगार्ड",
        "upload_text": "रोगों का पता लगाने और तत्काल देखभाल की सिफारिशें प्राप्त करने के लिए पौधे की पत्ती की छवि अपलोड करें।",
        "language_settings": "भाषा सेटिंग्स",
        "choose_language": "अपनी पसंदीदा भाषा चुनें",
        "statistics": "आंकड़े",
        "diseases_detectable": "पता लगाने योग्य रोग",
        "accuracy": "सटीकता",
        "quick_tips": "त्वरित सुझाव",
        "tips_content": "• अच्छी रोशनी सुनिश्चित करें\n• पत्ती को छवि के केंद्र में रखें\n• स्पष्ट, फोकस की गई तस्वीरें उपयोग करें",
        "upload_image": "छवि अपलोड करें",
        "analysis_results": "विश्लेषण परिणाम",
        "analyzing": "आपके पौधे का विश्लेषण किया जा रहा है...",
        "detected_condition": "पता लगाई गई स्थिति:",
        "treatment_guide": "उपचार और देखभाल गाइड",
        "footer": "प्लांटगार्ड AI द्वारा संचालित • पौधों की रक्षा, किसानों का सशक्तिकरण"
    },
    "Marathi": {
        "title": "🌿 प्लांटगार्ड",
        "upload_text": "रोग शोधण्यासाठी आणि त्वरित काळजी सूचना मिळवण्यासाठी वनस्पतीच्या पानाची प्रतिमा अपलोड करा.",
        "language_settings": "भाषा सेटिंग्ज",
        "choose_language": "तुमची पसंतीची भाषा निवडा",
        "statistics": "आकडेवारी",
        "diseases_detectable": "शोधू शकणारे रोग",
        "accuracy": "अचूकता",
        "quick_tips": "जलद टिप्स",
        "tips_content": "• चांगला प्रकाश असल्याची खात्री करा\n• पान प्रतिमेच्या मध्यभागी ठेवा\n• स्पष्ट, फोकस केलेले शॉट्स वापरा",
        "upload_image": "प्रतिमा अपलोड करा",
        "analysis_results": "विश्लेषण निकाल",
        "analyzing": "तुमच्या वनस्पतीचे विश्लेषण केले जात आहे...",
        "detected_condition": "आढळलेली स्थिती:",
        "treatment_guide": "उपचार आणि काळजी मार्गदर्शक",
        "footer": "प्लांटगार्ड AI द्वारे संचालित • वनस्पतींचे संरक्षण, शेतकऱ्यांचे सक्षमीकरण"
    }
}

# Sidebar
with st.sidebar:
    st.image("logo.png", width=150)
    st.markdown("---")
    st.markdown(f"### 🌐 {UI_TRANSLATIONS[st.session_state.language]['language_settings']}")
    selected_lang = st.selectbox(
        UI_TRANSLATIONS[st.session_state.language]['choose_language'],
        ["English", "Hindi", "Marathi"],
        index=["English", "Hindi", "Marathi"].index(st.session_state.language),
        key="lang_selector"
    )
    
    # Update session state when language changes
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()

    st.markdown("---")
    st.markdown(f"### 📊 {UI_TRANSLATIONS[st.session_state.language]['statistics']}")
    st.metric(label=UI_TRANSLATIONS[st.session_state.language]['diseases_detectable'], value="38")
    st.metric(label=UI_TRANSLATIONS[st.session_state.language]['accuracy'], value="96%")
    st.markdown("---")
    st.markdown(f"### 💡 {UI_TRANSLATIONS[st.session_state.language]['quick_tips']}")
    st.info(UI_TRANSLATIONS[st.session_state.language]['tips_content'])

# Main content
st.title(UI_TRANSLATIONS[st.session_state.language]['title'])
st.markdown(f"<p class='upload-text'>{UI_TRANSLATIONS[st.session_state.language]['upload_text']}</p>", unsafe_allow_html=True)

# Create two columns for upload and preview
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"### 📸 {UI_TRANSLATIONS[st.session_state.language]['upload_image']}")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption=UI_TRANSLATIONS[st.session_state.language]['upload_image'], use_container_width=True)
    
    with col2:
        st.markdown(f"### 🔍 {UI_TRANSLATIONS[st.session_state.language]['analysis_results']}")
        with st.spinner(UI_TRANSLATIONS[st.session_state.language]['analyzing']):
            label = predict_image(image, model)
            st.markdown(
                f"""<div class='success-box'>
                    ✅ <span class='result-text'>{UI_TRANSLATIONS[st.session_state.language]['detected_condition']}</span><br>
                    <strong style='font-size: 1.3rem;'>{label.replace('_', ' ')}</strong>
                </div>""", 
                unsafe_allow_html=True
            )
            
            # Generate explanation and solution
            eng_prompt = f"Give a simple explanation and solution for this plant disease: {label.replace('_', ' ')}"
            response = model_gemini.generate_content(eng_prompt)
            
            # Handle translation
            if lang in ["Hindi", "Marathi"]:
                translation_prompt = f"Translate the following text to {lang}:\n\n{response.text}"
                translated = model_gemini.generate_content(translation_prompt)
                final_response = translated.text
            else:
                final_response = response.text

            st.markdown(f"### 🧪 {UI_TRANSLATIONS[st.session_state.language]['treatment_guide']}")
            st.markdown(
                f"""<div class='info-box'>
                    <span class='treatment-text'>{final_response}</span>
                </div>""", 
                unsafe_allow_html=True
            )

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #666;'>{UI_TRANSLATIONS[st.session_state.language]['footer']}</div>",
    unsafe_allow_html=True
)
