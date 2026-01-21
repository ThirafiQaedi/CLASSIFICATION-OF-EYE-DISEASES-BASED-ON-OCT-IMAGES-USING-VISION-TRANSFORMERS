import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image


st.set_page_config(page_title="OCT Eye Disease Classification (ViT)", layout="centered")
IMG_SIZE = 224
MODEL_PATH = "final_vittiny_model_16p_811.keras"

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

THERAPY_RECOMMENDATION = {
    "CNV": [
        "Injeksi intravitreal anti-VEGF sebagai terapi utama CNV aktif (terutama pada AMD neovaskular dan CNV miopia).",
        "Monitoring respon terapi secara berkala melalui pemeriksaan klinis dan pencitraan, karena sering memerlukan terapi berulang atau penyesuaian protokol.",
        "Pemantauan efek samping dan pengawasan klinis diperlukan pada terapi anti-VEGF."
    ],
    "DME": [
        "Anti-VEGF intravitreal sebagai terapi kunci pada DME untuk perbaikan fungsi penglihatan.",
        "Kortikosteroid intravitreal dapat dipertimbangkan pada DME persisten atau kurang respons, dengan monitoring tekanan intraokular dan risiko katarak.",
        "Laser fotokoagulasi umumnya sebagai adjuvan pada kondisi tertentu.",
        "Vitrektomi dipertimbangkan bila ada indikasi bedah seperti komponen traksi vitreomakular.",
        "Kontrol faktor sistemik seperti gula darah, tekanan darah, dan lipid sebagai bagian tata laksana komprehensif."
    ],
    "DRUSEN": [
        "Suplemen AREDS/AREDS2 pada kondisi AMD tertentu, terutama intermediate AMD atau drusen besar, untuk memperlambat progresi.",
        "Modifikasi gaya hidup dan pemantauan berkala untuk deteksi progresi.",
        "Jika berkembang menjadi AMD neovaskular dengan CNV, terapi beralih ke anti-VEGF."
    ],
    "NORMAL": [
        "Tidak ada terapi spesifik berdasarkan hasil klasifikasi.",
        "Kontrol rutin sesuai kondisi klinis dan faktor risiko pasien."
    ]
}

# =========================
# CUSTOM LAYERS UNTUK LOAD MODEL VIT
# =========================
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        b = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        d = patches.shape[-1]
        return tf.reshape(patches, [b, -1, d])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.proj = layers.Dense(embed_dim)
        self.pos_emb = layers.Embedding(num_patches, embed_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=tf.shape(patches)[1])
        return self.proj(patches) + self.pos_emb(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches, "embed_dim": self.embed_dim})
        return cfg


# =========================
# LOAD MODEL 
# =========================
@st.cache_resource
def load_model():
    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder},
        compile=False
    )
    return model


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)  # (1,224,224,3)
    return x


def predict_top1(model, x: np.ndarray):
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
    conf = float(probs[idx])
    return label, conf


# =========================
# UI
# =========================
st.title("Klasifikasi Penyakit Mata OCT dengan ViT")
st.write("Upload 1 gambar Mata OCT. Sistem akan menampilkan prediksi TOP-1 dan rekomendasi terapi berbasis kelas.")
st.info("Catatan: Ini sistem pendukung keputusan. Bukan pengganti diagnosis dokter.", icon="ℹ️")

# Load model 
with st.spinner("Memuat model..."):
    model = load_model()

uploaded = st.file_uploader("Upload gambar OCT", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])

if uploaded is not None:
    pil_img = Image.open(uploaded)
    st.image(
        pil_img,
        caption="Gambar OCT yang diunggah"
)

    x = preprocess_image(pil_img)
    label, conf = predict_top1(model, x)

    st.subheader("Hasil Prediksi")
    st.success(f"Prediksi: {label} | Confidence: {conf:.4f}")

    st.subheader("Rekomendasi Terapi")
    recs = THERAPY_RECOMMENDATION.get(label, [])
    if recs:
        for i, r in enumerate(recs, 1):
            st.write(f"{i}. {r}")
    else:
        st.write("Tidak ada rekomendasi yang tersedia untuk kelas ini.")

    st.caption("Gunakan hasil ini sebagai informasi tambahan. Tetapkan keputusan klinis melalui evaluasi dokter mata.")
