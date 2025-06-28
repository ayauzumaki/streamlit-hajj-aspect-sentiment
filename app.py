import streamlit as st
import torch
import os
import zipfile
import requests
from io import BytesIO

# --- Link ZIP dari Google Drive (pastikan public)
ZIP_ID = "1so8pb-lcxxC5M7pD0qIFb4PTaXPzdmL7"
ZIP_URL = f"https://drive.google.com/file/d/1so8pb-lcxxC5M7pD0qIFb4PTaXPzdmL7/view?usp=sharing"

@st.cache_resource
def download_models():
    response = requests.get(ZIP_URL)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("models")
    return True

if not os.path.exists("models/aspek/petugas.pt"):
    st.info("Mengunduh model...")
    download_models()

# Daftar aspek
aspek_list = ["petugas", "ibadah", "akomodasi", "konsumsi", "transportasi", "lainnya"]

# Load semua model aspek
model_aspek_dict = {
    aspek: torch.load(f"models/aspek/{aspek}.pt") for aspek in aspek_list
}
# Load semua model sentimen
model_sentimen_dict = {
    aspek: torch.load(f"models/sentimen/{aspek}.pt") for aspek in aspek_list
}

for m in model_aspek_dict.values():
    m.eval()
for m in model_sentimen_dict.values():
    m.eval()

# UI
st.title("Prediksi Aspek dan Sentimen (Multi-label)")

user_input = st.text_area("Masukkan teks ulasan:")

if st.button("Prediksi"):
    if not user_input.strip():
        st.warning("Tolong isi teks ulasan terlebih dahulu.")
    else:
        st.subheader("Hasil Prediksi:")
        aspek_terdeteksi = []

        for aspek in aspek_list:
            model = model_aspek_dict[aspek]
            with torch.no_grad():
                # Anda bisa ubah representasi input ini sesuai modelmu
                input_tensor = torch.tensor([user_input])  # Placeholder
                pred = model.predict([user_input])[0]  # ganti ini sesuai modelmu
                if pred == 1:  # ya
                    aspek_terdeteksi.append(aspek)

        if not aspek_terdeteksi:
            st.info("Tidak ada aspek terdeteksi.")
        else:
            for aspek in aspek_terdeteksi:
                st.markdown(f"**Aspek:** {aspek.capitalize()}")
                model_sent = model_sentimen_dict[aspek]
                with torch.no_grad():
                    sent_pred = model_sent.predict([user_input])[0]  # ganti sesuai model
                st.markdown(f"→ Sentimen: **{sent_pred}**")
