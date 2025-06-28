import streamlit as st
import torch
import os
import zipfile
import gdown

# --- Fungsi download ZIP model dari Google Drive
@st.cache_resource
def download_models():
    ZIP_ID = "1so8pb-lcxxC5M7pD0qIFb4PTaXPzdmL7"  # Ganti dengan ID Google Drive milikmu
    ZIP_FILE = "models.zip"

    # Unduh file ZIP dari Google Drive
    gdown.download(id=ZIP_ID, output=ZIP_FILE, quiet=False)

    # Ekstrak ZIP ke folder "models"
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall("models")

    # Hapus file ZIP
    os.remove(ZIP_FILE)

    return True

# --- Cek apakah model sudah ada, kalau belum download
if not os.path.exists("models/aspek/petugas.pt"):
    st.info("Mengunduh model...")
    download_models()

# --- Daftar aspek
aspek_list = ["petugas", "ibadah", "akomodasi", "konsumsi", "transportasi", "lainnya"]

# --- Load semua model aspek
model_aspek_dict = {
    aspek: torch.load(f"models/aspek/{aspek}.pt", map_location=torch.device('cpu')) for aspek in aspek_list
}
# --- Load semua model sentimen
model_sentimen_dict = {
    aspek: torch.load(f"models/sentimen/{aspek}.pt", map_location=torch.device('cpu')) for aspek in aspek_list
}

# --- Set semua model ke eval mode
for model in model_aspek_dict.values():
    model.eval()
for model in model_sentimen_dict.values():
    model.eval()

# --- UI Streamlit
st.title("Prediksi Aspek dan Sentimen Ulasan")

user_input = st.text_area("Masukkan teks ulasan:")

if st.button("Prediksi"):
    if not user_input.strip():
        st.warning("Tolong masukkan teks ulasan terlebih dahulu.")
    else:
        st.subheader("Hasil Prediksi:")
        aspek_terdeteksi = []

        for aspek in aspek_list:
            model = model_aspek_dict[aspek]
            with torch.no_grad():
                # Ubah bagian ini sesuai input tensor yang cocok dengan modelmu
                pred = model.predict([user_input])[0]  # sesuaikan jika pakai tokenizer/vectorizer sendiri
                if pred == 1:
                    aspek_terdeteksi.append(aspek)

        if not aspek_terdeteksi:
            st.info("Tidak ada aspek terdeteksi.")
        else:
            for aspek in aspek_terdeteksi:
                st.markdown(f"**Aspek:** {aspek.capitalize()}")
                model_sent = model_sentimen_dict[aspek]
                with torch.no_grad():
                    sent_pred = model_sent.predict([user_input])[0]
                st.markdown(f"→ Sentimen: **{sent_pred}**")
