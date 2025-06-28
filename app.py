import streamlit as st
import torch
import torch.nn.functional as F

st.title("Upload Model PyTorch dan Prediksi Aspek + Sentimen")

# Upload model aspek (.pt)
uploaded_model_aspek = st.file_uploader("Upload file model aspek (.pt)", type=["pt"])
model_aspek = None
if uploaded_model_aspek is not None:
    try:
        model_aspek = torch.load(uploaded_model_aspek)
        model_aspek.eval()
        st.success("Model aspek berhasil diupload dan dimuat!")
    except Exception as e:
        st.error(f"Gagal load model aspek: {e}")

# Upload model sentimen (.pt)
uploaded_model_sentimen = st.file_uploader("Upload file model sentimen (.pt)", type=["pt"])
model_sentimen = None
if uploaded_model_sentimen is not None:
    try:
        model_sentimen = torch.load(uploaded_model_sentimen)
        model_sentimen.eval()
        st.success("Model sentimen berhasil diupload dan dimuat!")
    except Exception as e:
        st.error(f"Gagal load model sentimen: {e}")

st.markdown("---")

user_input = st.text_area("Masukkan teks ulasan:")

# *** Contoh tokenizer sederhana, sesuaikan dengan yang kamu pakai saat training! ***
def simple_tokenizer(text):
    # misal: lowercase, split spasi, ubah jadi index (dummy)
    tokens = text.lower().split()
    # buat tensor dummy: convert tiap kata ke index, contoh:
    word_to_idx = {"petugas":0, "ibadah":1, "transportasi":2, "akomodasi":3, "konsumsi":4, "lainnya":5}
    indices = [word_to_idx.get(w, 6) for w in tokens]  # 6 = unknown token
    return torch.tensor(indices).unsqueeze(0)  # batch size 1

if st.button("Prediksi"):
    if model_aspek is None or model_sentimen is None:
        st.warning("Silakan upload kedua model terlebih dahulu!")
    elif not user_input.strip():
        st.warning("Masukkan teks terlebih dahulu!")
    else:
        try:
            # Tokenize input
            input_tensor = simple_tokenizer(user_input)

            # Prediksi aspek
            with torch.no_grad():
                output_aspek = model_aspek(input_tensor)  # output logits
                probs_aspek = F.softmax(output_aspek, dim=1)
                pred_idx_aspek = torch.argmax(probs_aspek, dim=1).item()

                # Ambil nama kelas aspek, ganti dengan nama kelas sesuai modelmu
                kelas_aspek = ["petugas", "ibadah", "transportasi", "akomodasi", "konsumsi", "lainnya"]
                aspek_terpilih = kelas_aspek[pred_idx_aspek]

                # Prediksi sentimen (misal model sentimen pakai input yang sama)
                output_sentimen = model_sentimen(input_tensor)
                probs_sentimen = F.softmax(output_sentimen, dim=1)
                pred_idx_sentimen = torch.argmax(probs_sentimen, dim=1).item()
                kelas_sentimen = ["negatif", "netral", "positif"]
                sentimen_terpilih = kelas_sentimen[pred_idx_sentimen]

            st.write(f"**Aspek terdeteksi:** {aspek_terpilih}")
            st.write(f"**Sentimen:** {sentimen_terpilih}")
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")
