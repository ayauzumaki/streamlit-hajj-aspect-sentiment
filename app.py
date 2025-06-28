import streamlit as st
import joblib

st.title("Upload Model dan Prediksi Aspek + Sentimen")

# Upload model aspek
uploaded_model_aspek = st.file_uploader("Upload file model aspek (.pkl)", type=["pkl"])
model_aspek = None
if uploaded_model_aspek is not None:
    try:
        model_aspek = joblib.load(uploaded_model_aspek)
        st.success("Model aspek berhasil diupload dan dimuat!")
    except Exception as e:
        st.error(f"Gagal load model aspek: {e}")

# Upload model sentimen
uploaded_model_sentimen = st.file_uploader("Upload file model sentimen (.pkl)", type=["pkl"])
model_sentimen = None
if uploaded_model_sentimen is not None:
    try:
        model_sentimen = joblib.load(uploaded_model_sentimen)
        st.success("Model sentimen berhasil diupload dan dimuat!")
    except Exception as e:
        st.error(f"Gagal load model sentimen: {e}")

st.markdown("---")

# Input teks user
user_input = st.text_area("Masukkan teks ulasan:")

if st.button("Prediksi"):

    if model_aspek is None or model_sentimen is None:
        st.warning("Silakan upload kedua model terlebih dahulu!")
    elif not user_input.strip():
        st.warning("Masukkan teks terlebih dahulu!")
    else:
        # Fungsi prediksi aspek, contoh asumsikan model_aspek punya method predict_proba
        def predict_aspect(text):
            X = [text]
            probs = model_aspek.predict_proba(X)[0]  # misal keluaran array probabilitas
            classes = model_aspek.classes_            # list nama kelas aspek
            max_idx = probs.argmax()
            aspek_terpilih = classes[max_idx]
            return aspek_terpilih

        # Fungsi prediksi sentimen per aspek
        def predict_sentiment(text, aspek):
            X = [text]
            pred = model_sentimen.predict(X)[0]
            return pred

        # Prediksi aspek
        aspek = predict_aspect(user_input)
        st.write(f"**Aspek terdeteksi:** {aspek}")

        # Prediksi sentimen untuk aspek yang terdeteksi
        sentimen = predict_sentiment(user_input, aspek)
        st.write(f"**Sentimen:** {sentimen}")