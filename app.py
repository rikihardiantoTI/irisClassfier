import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# 1. Mengatur judul halaman aplikasi
st.title("🌸 Iris Flower Classification App")

# 2. Membuat sidebar untuk input parameter dari pengguna
st.sidebar.header("Input Features")

def user_input_features():
    # Menyesuaikan nilai min, max, dan default sesuai dengan gambar di modul
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.30, 7.90, 5.04)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.00, 4.40, 2.53)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.00, 6.90, 5.26)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.10, 2.50, 1.20)
    
    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# 3. Memuat dataset Iris bawaan dari Scikit-Learn
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# 4. Membangun model Machine Learning (Random Forest)
# Modul menyebutkan penggunaan metode Random Forest untuk klasifikasi bunga iris
clf = RandomForestClassifier()
clf.fit(X, Y)

# 5. Melakukan Prediksi berdasarkan input pengguna
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# 6. Menampilkan Hasil Prediksi
st.subheader("Prediction Result")
# Menampilkan nama spesies berdasarkan hasil prediksi
species_names = iris.target_names
st.success(f"🌷 Species: **{species_names[prediction[0]]}**")

# 7. Menampilkan Probabilitas Prediksi menggunakan Bar Chart
st.subheader("Prediction Probabilities")
proba_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.bar_chart(proba_df.T)