from flask import Flask, render_template, request, jsonify
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Modeli ve TF-IDF vektörizeri yükle
model = load_model("model.keras")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/tahmin", methods=["POST"])
def tahmin():
    cumle = request.form["cumle"]
    if cumle:
        cumle_tfidf = tfidf_vectorizer.transform([cumle])
        cumle_tfidf.sort_indices()  # Değişiklik burada
        tahmin = model.predict(cumle_tfidf)
        tahmin = tahmin[0]
        tahmin_olasiliklari = [(label, olasilik) for label, olasilik in zip(label_encoder.classes_, tahmin)]
        tahmin_olasiliklari.sort(key=lambda x: x[1], reverse=True)
        en_yuksek_uc = tahmin_olasiliklari[:3]
        tahmin_metni = "\n".join([f"{label}: %{round(olasilik*100, 2)}" for label, olasilik in en_yuksek_uc])
        return jsonify({"tahmin": tahmin_metni})
    else:
        return jsonify({"error": "Lütfen bir cümle girin."})

if __name__ == "__main__":
    app.run(debug=True)

#derse geç kaldığım için hoca ilk derse beni almadı dışarı çıktım sonra 2.derse girebildiim .
#cüzdanımı unuttuğum için eve yürüyerek dönmek zorunda kaldım 
#en sevdiğim yemek bugün indirimdeydi 
#maaşıma zam yapmamışlar ben yeni çalışacak yer bulsam iyi olur 
#doğum günümde böyle güzel sürpriz beklemiyodum 
