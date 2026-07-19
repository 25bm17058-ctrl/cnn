from flask import Flask, render_template, request
import tensorflow as tf
import os

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = tf.keras.models.load_model("cat_dog_classify.keras")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Check if an image was uploaded
    if "image" not in request.files:
        return render_template("index.html", prediction="No image selected")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", prediction="No image selected")

    # Save uploaded image
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Load and preprocess image
    img = tf.keras.utils.load_img(filepath, target_size=(160, 160))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Predict
    prediction = model.predict(img_array, verbose=0)

    probability = float(prediction[0][0])

    if probability > 0.5:
        result = "🐶 Dog"
        confidence = probability * 100
    else:
        result = "🐱 Cat"
        confidence = (1 - probability) * 100

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2),
        image=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)