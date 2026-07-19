import tensorflow as tf
model = tf.keras.models.load_model("cat_dog_classify.keras")
image_path = ("cat.jpg")
img = tf.keras.utils.load_img(image_path, target_size=(160, 160))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
predictions=model.predict(img_array)
if predictions[0][0] > 0.5:
    print("Predicted: Dog")
else:
    print("Predicted: Cat")