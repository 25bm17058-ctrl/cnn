import tensorflow as tf
from tensorflow.keras import layers

img_size=(160,160)


train_ds = tf.keras.utils.image_dataset_from_directory("dataset/train", image_size=img_size, label_mode='binary')
testing_ds = tf.keras.utils.image_dataset_from_directory("dataset/validation", image_size=img_size, label_mode='binary')
print("Training dataset size:", len(train_ds))
print("Testing dataset size:", len(testing_ds))

base=tf.keras.applications.MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
base.trainable=False
model=tf.keras.Sequential([
    layers.Rescaling(1./127.5, input_shape=img_size + (3,)),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=testing_ds, epochs=5)
model.save("cat_dog_classify.keras")
print("Model saved as cat_dog_classify.keras")