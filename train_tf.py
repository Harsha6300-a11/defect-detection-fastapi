import tensorflow as tf
import os

DATA_DIR = "dataset/NEU-DET"
TRAIN_DIR = os.path.join(DATA_DIR, "train", "images")
VAL_DIR   = os.path.join(DATA_DIR, "validation", "images")
IMG_SIZE = (200, 200)   
BATCH_SIZE = 32
EPOCHS = 5
MODEL_PATH = "defect_model.h5"
METRICS_PATH = "metrics.txt"


train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"   
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
print("Classes:", class_names)

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))


base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


# Save model
model.save(MODEL_PATH)
print(f"✅ Model saved as {MODEL_PATH}")

# Evaluate on validation set
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

# Save validation accuracy to file
with open("metrics.txt", "w") as f:
    f.write(str(val_acc))
