
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

DATASET_DIR = Path("dataset")
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 123

MODEL_PATH = "wheat_mobilenetv2.keras"
CLASSES_PATH = "mobilenet_class_names.json"

tf.keras.utils.set_random_seed(SEED)

# --------------------------------------------------
# 2. COLLECT IMAGE PATHS AND LABELS
# --------------------------------------------------

class_names = sorted(
    folder.name for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
)

if len(class_names) != 2:
    raise ValueError(
        f"Expected 2 class folders, found: {class_names}"
    )

paths = []
labels = []

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for label, class_name in enumerate(class_names):
    class_folder = DATASET_DIR / class_name

    for path in class_folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            paths.append(str(path))
            labels.append(label)

paths = np.array(paths)
labels = np.array(labels)

print("Classes:", class_names)
print("Total images:", len(paths))

for index, name in enumerate(class_names):
    print(f"{name}: {np.sum(labels == index)}")

# Stratified split keeps class proportions similar.
train_paths, val_paths, train_labels, val_labels = train_test_split(
    paths,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)

print("\nTraining images:", len(train_paths))
print("Validation images:", len(val_paths))

# Save class order for future predictions.
with open(CLASSES_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

# --------------------------------------------------
# 3. LOAD AND PREPROCESS IMAGES
# --------------------------------------------------

def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess_input(image)
    return image, label

def make_dataset(image_paths, image_labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, image_labels)
    )

    if training:
        dataset = dataset.shuffle(
            len(image_paths),
            seed=SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds = make_dataset(val_paths, val_labels)

# --------------------------------------------------
# 4. DATA AUGMENTATION
# --------------------------------------------------

augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
], name="data_augmentation")

# --------------------------------------------------
# 5. BUILD MOBILENETV2 MODEL
# --------------------------------------------------

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

# Initially freeze pretrained feature extractor.
base_model.trainable = False

inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.30)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

# --------------------------------------------------
# 6. TRAIN CLASSIFICATION HEAD
# --------------------------------------------------

print("\nStage 1: Training classification head")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# --------------------------------------------------
# 7. FINE-TUNE LAST PART OF MOBILENETV2
# --------------------------------------------------

print("\nStage 2: Fine-tuning last layers")

base_model.trainable = True

# Freeze most pretrained layers; train only the last 30.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep batch-normalization layers frozen for stability.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=callbacks
)

# Load the best checkpoint saved across training.
model = tf.keras.models.load_model(MODEL_PATH)

# --------------------------------------------------
# 8. EVALUATE ON HELD-OUT VALIDATION SET
# --------------------------------------------------

print("\nFinal validation evaluation")

true_labels = []
predicted_labels = []

for images, labels_batch in val_ds:
    probabilities = model.predict(images, verbose=0)
    predicted_labels.extend(np.argmax(probabilities, axis=1))
    true_labels.extend(labels_batch.numpy())

print("\nClassification report:")
print(classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names)))
))

print("\nSaved model:", MODEL_PATH)
print("Saved class names:", CLASSES_PATH)
