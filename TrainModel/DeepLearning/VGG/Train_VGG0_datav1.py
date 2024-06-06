import time
import math
import json
import os
import sys
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import tensorflow as tf

DATA_DIR = 'D:/Projects/ThyroidCancer/Data/slice_datav1'
TARGET_SIZE = (224, 224)
BATCH_SIZE = 12

# Data generators
train_datagen = ImageDataGenerator(rescale=1/255.0)
valid_datagen = ImageDataGenerator(rescale=1/255.0)
test_datagen = ImageDataGenerator(rescale=1/255.0)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'valid'),
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Convert generators to tf.data.Dataset
def generator_to_tfdata(generator):
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(None, TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(generator.class_indices)), dtype=tf.float32),
        )
    )
    return dataset

train_dataset = generator_to_tfdata(train_generator).repeat()
valid_dataset = generator_to_tfdata(valid_generator).repeat()

# Display information
print("Class Indices: ", train_generator.class_indices)
print(f"Number of training samples: {train_generator.samples}")
print(f"Number of validation samples: {valid_generator.samples}")
print(f"Number of test samples: {test_generator.samples}")

print("Training set:")
for class_name, idx in train_generator.class_indices.items():
    num_files = len(os.listdir(os.path.join(DATA_DIR, 'train', class_name)))
    print(f"{class_name} ({idx}): {num_files} files")

print("Validation set:")
for class_name, idx in valid_generator.class_indices.items():
    num_files = len(os.listdir(os.path.join(DATA_DIR, 'valid', class_name)))
    print(f"{class_name} ({idx}): {num_files} files")

print("Test set:")
for class_name, idx in test_generator.class_indices.items():
    num_files = len(os.listdir(os.path.join(DATA_DIR, 'test', class_name)))
    print(f"{class_name} ({idx}): {num_files} files")

def save_history(history):
    acc = pd.Series(history.history["accuracy"], name="accuracy")
    loss = pd.Series(history.history["loss"], name="loss")
    val_acc = pd.Series(history.history["val_accuracy"], name="val_accuracy")
    val_loss = pd.Series(history.history["val_loss"], name="val_loss")
    com = pd.concat([acc, loss, val_acc, val_loss], axis=1)
    com.to_csv("slice_datav1_vgg2_history.csv", index=False)

def plot_history(history):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Accuracy and Loss")
    plt.ylabel("Accuracy/Loss")
    plt.xlabel("Epoch")
    plt.legend(["accuracy", "val_accuracy", "loss", "val_loss"], loc="upper right")
    plt.savefig("slice_datav1_vgg2_model_accuracy_loss.png")
    # plt.show()

start = time.process_time()
# Tính số bước cho mỗi epoch
num_train_steps = math.ceil(train_generator.samples / BATCH_SIZE)
num_valid_steps = math.ceil(valid_generator.samples / BATCH_SIZE)
classes = list(iter(train_generator.class_indices))
print(f'Num train step: {num_train_steps}\nNum valid step: {num_valid_steps}\nClasses: {classes}')

Inp = Input((224, 224, 3))
base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model(Inp)
x = GlobalAveragePooling2D()(x) # số lượng đặc trưng được duỗi ra là 512
x = Dense(1024, activation='relu')(x)   # như vậy 1024 neutral ở tầng dense là đủ để học đặc trưng
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)   # như vậy 1024 neutral ở tầng dense là đủ để học đặc trưng
x = Dropout(0.5)(x)
predictions = Dense(len(classes), activation="softmax")(x)
finetuned_model = Model(inputs=Inp, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = True

freeze_layers = 2

# Đóng băng các tầng cuối cùng
for layer in base_model.layers[-freeze_layers:]:
    layer.trainable = False

finetuned_model.load_weights('D:/Projects/ThyroidCancer/TrainModel/DeepLearning/VGG/vgg1_best_slice_datav1.keras')

print('Model architecture:')
finetuned_model.summary()

finetuned_model.compile(
    optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"]
)

for c in train_generator.class_indices:
    classes[train_generator.class_indices[c]] = c
finetuned_model.classes = classes
early_stopping = EarlyStopping(patience=15)
checkpointer = ModelCheckpoint(
    "vgg2_best_slice_datav1.keras",
    verbose=1,
    save_best_only=True,
)
History = finetuned_model.fit(
    train_dataset,
    steps_per_epoch=num_train_steps,
    epochs=100,
    callbacks=[early_stopping, checkpointer],
    validation_data=valid_dataset,
    validation_steps=num_valid_steps,
)
save_history(History)
plot_history(History)
end2 = time.process_time()
print("final is in ", end2 - start)
# start 02h55 23/05/2024 - 