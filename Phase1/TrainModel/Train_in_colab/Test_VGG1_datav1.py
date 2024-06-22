import time
import math
import json
import os
import sys
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

DATA_DIR = 'D:/Projects/ThyroidCancer/Data/slice_datav1'
TARGET_SIZE = (224, 224)
BATCH_SIZE = 12 * 10

# Data generators
test_datagen = ImageDataGenerator(rescale=1/255.0)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Model
classes = list(iter(test_generator.class_indices))
Inp = Input((224, 224, 3))
base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model(Inp)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)   # như vậy 1024 neutral ở tầng dense là đủ để học đặc trưng
x = Dropout(0.5)(x)
predictions = Dense(len(classes), activation="softmax")(x)
finetuned_model = Model(inputs=Inp, outputs=predictions)

finetuned_model.load_weights('D:/Projects/ThyroidCancer/TrainModel/DeepLearning/VGG/vgg1_best_slice_datav1.keras')

finetuned_model.summary()

# Predict on test data
test_steps = math.ceil(test_generator.samples / BATCH_SIZE)
test_generator.reset()
predictions = finetuned_model.predict(test_generator, steps=test_steps, verbose=1)

# Convert predictions to class labels
predicted_classes = predictions.argmax(axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Calculate metrics
conf_matrix = confusion_matrix(true_classes, predicted_classes)
report = classification_report(true_classes, predicted_classes, output_dict=True)

# Print metrics
print("Confusion Matrix")
print(conf_matrix)
print("Classification Report")
print(report)

# Tạo nhãn lớp (class labels)
class_labels = ["B2", "B5", "B6"]  # Thay thế bằng nhãn lớp thực tế của bạn

# Chuyển đổi báo cáo phân loại thành DataFrame
report_df = pd.DataFrame(report).transpose()

# Lưu confusion matrix và classification report vào tệp txt
with open('Test_VGG1_datav1.txt', 'w') as f:
    # Lưu confusion matrix
    f.write("Confusion Matrix\n")
    f.write(np.array2string(conf_matrix, separator=', ') + "\n\n")
    
    # Lưu classification report
    f.write("Classification Report\n")
    f.write(report_df.to_string() + "\n")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Lưu figure thành file ảnh
plt.savefig('Test_VGG1_datav1_result.png')

# Hiển thị ảnh
plt.show()
