import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# Enable mixed precision (if GPU supports) for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Paths
split_base = "/kaggle/working/BreaKHis_split"
train_dir = os.path.join(split_base, 'train')
val_dir = os.path.join(split_base, 'val')
test_dir = os.path.join(split_base, 'test')

# Parameters
IMG_SIZE = (300, 300)  # B3 default size
BATCH_SIZE = 32
EPOCHS_TOP = 50
EPOCHS_FINE = 50

# Data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Data generators
train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# Class weights
y_train = train_gen.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

# Build model
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
output = Dense(train_gen.num_classes, activation='softmax', dtype='float32')(x)  # ensure output in float32
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model first
for layer in base_model.layers:
    layer.trainable = False

# Compile top layers
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint("best_breakhis_model.h5", monitor='val_accuracy',
                             save_best_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=8,
                           restore_best_weights=True, verbose=1)

# Train top layers
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_TOP,
    class_weight=class_weights_dict,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# Fine-tune: unfreeze last 100 layers
for layer in base_model.layers[-100:]:
    layer.trainable = True

# Compile for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tune training
history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    class_weight=class_weights_dict,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# Load best model
model.load_weights("best_breakhis_model.h5")

# Evaluate on test set
loss, acc = model.evaluate(test_gen)
print(f"\n✅ Final Test Accuracy: {acc*100:.2f}%")

# Save final model
model.save("/kaggle/working/breakhis_classifier_final.h5")
print("💾 Model saved as breakhis_classifier_final.h5")