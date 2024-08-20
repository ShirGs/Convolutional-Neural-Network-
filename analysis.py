!pip install tensorflow
!pip install tensorflow tensorboard
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# Download, preprocess and subset the CIFAR10 data.

(x_data, y_data), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

x_train = []
y_train = []
for i in range(10):
    x_train_cls = x_data[y_data[:,0] == i]
    y_train_cls = y_data[y_data[:,0] == i]
    if i < 4:
        x_train_cls = x_train_cls[:500]
        y_train_cls = y_train_cls[:500]
    x_train.append(x_train_cls)
    y_train.append(y_train_cls)

x_train = np.vstack(x_train)
y_train = np.vstack(y_train)

index = np.arange(x_train.shape[0])
np.random.shuffle(index)
x_train = x_train[index]
y_train = y_train[index]

x_train_6 = x_train[y_train[:,0] > 3]
x_train_6 = (x_train_6 / 255.0).astype(np.float32)
y_train_6 = y_train[y_train[:,0] > 3] - 4
y_train_6_1_hot = tf.keras.utils.to_categorical(y_train_6)

x_test_6 = x_test[y_test[:,0] > 3]
x_test_6 = (x_test_6 / 255.0).astype(np.float32)
y_test_6 = y_test[y_test[:,0] > 3] - 4
y_test_6_1_hot = tf.keras.utils.to_categorical(y_test_6)

x_train_4 = x_train[y_train[:,0] < 4]
x_train_4 = (x_train_4 / 255.0).astype(np.float32)
y_train_4 = y_train[y_train[:,0] < 4]
y_train_4_1_hot = tf.keras.utils.to_categorical(y_train_4)

x_test_4 = x_test[y_test[:,0] < 4]
x_test_4 = (x_test_4 / 255.0).astype(np.float32)
y_test_4 = y_test[y_test[:,0] < 4]
y_test_4_1_hot = tf.keras.utils.to_categorical(y_test_4)

x_train_all = (x_train / 255.0).astype(np.float32)
y_train_all_1_hot = tf.keras.utils.to_categorical(y_train)
x_test_all = (x_test / 255.0).astype(np.float32)
y_test_all_1_hot = tf.keras.utils.to_categorical(y_test)



## Hyperparameter tuning, batch size = 32, lr = 0.001, epochs = 100

# Define model architecture
def create_model(num_classes, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameters
batch_size = 32
learning_rate = 0.001
epochs = 100
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# TensorBoard log directory
log_dir = f'logs/bs_{batch_size}_lr_{learning_rate}'
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Create and train the model
model_6 = create_model(6, learning_rate)
history_6 = model_6.fit(
    x_train_6, y_train_6_1_hot,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test_6, y_test_6_1_hot),
    callbacks=[tensorboard_callback, early_stopping],
    verbose=0
)

# Evaluate model
train_loss_6, train_acc_6 = model_6.evaluate(x_train_6, y_train_6_1_hot, verbose=0)
test_loss_6, test_acc_6 = model_6.evaluate(x_test_6, y_test_6_1_hot, verbose=0)
print(f"Training Accuracy: {train_acc_6*100:.2f}%")
print(f"Test Accuracy: {test_acc_6*100:.2f}%")


### Function for different kind of architechtures ###

# Baseline architecture
def create_baseline_model(num_classes, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Deeper architecture
def create_deeper_model(num_classes, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Wider architecture
def create_wider_model(num_classes, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model



# Filter datasets for 4-class and 6-class tasks
x_train_all = x_train
y_train_all = y_train
x_test_all = x_test
y_test_all = y_test

# Load the base model (VGG16 in this case) without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Function to add new top layers
def add_top_layers(base_model, num_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile and train model function
def compile_and_train(model, x_train, y_train, x_val, y_val, class_weights=None):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        x_train, y_train, epochs=50, batch_size=32,
        validation_data=(x_val, y_val),
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=0
    )
    return model, history

# Evaluate and collect per-class accuracy
def evaluate_per_class_accuracy(model, x_test, y_test, num_classes):
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=-1)
    y_true = np.argmax(y_test, axis=-1)
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    for i in range(len(y_test)):
        label = y_true[i]
        pred = y_pred[i]
        class_correct[int(label)] += (pred == int(label))
        class_total[int(label)] += 1
    class_accuracies = {i: (100 * class_correct[i] / class_total[i]) for i in range(num_classes)}
    return class_accuracies

# Plotting the results
def plot_class_accuracies(class_accuracies, dataset_name, labels):
    classes = list(class_accuracies.keys())
    class_acc_values = list(class_accuracies.values())

    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(classes))

    plt.bar(index, class_acc_values, bar_width, label='Test Accuracy')

    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Per-Class Accuracy for {dataset_name}')
    plt.xticks(index, labels, rotation=45)
    plt.legend()

    # Adding data labels
    for i, v in enumerate(class_acc_values):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Plotting training and validation accuracy/loss
def plot_training_history(history, dataset_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy for {dataset_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title(f'Training and Validation Loss for {dataset_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Train and evaluate for the 4-class dataset
model_4 = add_top_layers(base_model, 4)
model_4, history_4 = compile_and_train(model_4, x_train_4, y_train_4_1_hot, x_test_4, y_test_4_1_hot)
train_loss_4, train_acc_4 = model_4.evaluate(x_train_4, y_train_4_1_hot, verbose=0)
test_loss_4, test_acc_4 = model_4.evaluate(x_test_4, y_test_4_1_hot, verbose=0)
class_accuracies_4 = evaluate_per_class_accuracy(model_4, x_test_4, y_test_4_1_hot, 4)

# Calculate class weights for the full imbalanced dataset
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_all), y=y_train_all.flatten())
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Train and evaluate for the full imbalanced dataset
model_all = add_top_layers(base_model, 10)
model_all, history_all = compile_and_train(model_all, x_train_all, y_train_all_1_hot, x_test_all, y_test_all_1_hot, class_weights=class_weights_dict)
train_loss_all, train_acc_all = model_all.evaluate(x_train_all, y_train_all_1_hot, verbose=0)
test_loss_all, test_acc_all = model_all.evaluate(x_test_all, y_test_all_1_hot, verbose=0)
class_accuracies_all = evaluate_per_class_accuracy(model_all, x_test_all, y_test_all_1_hot, 10)

# Plotting for four-class dataset
labels_4_class = ['airplane', 'automobile', 'bird', 'cat']
plot_class_accuracies(class_accuracies_4, '4-Class Dataset', labels_4_class)
plot_training_history(history_4, '4-Class Dataset')

# Plotting for full imbalanced dataset
labels_full_imbalanced = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plot_class_accuracies(class_accuracies_all, 'Full Imbalanced Dataset', labels_full_imbalanced)
plot_training_history(history_all, 'Full Imbalanced Dataset')

# Print overall training and test accuracy
print(f"Training Accuracy for 4-Class Dataset: {train_acc_4*100:.2f}%")
print(f"Test Accuracy for 4-Class Dataset: {test_acc_4*100:.2f}%")
for i, label in enumerate(labels_4_class):
    print(f"Accuracy of {label}: {class_accuracies_4[i]:.2f}%")

print(f"\nTraining Accuracy for Full Imbalanced Dataset: {train_acc_all*100:.2f}%")
print(f"Test Accuracy for Full Imbalanced Dataset: {test_acc_all*100:.2f}%")
for i, label in enumerate(labels_full_imbalanced):
    print(f"Accuracy of {label}: {class_accuracies_all[i]:.2f}%")

