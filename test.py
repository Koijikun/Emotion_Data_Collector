import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

img_size = (48, 48)
batch_size = 16
color_mode = 'grayscale'
num_classes = 6

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'model_data/test',
    target_size=img_size,
    batch_size=batch_size,
    color_mode=color_mode,
    class_mode='categorical',
    shuffle=False
)

model = load_model('saved_models/best_emotion_recognition_model.keras')

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy:.2f}')
print(f'Test loss: {test_loss:.2f}')

# Predicting on test data (optional, if you want to see the predictions)
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())


print("Predictions:")
for i in range(10):
    print(f"True label: {class_labels[true_classes[i]]}, Predicted: {class_labels[predicted_classes[i]]}")

# Generate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))