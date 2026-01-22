import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
import cv2
import random

class HandwrittenAlphabetClassifier:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def load_and_prepare_data(self, csv_path):
        # Load data 
        df = pd.read_csv(csv_path, header=None)  # header=None means there are no column names

        # Separate labels and features
        self.y = df.iloc[:, 0].values  # First column as labels
        self.X = df.iloc[:, 1:].values  # Remaining columns as features
        self.unique_classes = np.unique(self.y)
        self.class_distribution = np.bincount(self.y)
        
        # Normalize data
        self.X_normalized = self.scaler.fit_transform(self.X)

        print("Reshaping features to 28x28 images for visualization purposes..")
        self.images = self.X_normalized.reshape(-1, 28, 28)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_normalized, self.y, test_size=0.2, random_state=42
        )

        # Splitting the train data further into training and validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )
    
    def analyze_data(self):
        # Plot class distribution
        sns.barplot(x=np.arange(len(self.class_distribution)), y=self.class_distribution, color='blue')
        plt.title('Distribution of Classes')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.savefig('class_distribution.png')
        print("Class distribution saved as 'class_distribution.png'.")
        
        # Display sample images
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(self.images[i], cmap='gray')
            plt.title(f"Label: {self.y[i]}")
            plt.axis('off')
        plt.savefig('sample_images.png')
        print("Sample images saved as 'sample_images.png'.")
    
    def train_svm_models(self):
        self.svm_linear = SVC(kernel='linear', random_state=42, 
                            cache_size=1000,  # Increase cache size to speed up training
                            max_iter=1000)    # Limit iterations for the SVM to not take a long time training
        
        self.svm_rbf = SVC(kernel='rbf', random_state=42,
                          cache_size=1000,
                          max_iter=1000)
        
        print("Training Linear SVM...")
        self.svm_linear.fit(self.X_train, self.y_train)
        
        print("Training RBF SVM...")
        self.svm_rbf.fit(self.X_train, self.y_train)
        
        # Evaluate models
        linear_pred = self.svm_linear.predict(self.X_test)
        rbf_pred = self.svm_rbf.predict(self.X_test)
        
        # Plot confusion matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
        sns.heatmap(confusion_matrix(self.y_test, linear_pred), ax=ax1)
        ax1.set_title('Linear SVM Confusion Matrix')
        
        sns.heatmap(confusion_matrix(self.y_test, rbf_pred), ax=ax2)
        ax2.set_title('RBF SVM Confusion Matrix')
        
        plt.savefig('svm_confusion_matrices.png')
        plt.close()
        
        # Calculate F1 scores
        with open('svm_results.txt', 'w') as f:
            f.write(f"Linear SVM F1 Score: {f1_score(self.y_test, linear_pred, average='macro')}\n")
            f.write(f"RBF SVM F1 Score: {f1_score(self.y_test, rbf_pred, average='macro')}\n")
    
    def logistic_regression_from_scratch(self):
        class OneVsAllLogisticRegression:
            def __init__(self, num_classes, num_features, learning_rate=0.01, regularization=0.01):
                self.num_classes = num_classes
                self.weights = np.zeros((num_classes, num_features))
                self.bias = np.zeros(num_classes)
                self.initial_lr = learning_rate
                self.lr = learning_rate
                self.regularization = regularization
                
            def sigmoid(self, z):
                return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Clip to prevent overflow
                
            def lr_schedule(self, epoch, initial_lr, decay_rate=0.1):
                return initial_lr / (1 + decay_rate * epoch)
            
            def compute_metrics(self, X, y):
                y_pred_prob = self.sigmoid(np.dot(X, self.weights.T) + self.bias)
                y_pred = np.argmax(y_pred_prob, axis=1)
                y_one_hot = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
                loss = self._compute_loss(y_one_hot, y_pred_prob)
                accuracy = self._compute_accuracy(y, y_pred)
                return loss, accuracy
                
            def fit(self, X_train, y_train, X_val, y_val, epochs=100, patience=5, batch_size=32):
                self.train_loss = []
                self.train_acc = []
                self.val_loss = []
                self.val_acc = []
                
                n_samples = X_train.shape[0]
                best_val_loss = float('inf')
                patience_counter = 0
                best_weights = None
                best_bias = None
                
                for epoch in range(epochs):
                    # Update learning rate
                    self.lr = self.lr_schedule(epoch, self.initial_lr)
                    
                    # Mini-batch training
                    indices = np.random.permutation(n_samples)
                    for i in range(0, n_samples, batch_size):
                        batch_indices = indices[i:min(i + batch_size, n_samples)]
                        X_batch = X_train[batch_indices]
                        y_batch = y_train[batch_indices]
                        
                        # Forward pass
                        z = np.dot(X_batch, self.weights.T) + self.bias
                        y_pred = self.sigmoid(z)
                        
                        # Convert labels to one-hot encoding
                        y_one_hot = tf.keras.utils.to_categorical(y_batch, num_classes=self.num_classes)
                        
                        # Compute gradients with L2 regularization
                        error = y_pred - y_one_hot
                        dw = (np.dot(error.T, X_batch) + self.regularization * self.weights) / len(X_batch)
                        db = np.sum(error, axis=0) / len(X_batch)
                        
                        # Update weights and bias
                        self.weights -= self.lr * dw
                        self.bias -= self.lr * db
                    
                    # Compute metrics for full training and validation sets
                    train_loss, train_acc = self.compute_metrics(X_train, y_train)
                    val_loss, val_acc = self.compute_metrics(X_val, y_val)
                    
                    self.train_loss.append(train_loss)
                    self.train_acc.append(train_acc)
                    self.val_loss.append(val_loss)
                    self.val_acc.append(val_acc)
                    
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_weights = self.weights.copy()
                        best_bias = self.bias.copy()
                    else:
                        patience_counter += 1
                    
                    # Print progress
                    if (epoch + 1) % 10 == 0:
                        print(f'Epoch {epoch + 1}/{epochs}:')
                        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                    
                    if patience_counter >= patience:
                        print(f'Early stopping triggered at epoch {epoch + 1}')
                        break
                
                # Restore best weights
                if best_weights is not None:
                    self.weights = best_weights
                    self.bias = best_bias
            
            def predict(self, X):
                z = np.dot(X, self.weights.T) + self.bias
                y_pred = self.sigmoid(z)
                return np.argmax(y_pred, axis=1)
            
            def _compute_loss(self, y_true, y_pred):
                epsilon = 1e-15
                loss = -np.mean(y_true * np.log(y_pred + epsilon) + 
                            (1 - y_true) * np.log(1 - y_pred + epsilon))
                l2_loss = 0.5 * self.regularization * np.sum(self.weights ** 2)
                return loss + l2_loss
            
            def _compute_accuracy(self, y_true, y_pred):
                return np.mean(y_true == y_pred)
        
        # Train logistic regression
        num_classes = len(np.unique(self.y_train))
        self.log_reg = OneVsAllLogisticRegression(num_classes, self.X_train.shape[1])
        
        # Train with validation data
        self.log_reg.fit(self.X_train, self.y_train, self.X_val, self.y_val, epochs=100)
        
        # Plot learning curves
        plt.figure(figsize=(15, 5))
        
        # Plot loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.log_reg.train_loss, label='Training Loss')
        plt.plot(self.log_reg.val_loss, label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(self.log_reg.train_acc, label='Training Accuracy')
        plt.plot(self.log_reg.val_acc, label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('logistic_regression_curves.png')
        plt.close()
        
        # Evaluate model on test set
        y_pred = self.log_reg.predict(self.X_test)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Logistic Regression Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('logistic_regression_confusion.png')
        plt.close()
        
        # Calculate and save F1 scores
        f1_macro = f1_score(self.y_test, y_pred, average='macro')
        f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
        
        with open('logistic_regression_results.txt', 'w') as f:
            f.write(f"Logistic Regression Macro F1 Score: {f1_macro:.4f}\n")
            f.write(f"Logistic Regression Weighted F1 Score: {f1_weighted:.4f}\n")
    
    def create_neural_networks(self):
        # First NN with Simple architecture
        self.nn1 = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.X.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'), 
            tf.keras.layers.Dense(len(self.unique_classes), activation='softmax')
        ])
        
        # Second NN with More complex architecture
        self.nn2 = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(self.X.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.unique_classes), activation='softmax')
        ])
        
        # Compile models
        self.nn1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.nn2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train models
        history1 = self.nn1.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_train, self.y_train),
            epochs=10,
            validation_split=0.2, 
            verbose=1
        )
        
        history2 = self.nn2.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_train, self.y_train),
            epochs=10,
            validation_split=0.2, 
            verbose=1
        )
        
        # Plot learning curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Model 1
        axes[0, 0].plot(history1.history['loss'], label='train')
        axes[0, 0].plot(history1.history['val_loss'], label='val')
        axes[0, 0].set_title('Model 1 Loss')
        axes[0, 0].legend()
        
        axes[0, 1].plot(history1.history['accuracy'], label='train')
        axes[0, 1].plot(history1.history['val_accuracy'], label='val')
        axes[0, 1].set_title('Model 1 Accuracy')
        axes[0, 1].legend()
        
        # Model 2
        axes[1, 0].plot(history2.history['loss'], label='train')
        axes[1, 0].plot(history2.history['val_loss'], label='val')
        axes[1, 0].set_title('Model 2 Loss')
        axes[1, 0].legend()
        
        axes[1, 1].plot(history2.history['accuracy'], label='train')
        axes[1, 1].plot(history2.history['val_accuracy'], label='val')
        axes[1, 1].set_title('Model 2 Accuracy')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('neural_network_curves.png')
        plt.close()
        
        # Evaluate models
        y_pred1 = np.argmax(self.nn1.predict(self.X_test), axis=1)
        y_pred2 = np.argmax(self.nn2.predict(self.X_test), axis=1)

        f1_nn1 = f1_score(self.y_test, y_pred1, average='macro')
        f1_nn2 = f1_score(self.y_test, y_pred2, average='macro')
        
        # Save best model
        if f1_nn1 > f1_nn2:
            self.nn1.save('best_model.h5')
            best_pred = y_pred1
            best_model_str = "NN1"
        else:
            self.nn2.save('best_model.h5')
            best_pred = y_pred2
            best_model_str = "NN2"
        
        # Plot confusion matrix for best model
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(self.y_test, best_pred))
        plt.title('Best Neural Network Confusion Matrix')
        plt.savefig('best_nn_confusion.png')
        plt.close()


        with open('nn_results.txt', 'w') as f:
            f.write(f"NN1 F1 Score: {f1_nn1}\n")
            f.write(f"NN2 F1 Score: {f1_nn2}\n")
            f.write(f"Best Model: {best_model_str}\n")

# Finction that generates a 28x28 pixels letters to test them latter against the model
def generate_letter(letter_type, mode="synthetic"):
    """
    Generate letters on a 28x28 canvas.
    mode: "synthetic" for basic synthetic letters, "handwritten" for handwritten-style letters.
    """
    canvas = np.zeros((28, 28))

    def jitter(value, amount=2):
        """Add random jitter to a coordinate."""
        return value + random.randint(-amount, amount)

    if mode == "synthetic":
        if letter_type == 'A':
            cv2.line(canvas, (6, 24), (14, 4), 1, 2)  # Left diagonal
            cv2.line(canvas, (14, 4), (22, 24), 1, 2)  # Right diagonal
            cv2.line(canvas, (10, 15), (18, 15), 1, 2)  # Horizontal bar

        elif letter_type == 'H':
            cv2.line(canvas, (8, 4), (8, 24), 1, 2)    # Left vertical
            cv2.line(canvas, (20, 4), (20, 24), 1, 2)  # Right vertical
            cv2.line(canvas, (8, 14), (20, 14), 1, 2)  # Horizontal bar

        elif letter_type == 'M':
            cv2.line(canvas, (6, 24), (6, 4), 1, 2)    # Left vertical
            cv2.line(canvas, (22, 24), (22, 4), 1, 2)  # Right vertical
            cv2.line(canvas, (6, 4), (14, 14), 1, 2)   # Left diagonal
            cv2.line(canvas, (22, 4), (14, 14), 1, 2)  # Right diagonal

        elif letter_type == 'E':
            cv2.line(canvas, (8, 4), (8, 24), 1, 2)    # Vertical line
            cv2.line(canvas, (8, 4), (20, 4), 1, 2)    # Top horizontal
            cv2.line(canvas, (8, 14), (18, 14), 1, 2)  # Middle horizontal
            cv2.line(canvas, (8, 24), (20, 24), 1, 2)  # Bottom horizontal

        elif letter_type == 'D':
            cv2.line(canvas, (8, 4), (8, 24), 1, 2)    # Vertical line
            cv2.ellipse(canvas, (8, 14), (10, 12), 0, -90, 90, 1, 2)  # Curved part

        elif letter_type == 'O':
            cv2.ellipse(canvas, (14, 14), (8, 10), 0, 0, 360, 1, 2)

        elif letter_type == 'R':
            cv2.line(canvas, (8, 4), (8, 24), 1, 2)    # Vertical line
            cv2.ellipse(canvas, (8, 9), (10, 5), 0, -90, 90, 1, 2)  # Top curved part
            cv2.line(canvas, (8, 14), (20, 24), 1, 2)  # Diagonal line

        elif letter_type == 'I':
            cv2.line(canvas, (14, 4), (14, 24), 1, 2)  # Vertical line
            cv2.line(canvas, (8, 4), (20, 4), 1, 2)    # Top horizontal
            cv2.line(canvas, (8, 24), (20, 24), 1, 2)  # Bottom horizontal

        elif letter_type == 'Z':
            cv2.line(canvas, (8, 4), (20, 4), 1, 2)    # Top horizontal
            cv2.line(canvas, (20, 4), (8, 24), 1, 2)   # Diagonal
            cv2.line(canvas, (8, 24), (20, 24), 1, 2)  # Bottom horizontal

        elif letter_type == 'L':
            cv2.line(canvas, (8, 4), (8, 24), 1, 2)    # Vertical line
            cv2.line(canvas, (8, 24), (20, 24), 1, 2)  # Bottom horizontal

        elif letter_type == 'N':
            cv2.line(canvas, (8, 24), (8, 4), 1, 2)    # Left vertical
            cv2.line(canvas, (8, 4), (20, 24), 1, 2)   # Diagonal
            cv2.line(canvas, (20, 24), (20, 4), 1, 2)  # Right vertical

        else:
            raise ValueError(f"Unsupported letter type: {letter_type}")

    elif mode == "handwritten":
        if letter_type == 'A':
            cv2.line(canvas, (jitter(6), jitter(24)), (jitter(14), jitter(4)), 1, 2)  # Left diagonal
            cv2.line(canvas, (jitter(14), jitter(4)), (jitter(22), jitter(24)), 1, 2)  # Right diagonal
            cv2.line(canvas, (jitter(10), jitter(15)), (jitter(18), jitter(15)), 1, 2)  # Horizontal bar
        
        elif letter_type == 'H':
            cv2.line(canvas, (jitter(8), jitter(4)), (jitter(8), jitter(24)), 1, 2)    # Left vertical
            cv2.line(canvas, (jitter(20), jitter(4)), (jitter(20), jitter(24)), 1, 2)  # Right vertical
            cv2.line(canvas, (jitter(8), jitter(14)), (jitter(20), jitter(14)), 1, 2)  # Horizontal bar
            
        elif letter_type == 'M':
            cv2.line(canvas, (jitter(6), jitter(24)), (jitter(6), jitter(4)), 1, 2)    # Left vertical
            cv2.line(canvas, (jitter(22), jitter(24)), (jitter(22), jitter(4)), 1, 2)  # Right vertical
            cv2.line(canvas, (jitter(6), jitter(4)), (jitter(14), jitter(14)), 1, 2)   # Left diagonal
            cv2.line(canvas, (jitter(22), jitter(4)), (jitter(14), jitter(14)), 1, 2)  # Right diagonal
            
        elif letter_type == 'E':
            cv2.line(canvas, (jitter(8), jitter(4)), (jitter(8), jitter(24)), 1, 2)    # Vertical line
            cv2.line(canvas, (jitter(8), jitter(4)), (jitter(20), jitter(4)), 1, 2)    # Top horizontal
            cv2.line(canvas, (jitter(8), jitter(14)), (jitter(18), jitter(14)), 1, 2)  # Middle horizontal
            cv2.line(canvas, (jitter(8), jitter(24)), (jitter(20), jitter(24)), 1, 2)  # Bottom horizontal
            
        elif letter_type == 'D':
            cv2.line(canvas, (jitter(8), jitter(4)), (jitter(8), jitter(24)), 1, 2)    # Vertical line
            cv2.ellipse(canvas, (jitter(8), jitter(14)), (10, 12), 0, -90, 90, 1, 2)   # Curved part
            
        elif letter_type == 'O':
            cv2.ellipse(canvas, (jitter(14), jitter(14)), (jitter(8), jitter(10)), 0, 0, 360, 1, 2)
            
        elif letter_type == 'R':
            cv2.line(canvas, (jitter(8), jitter(4)), (jitter(8), jitter(24)), 1, 2)    # Vertical line
            cv2.ellipse(canvas, (jitter(8), jitter(9)), (10, 5), 0, -90, 90, 1, 2)     # Top curved part
            cv2.line(canvas, (jitter(8), jitter(14)), (jitter(20), jitter(24)), 1, 2)  # Diagonal line
            
        elif letter_type == 'I':
            cv2.line(canvas, (jitter(14), jitter(4)), (jitter(14), jitter(24)), 1, 2)  # Vertical line
            cv2.line(canvas, (jitter(8), jitter(4)), (jitter(20), jitter(4)), 1, 2)    # Top horizontal
            cv2.line(canvas, (jitter(8), jitter(24)), (jitter(20), jitter(24)), 1, 2)  # Bottom horizontal
            
        elif letter_type == 'Z':
            cv2.line(canvas, (jitter(8), jitter(4)), (jitter(20), jitter(4)), 1, 2)    # Top horizontal
            cv2.line(canvas, (jitter(20), jitter(4)), (jitter(8), jitter(24)), 1, 2)   # Diagonal
            cv2.line(canvas, (jitter(8), jitter(24)), (jitter(20), jitter(24)), 1, 2)  # Bottom horizontal
            
        elif letter_type == 'L':
            cv2.line(canvas, (jitter(8), jitter(4)), (jitter(8), jitter(24)), 1, 2)    # Vertical line
            cv2.line(canvas, (jitter(8), jitter(24)), (jitter(20), jitter(24)), 1, 2)  # Bottom horizontal
            
        elif letter_type == 'N':
            cv2.line(canvas, (jitter(8), jitter(24)), (jitter(8), jitter(4)), 1, 2)    # Left vertical
            cv2.line(canvas, (jitter(8), jitter(4)), (jitter(20), jitter(24)), 1, 2)   # Diagonal
            cv2.line(canvas, (jitter(20), jitter(24)), (jitter(20), jitter(4)), 1, 2)  # Right vertical
            
        else:
            raise ValueError(f"Unsupported letter type: {letter_type}")
        
    else:
        raise ValueError(f"Unsupported letter type: {letter_type}")
    
    return canvas.flatten()

def test_letters_with_saved_model(model_path, test_letters):
    """
    Test specific letters by loading the best NN model generated (chosen) previously
    """
    # Load the saved model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate test cases for each letter
    test_cases = {letter: generate_letter(letter) for letter in test_letters}
    
    # Test the model
    print("\nTesting letters:")
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    results = []
    
    for letter, array in test_cases.items():
        # Reshape the array to match the model's input shape
        input_data = array[np.newaxis, :]
        
        # Model prediction
        prediction_probabilities = model.predict(input_data, verbose=0)
        prediction = np.argmax(prediction_probabilities, axis=1)
        predicted_label = chr(prediction[0] + ord('A'))
        
        # Calculate confidence score
        confidence = float(prediction_probabilities[0][prediction[0]] * 100)
        
        # Check if prediction is correct
        is_correct = letter == predicted_label
        status = "✓" if is_correct else "✗"
        
        # Store result
        results.append({
            'actual': letter,
            'predicted': predicted_label,
            'confidence': confidence,
            'correct': is_correct
        })

        print(f"Actual: {letter}, Predicted: {predicted_label} (Confidence: {confidence:.2f}%) {status}")
        
        if is_correct:
            correct_predictions += 1
    
    # Calculate and print overall accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\nOverall Accuracy on required letters: {accuracy:.2f}%")


def main():

    classifier = HandwrittenAlphabetClassifier()    

    # Loading and preforming operations on data such as splitting labels and features, splitting train and test data, etc.
    classifier.load_and_prepare_data('A_Z_Handwritten_Data.csv')
    
    # Analyze Data
    classifier.analyze_data()
    
    # Train and evaluate SVM models
    classifier.train_svm_models()
    
    # Train and evaluate logistic regression
    classifier.logistic_regression_from_scratch()
    
    # Train and evaluate neural networks
    classifier.create_neural_networks()
    
    # Generate all the letters that construct the names of the group students and test them.
    test_letters = ['A', 'H', 'M', 'E', 'D', 'O', 'R', 'I', 'Z', 'L', 'N']
    model_path = 'best_model.h5'
    test_letters_with_saved_model(model_path, test_letters)


if __name__ == "__main__":
    main()
