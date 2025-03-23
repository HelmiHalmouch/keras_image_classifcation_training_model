# Keras Deep Learning Model Training Methods for Custom Image Classification

## 1. Main Objective

This project demonstrates the application of several Keras methods for training a model:

- **a.** `fit`  
- **b.** `fit_generator`  
- **c.** `train_on_batch`  

These methods are commonly used for training deep learning models on different datasets and for fine-tuning the training process.

---

## 2. General Keras Training Functions

Here are the three primary Keras training functions used:

2.1. **`fit`**  
   This is the most commonly used method to train a model on a dataset.  
   Example:
   ```python
   model.fit(trainX, trainY, batch_size=32, epochs=50)
   ```
2.2. **`fit_generator`**
```
model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                    validation_data=(testX, testY),
                    steps_per_epoch=len(trainX) // BS,
                    epochs=EPOCHS)
```
2.3. **`train_on_batch`**
```
   model.train_on_batch(batchX, batchY)
```
### 3. How to Choose the Appropriate Keras Training Function?
- fit: Use this when you have a small dataset that can fit entirely into memory. It's the most straightforward and commonly used method.

- fit_generator: Use this when you need to perform data augmentation or when your dataset is too large to fit into memory. It allows you to generate data on the fly during training.

- train_on_batch: Use this when you require the finest-grained control over the training process. This method is beneficial for advanced customizations like implementing custom training loops or experimenting with different batch sizes.

### 4. Requirements
Before running the code, ensure that the following dependencies are installed:
```
Python 3.x (>=3.4)
TensorFlow + Keras
Scikit-learn
NumPy
Matplotlib
```
### 5. Dataset Information
The datasets used in this project can be downloaded from the following link:
Oxford Flowers 17 Dataset
The dataset files used in this project are:
```
flowers17_testing.csv
flowers17_training.csv
```
### 6. How to Run the Code
To run the code, execute the following Python script:
```
python keras_fit_fitgenerator.py
```
### 7. Model Architecture Used (MiniVGGNet)
The model used for classification in this project is a simplified version of VGGNet, known as MiniVGGNet. Below is the architecture visualization:
![minivggnet](https://user-images.githubusercontent.com/40611217/50492086-e2f7ea80-0a15-11e9-9f7f-2a09f57bbc23.JPG)

### 8. TensorBoard Logging (Optional)
To view training progress in TensorBoard, run the following command in your terminal:
```
tensorboard --logdir=runs
```
Then, open your browser and go to http://localhost:6006/ to visualize the training and validation metrics.


