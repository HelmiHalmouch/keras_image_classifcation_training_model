# -*- coding: utf-8 -*-

'''
Application of fit, fit_generator, and train_on_batch methods
in Keras for training a model.

GHNAMI Helmi
27/12/2018
'''

# Import libraries and packages
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from minivggnet_model.minivggnet import MiniVGGNet


class ImageClassificationModel:
    def __init__(self, train_csv, test_csv, epochs=75, batch_size=32, img_size=(64, 64), learning_rate=1e-2):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.learning_rate = learning_rate

        self.model = None
        self.train_gen = None
        self.test_gen = None
        self.lb = None
        self.history = None

    def load_and_preprocess_data(self):
        # Load and preprocess the datasets
        labels = set()
        test_labels = []

        NUM_TRAIN_IMAGES = 0
        NUM_TEST_IMAGES = 0

        # Load the training CSV and extract labels
        with open(self.train_csv, 'r') as f:
            for line in f:
                label = line.strip().split(",")[0]
                labels.add(label)
                NUM_TRAIN_IMAGES += 1

        # Load the testing CSV and extract labels
        with open(self.test_csv, 'r') as f:
            for line in f:
                label = line.strip().split(",")[0]
                test_labels.append(label)
                NUM_TEST_IMAGES += 1

        # Initialize the label binarizer and one-hot encode the labels
        self.lb = LabelBinarizer()
        self.lb.fit(list(labels))
        test_labels = self.lb.transform(test_labels)

        # Initialize the data augmentation object
        aug = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")

        # Initialize the training and testing data generators
        self.train_gen = self.csv_image_generator(self.train_csv, self.batch_size, self.lb, mode="train", aug=aug)
        self.test_gen = self.csv_image_generator(self.test_csv, self.batch_size, self.lb, mode="train", aug=None)

        return NUM_TRAIN_IMAGES, NUM_TEST_IMAGES

    def csv_image_generator(self, input_path, batch_size, lb, mode="train", aug=None):
        '''
        Generates batches of images and labels from a CSV file.
        input_path: path to the CSV dataset file
        batch_size: batch size for training
        lb: label binarizer object containing class labels
        mode: "train" for training, "eval" for evaluation
        aug: data augmentation object (optional)
        '''
        with open(input_path, "r") as f:
            while True:
                images = []
                labels = []
                while len(images) < batch_size:
                    line = f.readline()
                    if line == "":
                        f.seek(0)
                        line = f.readline()
                        if mode == "eval":
                            break

                    line = line.strip().split(",")
                    label = line[0]
                    image = np.array([int(x) for x in line[1:]], dtype="uint8")
                    image = image.reshape((self.img_size[0], self.img_size[1], 3))

                    images.append(image)
                    labels.append(label)

                labels = lb.transform(np.array(labels))

                if aug is not None:
                    images, labels = next(aug.flow(np.array(images), labels, batch_size=batch_size))

                yield np.array(images), labels

    def build_model(self, num_classes):
        # Initialize and compile the model
        self.model = MiniVGGNet.build(self.img_size[0], self.img_size[1], 3, num_classes)
        self.model.summary()

        opt = SGD(lr=self.learning_rate, momentum=0.9, decay=self.learning_rate / self.epochs)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    def train_model(self, num_train_images, num_test_images):
        # Train the network using the training generator
        print("[INFO] training with generator...")
        self.history = self.model.fit_generator(
            self.train_gen,
            steps_per_epoch=num_train_images // self.batch_size,
            validation_data=self.test_gen,
            validation_steps=num_test_images // self.batch_size,
            epochs=self.epochs
        )

    def evaluate_model(self, num_test_images):
        # Evaluate the model on the test data
        print("[INFO] evaluating network...")
        test_gen_eval = self.csv_image_generator(self.test_csv, self.batch_size, self.lb, mode="eval", aug=None)

        pred_idxs = self.model.predict_generator(test_gen_eval, steps=(num_test_images // self.batch_size) + 1)
        pred_idxs = np.argmax(pred_idxs, axis=1)

        print(classification_report(self.lb.transform(test_labels).argmax(axis=1), pred_idxs, target_names=self.lb.classes_))

    def plot_results(self):
        # Plot the training loss and accuracy
        N = self.epochs
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), self.history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), self.history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), self.history.history["acc"], label="train_acc")
        plt.plot(np.arange(0, N), self.history.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("Training_result.png")
        print('Processing finished!!')

    def save_model(self):
        # Save the trained model
        self.model.save('the_trained_model.h5')


# Usage example
if __name__ == "__main__":
    # Paths to the dataset CSV files
    TRAIN_CSV = "flowers17_training.csv"
    TEST_CSV = "flowers17_testing.csv"

    # Initialize the model
    model = ImageClassificationModel(TRAIN_CSV, TEST_CSV)

    # Load and preprocess data
    num_train_images, num_test_images = model.load_and_preprocess_data()

    # Build the model
    model.build_model(num_classes=len(model.lb.classes_))

    # Train the model
    model.train_model(num_train_images, num_test_images)

    # Evaluate the model
    model.evaluate_model(num_test_images)

    # Plot results
    model.plot_results()

    # Save the model
    model.save_model()
