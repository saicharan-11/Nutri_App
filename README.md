Food Classification and Nutrition-Based Disease Suitability Prediction

This project focuses on classifying food images from the Food-101 dataset and analyzing their nutritional values to predict whether they are suitable for individuals with pre-existing conditions such as diabetes, hypertension, heart disease, and kidney disease. The experiments involved building and improving models for food classification and integrating nutritional data to assess disease suitability.

Dataset
Food-101 Dataset:

Contains 101 food classes with 1,000 images per class.
Images are diverse and include various cuisines and food types.
Used for training and testing food classification models.

Nutritional Data:

Each food class is mapped to its nutritional values (e.g., calories, protein, carbohydrates, fats, fiber, sugars, sodium).
Nutritional values are scaled based on user-defined serving sizes.

Objective

Food Classification:
Classify food images into one of the 101 classes using state-of-the-art deep learning models.
Disease Suitability Prediction:
Use the nutritional values of the predicted food class to determine its suitability for individuals with:
Diabetes
Hypertension
Heart Disease
Kidney Disease

Experiments
1. Initial Experiment: Simple Neural Network
Model: A basic neural network with 101 output classes.
Dataset: Sample size of 100 images per class.
Results: Low accuracy due to the complexity of the dataset and lack of spatial feature extraction.

3. Experiment with Convolutional Neural Networks (CNNs)
Model: A custom CNN architecture with convolutional and pooling layers.
Dataset: Sample size of 100 images per class.
Results: Low accuracy same as simple neural network, but still limited by the dataset size and model complexity.


5. Transfer Learning with ResNet and EfficientNet B0
Approach:
Leveraged pre-trained models (ResNet and EfficientNet B0) for transfer learning.
Fine-tuned the models on the Food-101 dataset.
Dataset:
Initial experiments used 100 images per class.
Gradually increased the sample size to 500 and 1,000 images per class for EfficientNet B0.
Results:
ResNet: Achieved moderate accuracy but required significant computational resources.
EfficientNet B0:
Achieved 76.39% accuracy on the test data with 1,000 images per class.
Demonstrated better performance and efficiency compared to ResNet.
Disease Suitability Prediction
Nutritional Analysis:

Each food class is associated with its nutritional values (e.g., calories, protein, carbohydrates, fats, fiber, sugars, sodium).
Nutritional values are scaled based on user-defined serving sizes.

Prediction Models:

A Random Forest model was trained to predict disease suitability based on nutritional values.
Labels for suitability:
"Suitable": The food meets the dietary requirements for the condition.
"Not Suitable": The food exceeds the dietary thresholds for the condition.
Conditions Evaluated:

Diabetes: Focused on sugar and carbohydrate content.
Hypertension: Focused on sodium levels.
Heart Disease: Focused on fats and cholesterol.
Kidney Disease: Focused on protein and sodium levels.

Key Results
Food Classification:

EfficientNet B0 achieved the best performance with 76.39% accuracy on the test data.


Disease Suitability:

The Random Forest model successfully predicted disease suitability based on nutritional values.
Predictions were displayed as "Suitable" or "Not Suitable" for each condition.

Conclusion:

This project demonstrates the potential of combining deep learning for food classification with nutritional analysis to provide actionable insights for individuals with pre-existing health conditions. The use of EfficientNet B0 for classification and a Random Forest model for disease suitability prediction highlights the effectiveness of integrating machine learning techniques for real-world applications.



App can be Acessed at https://nutriapp.streamlit.app/
