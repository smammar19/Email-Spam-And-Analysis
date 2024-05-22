# Email-Spam-And-Analysis
This project is a web application that uses machine learning to classify SMS messages as spam or not spam. The application is built using Flask, a Python web framework, and utilizes various libraries such as NLTK, Pandas, Scikit-learn, and LIME for text preprocessing, model training, and interpretation.

## Features
1. User-friendly web interface to input SMS text and get the prediction (spam or not spam)
2. Displays the length of the input text
3. Shows the confidence scores (probabilities) for spam and not spam predictions
4. Provides an explanation for the prediction by highlighting the most important words
5. Generates charts and visualizations, including:
    - Dataset pie chart (distribution of spam and not spam messages)
    - Word importance chart (bar chart showing the weights of important words)
    - Average length bar chart (average length of spam and not spam messages)


## Installation
1. Clone the repository: git clone https://github.com/your-username/sms-spam-classifier.git
2. Navigate to the project directory: cd sms-spam-classifier
3. Install the required dependencies: pip install -r requirements.txt
4. Download the necessary data files (spam.csv, vectorizer.pkl, model.pkl) and place them in the project directory.

## Usage
Run the Flask application: python app.py
Open your web browser and visit http://localhost:5000 to access the SMS Spam Classifier.
Enter the SMS text in the provided text area and click the "Predict" button.
The prediction (spam or not spam), text length, and confidence scores will be displayed.
Use the buttons in the "Options" section to view different charts and visualizations, such as the dataset pie chart, word importance chart, average length bar chart, and explanation.

## Screenshots
![01](https://github.com/smammar19/Email-Spam-And-Analysis/assets/135743822/6f81b761-51df-4cbe-8006-0c83a0be1b5c)
![2](https://github.com/smammar19/Email-Spam-And-Analysis/assets/135743822/8f4b6260-5b9c-41cb-86b8-241d7856852c)
![6](https://github.com/smammar19/Email-Spam-And-Analysis/assets/135743822/3ea97981-6306-46b3-8198-6870379d1559)
![7](https://github.com/smammar19/Email-Spam-And-Analysis/assets/135743822/b90e91c7-d04d-40a3-be09-cf32741f190f)
![3](https://github.com/smammar19/Email-Spam-And-Analysis/assets/135743822/aafa5cbb-40fe-4f4e-adff-1d2a8b3b4387)
![4](https://github.com/smammar19/Email-Spam-And-Analysis/assets/135743822/4bc310f4-3e99-4ec4-a4df-369e227a9ea2)
![5](https://github.com/smammar19/Email-Spam-And-Analysis/assets/135743822/62da7d4b-6b35-4c60-afa2-d684a4a8d0eb)


## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgements:
The spam dataset used in this project is from the UCI Machine Learning Repository.

The project utilizes the following libraries:
Flask for the web framework
NLTK for natural language processing
Pandas for data manipulation
Scikit-learn for machine learning
LIME for model interpretation



