# Email-Spam-And-Analysis
This project is a web application that uses machine learning to classify SMS messages as spam or not spam. The application is built using Flask, a Python web framework, and utilizes various libraries such as NLTK, Pandas, Scikit-learn, and LIME for text preprocessing, model training, and interpretation.
Features

User-friendly web interface to input SMS text and get the prediction (spam or not spam)
Displays the length of the input text
Shows the confidence scores (probabilities) for spam and not spam predictions
Provides an explanation for the prediction by highlighting the most important words
Generates charts and visualizations, including:

Dataset pie chart (distribution of spam and not spam messages)
Word importance chart (bar chart showing the weights of important words)
Average length bar chart (average length of spam and not spam messages)



Installation
Clone the repository: git clone https://github.com/your-username/sms-spam-classifier.git
Navigate to the project directory: cd sms-spam-classifier
Install the required dependencies: pip install -r requirements.txt
Download the necessary data files (spam.csv, vectorizer.pkl, model.pkl) and place them in the project directory.

Usage
Run the Flask application:python app.py
Open your web browser and visit http://localhost:5000 to access the SMS Spam Classifier.
Enter the SMS text in the provided textarea and click the "Predict" button.
The prediction (spam or not spam), text length, and confidence scores will be displayed.
Use the buttons in the "Options" section to view different charts and visualizations, such as the dataset pie chart, word importance chart, average length bar chart, and explanation.

Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

Acknowledgements:
The spam dataset used in this project is from the UCI Machine Learning Repository.
The project utilizes the following libraries:

Flask for the web framework
NLTK for natural language processing
Pandas for data manipulation
Scikit-learn for machine learning
LIME for model interpretation



