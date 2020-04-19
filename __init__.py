from flask import Flask
from flask import render_template, url_for, request
import pandas as pd 
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    text = { 'content': 'Spam Classifier !' } 
    return render_template("home.html",
        title = 'Home',
        text = text)

@app.route('/predict', methods=['POST'])
def predict():
	val = { 'content': 'Results Page !'}
	
	cv=pickle.load(open('transform.pkl', 'rb'))
	clf=pickle.load(open('model.pkl', 'rb'))
	if request.method == 'POST':
		comment = request.form['comment']
		data =  [comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template("result.html", 
		prediction = my_prediction,
		title = 'Result', 
		text = val)


app.debug = True
if __name__ == "__main__":
    app.run()
