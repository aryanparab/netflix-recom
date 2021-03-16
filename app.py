from flask import Flask, render_template, url_for, request, redirect
import numpy as np
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)

df = pd.read_csv('dataframe.csv')
vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
indices = pd.Series(df['title'].index, index = df['title']).drop_duplicates()

corpus = []
for i in df['description_cleaned']:
    corpus.append(i)

ncorpus = vectorizer.transform(corpus)
similarity = cosine_similarity(ncorpus)

def rec_cosine(title,sig=similarity):
    idx = indices[title]
    sig_score = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_score, key = lambda x : x[1],reverse = True)
    sig_scores = sig_scores[1:11]
    movies_indices = [i[0] for i in sig_scores]
    return df.iloc[movies_indices]


@app.route('/',methods=['POST','GET'])
def index():
	val=[]
	if request.method == "POST":

		movie = request.form['C']
		if movie != '':
			df = rec_cosine(movie)
			val = []

			for i in range(df.shape[0]):
			    val.append(df.iloc[i].values)

			return render_template('home.html',context=val)

	else:
		return render_template('home.html')

if __name__ == '__main__':
	app.run(debug=False)