from flask import Flask, render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle

app=Flask(__name__)
lm=WordNetLemmatizer()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form1')
def form1():
    return render_template('form1.html')

@app.route('/thankyou')
def thankyou():
    first=request.args.get('first')
    second = request.args.get('second')
    corpus=[]
    review=first
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

    review = second
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    print(type(corpus))
    print(corpus)

    filename = "TFIDF.pkl"
    file_obj = open(filename, 'rb')
    transformer = pickle.load(file_obj)
    X_tfidf = transformer.transform(corpus)
    filename= "MULTI_NOMIAL.pkl"
    file_obj=open(filename, 'rb')
    classifier = pickle.load(file_obj)
    y_p = classifier.predict(X_tfidf)
    print(y_p)
    first = y_p[0]
    second = y_p[1]
    return render_template('thankyou.html',first=first, second=second)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'),404

if __name__=='__main__':
    app.run()
