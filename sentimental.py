from flask import Flask
from flask import render_template, flash, redirect
from forms import PredictForm, TrainForm
from online_classifier import OnlineClassifier
import re, os

app = Flask(__name__)

app.config.from_object('config')

root_path = app.config['SAVE_PATH']

classifier = OnlineClassifier()
print 'initializing from movie reviews'
classifier.initialize_from(random_sample=50, url="https://dl.dropboxusercontent.com/u/9015381/notebook/movie_reviews.txt")
print 'initializing from twitter'
classifier.initialize_from(random_sample=5000, url="https://dl.dropboxusercontent.com/u/9015381/notebook/twitter.txt")

pattern = re.compile('[\W_]+')

@app.route('/', methods = ['GET', 'POST'])
def home():
    form = PredictForm()
    if form.validate_on_submit():
        cleaned = pattern.sub(' ', form.example.data.lower())
        new_examples = [cleaned]
        predictions, probs = classifier.predict(new_examples)
        return render_template('result.html',
                            max_coef = classifier.get_max_coefficient(),
                            words = classifier.get_coefficients_for(new_examples[0]),
                            example = new_examples[0],
                            pos_prob = probs[1],
                            neg_prob = probs[0],
                            form = TrainForm())

    return render_template('index.html',
        form = form)

@app.route('/train', methods = ['POST'])
def train():
    form = TrainForm()

    if form.validate_on_submit():
        classifier.train([form.data['example']], [form.data['label']])
        flash('Thanks!', 'success')
    else:
        flash('An error occurred!', 'danger')
    return redirect('/')

@app.route('/status')
def status():
    num_words = 100
    coefficients = sorted(classifier.get_coefficients(num_words).items(), key=lambda x: x[1], reverse=True)
    return render_template('status.html',
                           num_words = num_words,
                           coefficients = coefficients,
                           vocab_size = len(classifier.vocabulary))

if __name__ == '__main__':
    app.run(debug=False)
