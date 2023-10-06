from flask import Flask, render_template, request, redirect
import os

from src.deploiement.app.process import predict_img

source = os.getcwd()
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html', title='About', name='Passed by variable')


@app.route("/predict")
def predict():
    return render_template("predict.html", title="Predict")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    target = os.path.join(APP_ROOT, 'temp/')
    if request.method == 'POST':
        file = request.files['img']  # 'img' is the id passed in input file form field
        filename = file.filename
        # filename = filename(filename)
        file.save("".join([target, filename]))  # saving file in temp folder
        print("upload Completed")  # printing on terminal

        return redirect('/prediction/{}'.format(filename))


@app.route("/prediction/<filename>", methods=["GET", "POST"])
def prediction(filename):
    target = os.path.join(APP_ROOT, 'temp', filename)

    # imported process.py
    image_base64 = predict_img(target)
    return render_template('output.html', image_base64=image_base64)
