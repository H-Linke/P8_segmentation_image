# # Étape 2 : Créez une API Flask
#
# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# import os
# from src.travail.multiclassunet import Unet, unet_view
#
# app = Flask(__name__)
#
# ##############################################
# #      Chargement variables et modèles       #
# ##############################################
#
# source = os.getcwd()
# model_path = os.path.join(source, "src/model/weights_model_final.h5").replace("\\", "/")
#
# model = unet_view(Unet(256, 512))
# model.load_weights(model_path)
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Obtenez les données de la requête POST
#         data = request.json  # Vous pouvez également utiliser request.form, selon le format de données
#
#         # Effectuez des prédictions avec le modèle
#         # Assurez-vous que vos données d'entrée correspondent aux attentes du modèle
#         input_data = np.array(data['input_data'])  # Adapté à votre modèle
#
#         # Effectuez la prédiction
#         predictions = model.predict(input_data)
#
#         # Préparez la réponse
#         response = {'predictions': predictions.tolist()}
#
#         return jsonify(response)
#
#     except Exception as e:
#         return jsonify({'error': str(e)})
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
#

from flask import Flask, render_template
import os

source = os.getcwd().replace("\\", "/")
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

predict_path = "{}/src/deploiement/app/templates/predict.html".format(source)


app = Flask(__name__)

# @app.route("/<name>")
# def hello(name):
#     return f"Hello, {name}! \n Nous sommes ici : {predict_path}"




@app.route("/predict")
def predict():
    return render_template(predict_path, title="Predict")

