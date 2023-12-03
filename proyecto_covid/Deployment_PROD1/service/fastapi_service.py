# Developed by Mirko J. Rodríguez mirko.rodriguezm@gmail.com

from fastapi import FastAPI, File, UploadFile
from starlette.middleware.cors import CORSMiddleware
from werkzeug.utils import secure_filename
import json
import numpy as np
from tensorflow.keras.preprocessing import image
import requests

UPLOAD_FOLDER = 'uploads/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get('/')
def main_page():
    return 'REST service is active via FastAPI'

@app.post("/model/predict/")
async def predict(file: UploadFile = File(...)):
    data = {"success": False}
    filename = file.filename
    if file and allowed_file(filename):
        print("\nFilename received:", filename)
        contents = await file.read()
        filename = secure_filename(filename)
        tmpfile = ''.join([UPLOAD_FOLDER, '/', filename])
        with open(tmpfile, 'wb') as f:
            f.write(contents)
        print("\nFilename stored:", tmpfile)

        # Model parameters
        model_name = 'covid'  # Replace with your model name
        model_version = '1'  # Replace with your model version
        port_HTTP='9501'
        port_gRPC='9500'       

        predictions = predict_via_HTTP(tmpfile, model_name, model_version, port_HTTP)
        # predictions = predict_via_gRPC(tmpfile, model_name, model_version, port_gRPC)

        index = np.argmax(predictions)
        CLASSES = ['covid', 'normal', 'neumonia']  # Adjust according to your model classes
        ClassPred = CLASSES[index]
        ClassProb = predictions[index]

        print("Index:", index)
        print("Pred:", ClassPred)
        print("Prob:", ClassProb)

        # Results as Json
        data["predictions"] = []
        r = {"label": ClassPred, "score": float(ClassProb)}
        data["predictions"].append(r)

        # Success
        data["success"] = True

    return data

def predict_via_HTTP(image_to_predict, model_name, model_version, port):
    print("\nImage:", image_to_predict)
    print("Model:", model_name)
    print("Model version:", model_version)
    print("Port:", port)

    test_image = image.load_img(image_to_predict, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image.astype('float32')
  

    data = json.dumps({"signature_name": "serving_default", "instances": test_image.tolist()})
    headers = {"content-type": "application/json"}
    uri = ''.join(['http://127.0.0.1:', port, '/v', model_version, '/models/', model_name, ':predict'])
    print("URI:", uri)

    json_response = requests.post(uri, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions'][0]
    print("\npredictions:", predictions)

    return predictions
