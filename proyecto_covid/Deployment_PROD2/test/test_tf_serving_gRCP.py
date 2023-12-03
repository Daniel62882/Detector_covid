# -----------------------------------
# Testing TensorFlow Serving via gRPC
# Author: Mirko Rodriguez
# -----------------------------------

import grpc
import argparse
import requests
import json

from tensorflow.python.framework import tensor_util
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Args
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image PATH is required.")
ap.add_argument("-m", "--model", required=True, help="Model NAME is required.")
ap.add_argument("-v", "--version", required=True, help="Model VERSION is required.")
ap.add_argument("-p", "--port", required=True, help="Model PORT number is required.")
args = vars(ap.parse_args())

image_path = args['image']
model_name = args['model']
model_version = args['version']
port = args['port']

print("\nModel:", model_name)
print("Model version:", model_version)
print("Image:", image_path)
print("Port:", port)

# Loading image
test_image = image.load_img(image_path, target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = test_image.astype('float32')
test_image = test_image / 255.0

# Create gRPC client and request
channel = grpc.insecure_channel(f"127.0.0.1:{port}")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = model_name
request.model_spec.version.value = int(model_version)
request.model_spec.signature_name = "serving_default"
request.inputs['vgg16_input'].CopyFrom(tensor_util.make_tensor_proto(test_image, shape=[1] + list(test_image.shape)))

# Send request
result_predict = stub.Predict(request, timeout=10.0)
result_dict = {}
for key in result_predict.outputs:
    tensor_proto = result_predict.outputs[key]
    result_dict[key] = tensor_util.MakeNdarray(tensor_proto)

# Extract predictions
predictions = result_dict['dense_3']
print("\npredictions:", predictions)

# Process predictions
index = int(predictions.argmax())
CLASSES = ['covid', 'normal', 'neumonia']  # Ajustar según las clases de tu modelo
ClassPred = CLASSES[index]
ClassProb = float(predictions[0][index])

print("Index:", index)
print("Pred:", ClassPred)
print("Prob:", ClassProb)
