import argparse
import base64
import json
import cv2

import numpy as np
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import model_from_json

sio = socketio.Server()
app = Flask(__name__)
model = None
output_shape = (20, 40)

def preprocess_image(image, output_shape):
    """
    Same pre-process as what did in traning
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    random_bright = .25 * np.random.uniform()
    image[:,:,2] = image[:,:,2] * random_bright
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    shape = image.shape
    image = image[65:140, 0:320]

    kernel_size = 5
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    image = cv2.resize(image, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_AREA)

    return image

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    image_array = preprocess_image(image_array, output_shape=output_shape) # pre process image here
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.3
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', default='model.json', type=str, help='Path to model definition json. Model weights should be on the same path.')

    args = parser.parse_args()

    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)
    model.summary()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
