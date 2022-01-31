from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image
import os, io, sys
import numpy as np
import cv2
import base64
from tfhub_styletransfer_wrapper import StyleHub, save_image, load_image


app = Flask(__name__)
# cors = CORS(app)


############################################## THE REAL DEAL ###############################################
@app.route('/styleImage', methods=['POST'])
# @cross_origin(origin='*')
def mask_image():
    if request.method == 'POST':
        # print(request.files , file=sys.stderr)
        file = request.files['input_filename'].read()  ## byte file
        npimg = np.fromstring(file, np.uint8)
        img_content = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img_content = cv2.cvtColor(img_content, cv2.COLOR_BGR2RGB)
        img_content = Image.fromarray(img_content)
        img_content.save('/tmp/_content.jpg', "JPEG")
        img_content = load_image("/tmp/_content.jpg", (512, 512))
        save_image(img_content, "/tmp/_content.jpg")

        file = request.files['style_filename'].read()
        npimg = np.fromstring(file, np.uint8)
        img_style = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img_style = cv2.cvtColor(img_style, cv2.COLOR_BGR2RGB)
        img_style = Image.fromarray(img_style)
        img_style.save('/tmp/_style.jpg', "JPEG")
        img_style = load_image("/tmp/_style.jpg", (256, 256))
        save_image(img_style, "/tmp/_style.jpg")

        stylehub = StyleHub()
        stylehub.load_content("/tmp/_content.jpg", 512)
        stylehub.load_style("/tmp/_style.jpg", 256)
        img_stylized = stylehub.evaluate(False)
        save_image(img_stylized, "/tmp/_stylized.jpg")


        ################################################
        img_content = Image.open("/tmp/_content.jpg")
        rawBytes = io.BytesIO()
        img_content.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64_content = base64.b64encode(rawBytes.read())

        img_style = Image.open("/tmp/_style.jpg")
        rawBytes = io.BytesIO()
        img_style.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64_style = base64.b64encode(rawBytes.read())

        img_stylized = Image.open("/tmp/_stylized.jpg")
        rawBytes = io.BytesIO()
        img_stylized.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64_stylized = base64.b64encode(rawBytes.read())

        response = jsonify({'content': str(img_base64_content),
                            'style': str(img_base64_style),
                            'stylized': str(img_base64_stylized)})
        # response.headers.add("Access-Control-Allow-Origin", "*")
        # response.headers.add_header("Access-Control-Allow-Origin", "*")
        return response


##################################################### THE REAL DEAL HAPPENS ABOVE ######################################

@app.route('/test', methods=['GET', 'POST'])
def test():
    print("log: got at test", file=sys.stderr)
    return jsonify({'status': 'succces'})


@app.route('/home')
def home():
    return render_template('index.jinja2')


@app.after_request
# @cross_origin(origin='*')
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    response.headers.add_header('Access-Control-Allow-Origin', '*')
    response.headers.add_header('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add_header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
    app.run(debug=True)