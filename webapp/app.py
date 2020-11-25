import os
import tempfile

import scipy
from matplotlib import pyplot
from numpy import resize, reshape

from evaluate import Evaluator
from scipy import ndimage, misc

from flask import Flask, render_template, request

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

evaluator = Evaluator("../checkpoints/models/model_108000.h5")

input_path = 'static/input.png'
output_path = 'static/output.png'

@app.route('/')
@app.route('/gan')
def gan():
    return render_template("gan.html")


@app.route('/gan/upload', methods=['POST'])
def gan_upload():
    is_image = request.args.get('type') == 'image'
    file = request.files['file']
    file.save(input_path)

    if is_image:
        generated_image_arr = evaluator.predict(input_path, (512, 512), plot=False)
        pyplot.imsave(output_path, generated_image_arr)
    else:
        raise Exception("Unsupported file type")

    return render_template("gan.html")


if __name__ == '__main__':
    app.run(port=9001)
