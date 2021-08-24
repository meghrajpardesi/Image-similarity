from flask import Flask, request
import os
import requests
from src.scripts.utils import post_processing, utils
from src.models.reduction_resnet import ReductionResNet

app = Flask(__name__)
model = ReductionResNet()

@app.route("/")
def index():
    return "Hello World!"


@app.route("/v1/predict", method=["GET", "POST"])
def predict():
    image1, image2 = _load_image()
    similarity_score = model.forward(image1,image2)
    res = post_processing(similarity_score)
    return res


def _load_image():
    if request.method == "POST":
        data = request.get_json()
        if data is None:
            return "no json received"
        return util.read_b64_image(data["image"], grayscale=True)
    if request.method == "GET":
        image_url = request.args.get("image_url")
        if image_url is None:
            return "no image_url defined in query string"
        logging.info("url {}".format(image_url))
        return util.read_image_pil(image_url, grayscale=True)
    raise ValueError("Unsupported HTTP method")



def _load_image():
    """ method take image from request return two image 
    Returns:
        image1, image2: two tensor image to be given to image
    """
    if request.method == "POST":
        data = request.get_json()
        if data is None:
            return "No json received"
        image1 = utils.read_b64_image(data["image1"], grayscale=True)
        image2 = utils.read_b64_image(data["image2"], grayscale=True)
    
    return image1, image2

def main():
    """Run the app..."""
    app.run(host="0.0.0.0", port=8000, debug=False)  # nosec
    
if __name__=="__main__":
    main()