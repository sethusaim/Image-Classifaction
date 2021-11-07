from flask import Flask, render_template, request
from predict import get_prediction

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def start():
    if request.method == "GET":
        return render_template("index.html")

    if request.method == "POST":
        print(request.files)

        if "file" not in request.files:
            print("File not uploaded")

        file = request.files["file"]

        image = file.read()

        result = get_prediction(image_bytes=image)

        return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
