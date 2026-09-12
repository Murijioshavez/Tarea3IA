from pathlib import Path

from flask import Flask, jsonify, render_template

from routes.api import api


PROJECT_ROOT = Path(__file__).resolve().parent

app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    UPLOAD_FOLDER=PROJECT_ROOT / "uploads",
)
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)
app.register_blueprint(api)


@app.route("/")
def home():
    return render_template("index.html")


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "El archivo supera el límite de 10 MB."}), 413


if __name__ == "__main__":
    app.run(debug=True)
