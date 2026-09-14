import os
from pathlib import Path

from flask import Flask, jsonify, render_template, send_from_directory

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


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.svg", mimetype="image/svg+xml")


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "El archivo supera el límite de 10 MB."}), 413


if __name__ == "__main__":
    # Debug mode exposes an interactive console to anyone who can reach the port, so it is
    # opt-in via FLASK_DEBUG rather than always on.
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"},
    )
