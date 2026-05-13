import os
import tempfile
from flask import Flask, request, jsonify
from analyze_video import analyze

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://www.skillsift.xyz"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Analyzer-Secret"
    return response

SECRET = os.environ.get("ANALYZER_SECRET", "")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/analyze-video", methods=["POST"])
def analyze_video():
    if SECRET and request.headers.get("X-Analyzer-Secret") != SECRET:
        return jsonify({"error": "Unauthorized"}), 401

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        result = analyze(tmp_path)
        return jsonify(result)
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({"error": "Analysis failed"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
