from flask import Flask, render_template, request, jsonify
from rag import evaluate_resume
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        jd_text = request.form["job_description"]
        resume_file = request.files["resume"]

        # Save resume temporarily
        save_path = os.path.join("uploads", resume_file.filename)
        os.makedirs("uploads", exist_ok=True)
        resume_file.save(save_path)

        # Call your RAG pipeline
        result = evaluate_resume(save_path, jd_text)
        return render_template("index.html", score=result["cosine_similarity_score"], llm=result["llm_evaluation"])

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
