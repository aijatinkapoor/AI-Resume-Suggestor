import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import json


# Initialize models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = OllamaLLM(model="tinyllama")

# ---- TEXT EXTRACTION ----
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text("text")
    return text.strip()

# ---- COSINE SIMILARITY ----
def get_cosine_score(resume_text, jd_text):
    embeddings = embedder.encode([resume_text, jd_text])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(score * 100), 2)

# ---- LLM EVALUATION ----
def get_llm_score(resume_text, jd_text):
    template = """
    You are an ATS evaluator. Compare the resume and job description.

    Return JSON:
    {{
      "score": (0-100),
      "missing_keywords": [],
      "suggestions": ""
    }}

    Resume:
    {resume_text}

    Job Description:
    {jd_text}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["resume_text", "jd_text"]
    )

    chain = RunnableSequence(prompt | llm | StrOutputParser())
    result = chain.invoke({"resume_text": resume_text, "jd_text": jd_text})

    try:
        data = json.loads(result)
    except:
        data = {"score": None, "missing_keywords": [], "suggestions": "No JSON found."}
    return data


def evaluate_resume(resume_path, job_description):
    resume_text = extract_text_from_pdf(resume_path)
    cosine_score = get_cosine_score(resume_text, job_description)
    llm_result = get_llm_score(resume_text, job_description)
    return {
        "cosine_similarity_score": cosine_score,
        "llm_evaluation": llm_result
    }