
from pydantic import BaseModel, Field
from typing import List, Dict

LLM_PARSING_MODEL = "mistralai/mistral-7b-instruct:free"
LLM_EXPLAIN_MODEL = "mistralai/mistral-7b-instruct:free"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K_RETRIEVAL = 10 

SCORING_WEIGHTS = {
    "must_have_skills": 0.35,
    "important_skills": 0.25,
    "nice_to_have_skills": 0.10,
    "experience_relevance": 0.15,
    "recency": 0.10,
    "domain_match": 0.05
}


class SkillCluster(BaseModel):
    must_have: List[str]
    important: List[str]
    nice_to_have: List[str]
    implicit_skills: List[str]

class ParsedJob(BaseModel):
    job_title: str
    required_years_experience: int
    skills: SkillCluster
    domain_keywords: List[str] = Field(default_factory=list, description="Industry keywords (e.g., 'FinTech', 'Healthcare')")
    responsibilities_summary: str

class ExperienceEntry(BaseModel):
    title: str
    company: str
    start_date: str
    end_date: str 
    description: str

class ParsedResume(BaseModel):
    name: str
    total_years_experience: int
    skills: List[str]
    education: List[str] = Field(default_factory=list, description="Degrees or certifications (e.g., 'B.S. in Computer Science')")
    domain_keywords: List[str] = Field(default_factory=list, description="Industry keywords (e.g., 'FinTech', 'E-commerce')")
    experience: List[ExperienceEntry]
    full_text_summary: str

class Explanation(BaseModel):
    overall_fit_score: int = Field(..., description="The final score from 0-100")
    confidence: str = Field(..., description="High, Medium, or Low")
    strengths: str = Field(..., description="A string (use bullet points) of what the candidate matches well.")
    gaps: str = Field(..., description="A string (use bullet points) of what the candidate seems to be missing.")
    notes: str = Field(..., description="A brief, overall recommendation.")


# -System Prompts -
PROMPT_PARSE_JOB = """
You are an expert JSON-generating HR analyst. You must analyze the job posting
and return *only* a valid JSON object. The JSON object must strictly
adhere to the following Pydantic JSON Schema:

```json
{schema}
```

Job Posting Text:
---
{job_text}
---

Respond *only* with the JSON. Do not add any other text.
"""

PROMPT_PARSE_RESUME = """
You are an expert JSON-generating resume parser. You must analyze the resume
and return *only* a valid JSON object. The JSON object must strictly
adhere to the following Pydantic JSON Schema:

```json
{schema}
```

Resume Text:
---
{resume_text}
---

Respond *only* with the JSON. Do not add any other text.
"""

PROMPT_EXPLAIN_MATCH = """
You are a helpful recruiting assistant. A job posting and a candidate's profile
have been scored across 6 dimensions. Your job is to generate a final report
as a JSON object.

**Data:**
- Job Summary: {job_summary}
- Resume Summary: {resume_summary}
- Score (Must-Have): {must_have_score}/100
- Score (Important): {important_score}/100
- Score (Experience): {experience_score}/100
- Score (Recency): {recency_score}/100
- Score (Domain): {domain_score}/100
- Final Score: {final_score}/100

Your task is to generate a final report for the hiring manager.
The report must be a JSON object with 5 keys:
'overall_fit_score' (int), 'confidence' (string), 'strengths' (string),
'gaps' (string), and 'notes' (string).
"""
print("File 'config.py' created.")
