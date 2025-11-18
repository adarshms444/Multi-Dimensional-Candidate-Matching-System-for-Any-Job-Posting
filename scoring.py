
import config
import re
from datetime import datetime

class ScoringEngine:
    def __init__(self):
        self.weights = config.SCORING_WEIGHTS
        print("ScoringEngine initialized.")

    def _score_skills(self, required: list, candidate_keywords: list) -> float:
        """
        --- CORRECTED LOGIC ---
        Scores skill match from 0 to 100.
        Checks if any candidate keyword (e.g., "Python", "B.S. in CS") exists
        inside the job requirement phrase (e.g., "5+ years of Python experience").
        """
        if not required:
            return 100.0  
        
        matched_count = 0
        candidate_keywords_lower = [s.lower() for s in candidate_keywords]
        
        for req_phrase in required:
            req_phrase_lower = req_phrase.lower()
            for skill_keyword in candidate_keywords_lower:
                if skill_keyword in req_phrase_lower:
                    matched_count += 1
                    break 
                
        return (matched_count / len(required)) * 100

    def _score_experience(self, required_years: int, candidate_years: int) -> float:
        """Scores experience match from 0 to 100 (capped at 100)."""
        if required_years == 0:
            return 100.0
        
        if candidate_years >= required_years:
            return 100.0
        else:
            return (candidate_years / required_years) * 100

    def _score_recency(self, experience_entries: list) -> float:
        """Scores recency. 100 if a job ended in the last 2 years or is 'Present'."""
        if not experience_entries:
            return 0.0
        
        today = datetime.today()
        for entry_data in experience_entries:
            end_date = entry_data.get('end_date', '').lower()
            if end_date == 'present':
                return 100.0
            
            try:
                end_year_match = re.search(r'(\d{4})', end_date)
                if end_year_match:
                    end_year = int(end_year_match.group(1))
                    if (today.year - end_year) <= 2:
                        return 100.0
            except:
                continue
                
        return 30.0

    def _score_domain(self, job_keywords: list, resume_keywords: list) -> float:
        """
        --- NEW, ROBUST LOGIC ---
        Scores domain match. Checks for any overlap between keyword lists.
        """
        if not job_keywords:
            return 50.0 
        
        job_set = set(k.lower() for k in job_keywords)
        resume_set = set(k.lower() for k in resume_keywords)
        
        if job_set.intersection(resume_set):
            return 100.0
        else:
            return 0.0

    def score_candidate(self, job: config.ParsedJob, resume: config.ParsedResume) -> dict:
        """Runs the full 6-dimension scoring for a single candidate."""
        print(f"Re-ranking candidate: {resume.name}")
        
        job_skills = job.skills.model_dump()
        resume_experience = [exp.model_dump() for exp in resume.experience]
        
        candidate_keywords = resume.skills + resume.education
        
        scores = {
            "must_have_score": self._score_skills(job_skills['must_have'], candidate_keywords),
            "important_score": self._score_skills(job_skills['important'], candidate_keywords),
            "nice_to_have_score": self._score_skills(job_skills['nice_to_have'], candidate_keywords),
            "experience_score": self._score_experience(job.required_years_experience, resume.total_years_experience),
            "recency_score": self._score_recency(resume_experience),
            "domain_score": self._score_domain(job.domain_keywords, resume.domain_keywords)
        }
        
        final_score = (
            scores["must_have_score"] * self.weights["must_have_skills"] +
            scores["important_score"] * self.weights["important_skills"] +
            scores["nice_to_have_score"] * self.weights["nice_to_have_skills"] +
            scores["experience_score"] * self.weights["experience_relevance"] +
            scores["recency_score"] * self.weights["recency"] +
            scores["domain_score"] * self.weights["domain_match"]
        )
        
        report_data = {
            "name": resume.name,
            "final_score": round(final_score, 2),
            "job_summary": job.responsibilities_summary,
            "resume_summary": resume.full_text_summary,
            **scores
        }
        
        return report_data

print("File 'scoring.py' (Final Version with Corrected Logic) created.")
