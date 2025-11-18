
# The main orchestrator class that runs the full pipeline.

import utils
import llm_interface
from retrieval import HybridRetriever
from scoring import ScoringEngine
import config
from typing import List
import traceback

class CandidateMatchingSystem:
    def __init__(self):
        self.job = None
        self.candidates_db = {} 
        self.retriever = HybridRetriever()
        self.scorer = ScoringEngine()
        print("CandidateMatchingSystem initialized.")

    def process_job_posting(self, job_file: str):
        """Loads and parses the job posting."""
        job_text = utils.extract_text(job_file)
        if not job_text:
            raise Exception("Job posting file is empty or unreadable.")
        
        self.job = llm_interface.parse_job_posting(job_text)
        if not self.job:
            raise Exception("LLM failed to parse job posting.")
        
        print(f"Successfully parsed job: {self.job.job_title}")

    def process_resumes(self, resume_files: List[str]):
        """Loads and parses all candidate resumes."""
        self.candidates_db = {}
        
        print(f"Starting processing for {len(resume_files)} resumes...")
        
        for f in resume_files:
            try:
                print(f"Processing file: {f}")
                resume_text = utils.extract_text(f)
                if not resume_text:
                    print(f"Skipping empty or unreadable file: {f}")
                    continue
                
                parsed_resume = llm_interface.parse_resume(resume_text)
                if parsed_resume:
                    file_id = f.split('/')[-1]
                    self.candidates_db[file_id] = parsed_resume
                    print(f"Successfully parsed: {file_id}")
                else:
                    print(f"LLM failed to parse resume: {f}")
            except Exception as e:
                print(f"Error processing resume {f}: {str(e)}")
                traceback.print_exc()
                continue 
        
        print(f"Successfully parsed {len(self.candidates_db)} out of {len(resume_files)} resumes.")

    def run_matching_pipeline(self) -> List[dict]:
        """
        Runs the full retrieve -> re-rank -> explain pipeline.
        """
        if not self.job or not self.candidates_db:
            raise Exception("Job and resumes must be processed first.")
            
        corpus = [res.full_text_summary for res in self.candidates_db.values()]
        corpus_ids = list(self.candidates_db.keys())
        
        if not corpus:
             return []

        self.retriever.index(corpus, corpus_ids)
        

        k_to_retrieve = min(len(corpus_ids), config.TOP_K_RETRIEVAL)
        query = self.job.responsibilities_summary
        
        candidate_ids_to_rank = self.retriever.search(query, top_k=k_to_retrieve)
        
        print(f"Re-ranking {len(candidate_ids_to_rank)} candidates...")
        
        reports = []
        for candidate_id in candidate_ids_to_rank:
            candidate = self.candidates_db.get(candidate_id)
            if not candidate:
                print(f"Warning: Could not find candidate for ID {candidate_id}")
                continue
                
            try:
                report = self.scorer.score_candidate(self.job, candidate)
                report["filename"] = candidate_id
                reports.append(report)
            except Exception as e:
                print(f"Error scoring candidate {candidate_id}: {e}")
                continue

        explained_reports = []
        for report in reports:
            try:
                explained_report = llm_interface.generate_explanation(report)
                explained_reports.append(explained_report)
            except Exception as e:
                 print(f"Error generating explanation for {report.get('filename')}: {e}")
                 explained_reports.append(report)
            
        sorted_reports = sorted(explained_reports, key=lambda r: r['final_score'], reverse=True)
        
        return sorted_reports

print("File 'matching_system.py' (Final Robust Version) created.")
