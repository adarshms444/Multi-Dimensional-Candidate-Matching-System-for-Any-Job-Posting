
import os
import json
from openai import OpenAI
from pydantic import BaseModel
from typing import Type
import config

from tenacity import retry, stop_after_attempt, wait_random_exponential

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def robust_llm_call(prompt: str, response_model: Type[BaseModel], model: str):
    """
    A robust function to call an LLM and parse the output into a Pydantic model
    using 'response_format'.
    """
    try:
        print(f"LLM Call: Pydantic parsing with {model}...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"} 
        )
        
        content = response.choices[0].message.content
        arguments = json.loads(content) 
        
        return response_model.model_validate(arguments)
        
    except Exception as e:
        print(f"Error in LLM call or JSON parsing: {e}")
        print(f"Model: {model}, Prompt: {prompt[:100]}...")
        raise e 

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def generative_llm_call(prompt: str, model: str):
    """
    A robust function for a simple generative LLM call expecting JSON.
    """
    try:
        print(f"LLM Call: Generative explanation with {model}...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in generative LLM call: {e}")
        raise e 

def parse_job_posting(job_text: str) -> config.ParsedJob:
    """Uses LLM to structure a job posting."""
    schema = config.ParsedJob.model_json_schema()
    prompt = config.PROMPT_PARSE_JOB.format(
        schema=json.dumps(schema, indent=2),
        job_text=job_text
    )
    return robust_llm_call(prompt, config.ParsedJob, config.LLM_PARSING_MODEL)

def parse_resume(resume_text: str) -> config.ParsedResume:
    """Uses LLM to structure a resume."""
    schema = config.ParsedResume.model_json_schema()
    prompt = config.PROMPT_PARSE_RESUME.format(
        schema=json.dumps(schema, indent=2),
        resume_text=resume_text
    )
    return robust_llm_call(prompt, config.ParsedResume, config.LLM_PARSING_MODEL)

def generate_explanation(report_data: dict) -> dict:
    """Uses LLM to generate the final human-readable report."""
    prompt = config.PROMPT_EXPLAIN_MATCH.format(**report_data)
    
    try:
        explanation_json = generative_llm_call(prompt, config.LLM_EXPLAIN_MODEL)
        report_data.update(explanation_json)
        return report_data
    except Exception as e:
        print(f"Error in explanation LLM call (final attempt failed): {e}")
        report_data.update({
            "strengths": "Error generating AI analysis.",
            "gaps": "Error generating AI analysis.",
            "notes": "System error."
        })
        return report_data

print("File 'llm_interface.py' (Final Version with Import Fix) created.")
