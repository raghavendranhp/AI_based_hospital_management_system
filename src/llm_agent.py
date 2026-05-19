import os
from dotenv import load_dotenv
from groq import Groq

class InsightGenerator:
    """
    handles integration with the groq api for generating insights.
    """

    def __init__(self, prompt_file: str = "system_prompt.txt"):
        """
        initializes the insight generator.
        """
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model_name = "llama-3.1-8b-instant"
        
        with open(prompt_file, "r") as file:
            self.system_prompt = file.read()
            
    def generate_insights(self, data_summary: str) -> str:
        """
        sends data summary to the groq api and returns actionable insights.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"analyze the following hospital data summary and provide insights:\n\n{data_summary}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3
        )
        
        return response.choices[0].message.content
