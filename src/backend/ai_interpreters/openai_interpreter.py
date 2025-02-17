import os
import openai
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class OpenAIDreamInterpreter:
    def __init__(self, api_key: Optional[str] = None):
        # Directly get the API key from environment variables
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        # Validate API key
        if not self.api_key:
            raise ValueError(
                "No OpenAI API Key found. "
                "Please set OPENAI_API_KEY in your .env file."
            )
        
        # Configure the OpenAI API
        openai.api_key = self.api_key

    def interpret_dream(self, description: str) -> str:
        try:
            # Craft a detailed prompt for dream interpretation
            prompt = f"""
            You are an expert dream psychologist. Provide a deep, insightful, and compassionate 
            interpretation of the following dream description. Include:
            1. Possible symbolic meanings
            2. Potential psychological insights
            3. Emotional undertones
            4. Constructive reflections

            Dream Description:
            {description}

            Interpretation:
            """

            # Generate interpretation using OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Return the text response
            return response['choices'][0]['message']['content'].strip() or "Unable to generate interpretation"

        except Exception as e:
            print(f"Error in dream interpretation: {e}")
            return f"An error occurred during interpretation: {str(e)}"

    def generate_dream_title(self, description: str) -> Optional[str]:
        """
        Generate a creative, evocative title for the dream
        """
        try:
            prompt = f"""
            Create a unique, poetic, and intriguing title for this dream description:
            {description}

            Title should be:
            - Concise (5-7 words)
            - Metaphorical
            - Capture the dream's essence
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response['choices'][0]['message']['content'].strip() or None

        except Exception as e:
            print(f"Error generating dream title: {e}")
            return None
