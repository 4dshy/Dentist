import os
from dotenv import load_dotenv
import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

def load_json_template(file_path):
    """Load the JSON template file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception(f"JSON file not found at {file_path}. Please check the file path.")
    except json.JSONDecodeError:
        raise Exception(f"Invalid JSON format in {file_path}. Please verify the file contents.")

def extract_data_from_transcript(transcript, json_template):
    """Extract structured data from the transcript using the specified JSON template."""
    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise Exception("API key not found. Ensure GOOGLE_API_KEY is set in your .env file.")

    # Set up the Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

    # Define the prompt template to extract structured data
    prompt = PromptTemplate(
        input_variables=["transcript", "json_template"],
        template=(
            "You are a dental assistant AI. Analyze the following transcript and extract the relevant dental data. "
            "The extracted data must match the provided JSON format exactly. If any information is missing, replace it with null. "
            "Here is the required JSON format:\n{json_template}\n\n"
            "Now, analyze this transcript:\n{transcript}\n"
            "Provide the output strictly in JSON format."
        ),
    )

    # Set up the LLM chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the transcript through the chain
    output = chain.run(transcript=transcript, json_template=json.dumps(json_template))
    return output

if __name__ == "__main__":
    # Example transcript input
    transcript = (
        "Alright, let's begin. Today, we have a pre-operation check for our patient, Sarah Johnson. "
        "She is a 28-year-old female. The date of this examination is January 24, 2025. \n\n"
        "Starting with the upper jaw, let's go tooth by tooth. Teeth 1 through 6 are present and healthy. However, Tooth 7 is missing. "
        "Moving further, Teeth 8 through 13 are also in good condition. For Tooth 14, there is slight mobility, categorized as mobility level 1. "
        "Probing depths for Tooth 14 are 3 millimeters on both the distal and mesial surfaces. Additionally, there was some bleeding noted during probing. "
        "Teeth 15 and 16 are fine, with no issues observed. \n\n"
        "Now, moving to the lower jaw. All teeth, from Tooth 17 through Tooth 32, are present, except for Tooth 31, which has an implant. "
        "Focusing on Tooth 18, there is gingival recession noted at 2 millimeters on the central surface. Probing depths for Tooth 18 are recorded at 4 millimeters on the mesial surface. "
        "Tooth 19 shows some plaque buildup, while the remaining teeth appear to be healthy. \n\n"
        "In summary, moderate plaque buildup was observed on the molars, especially in the lower jaw. Tooth 14 requires attention for slight mobility and bleeding during probing. "
        "I recommend improving Sarah's oral hygiene routine and scheduling a follow-up appointment to determine whether Tooth 14 needs a filling."
    )

    # Load the JSON template
    json_template_path = "output_temp.json"
    json_template = load_json_template(json_template_path)

    # Extract data from the transcript
    try:
        extracted_data = extract_data_from_transcript(transcript, json_template)
        print("Extracted Data:\n", extracted_data)
    except Exception as e:
        print(f"Error: {e}")
