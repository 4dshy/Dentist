import json
from groq import Groq
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()
Api_key = os.getenv("GROQ_API_KEY")
if not Api_key:
    raise Exception("API key not found. Ensure GOOGLE_API_KEY is set in your .env file.")
# Initialize Groq client
client = Groq(api_key=Api_key)

# System prompt with structured instructions
SYSTEM_PROMPT = """
You are a dental data extraction assistant. Your task is to analyze dental examination transcripts and output structured JSON data for real-time updates.

Follow these rules:
1. Extract ONLY the following categories:
   - Tooth Presence
   - Mobility Grade
   - Furcation Involvement
   - Gingival Margin
   - Probing Depth
   - Bleeding on Probing
   - Plaque Index

2. Output format guidelines:
   - Always use the exact JSON structures provided
   - Use tooth numbers 1-32 (FDI system)
   - Only include mentioned attributes
   - Use null for missing sections
   - Output ONE JSON object per observation
   - Multiple observations should be separate JSON objects

3. Processing rules:
   - Ignore narrative text and non-data sentences
   - Interpret implied positions (e.g., "mesial" -> mesial section)
   - Convert mm mentions to numbers (e.g., "3mm" → 3)
   - Assume standard positions if unspecified (distal, central, mesial)

Examples of valid outputs:
{}
""".format(json.dumps([
    {
        "tooth_number": "14",
        "mobility_grade": 1
    },
    {
        "tooth_number": "14",
        "buccal": {
            "distal": {"probing_depth": 3},
            "mesial": {"probing_depth": 3}
        }
    }
]))

async def process_transcript_chunk(chunk):
    """Process a transcript chunk with Groq API"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chunk}
            ],
            model="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )

        response = chat_completion.choices[0].message.content
        return json.loads(response)
    
    except json.JSONDecodeError:
        print(f"Failed to parse JSON response: {response}")
        return None
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None

async def live_processing(transcript):
    """Simulate real-time processing with stream segmentation"""
    # In real use, you'd get chunks from your speech-to-text service
    # Here we'll split the sample transcript into logical segments
    chunks = [
        "Teeth 1 through 6 are present and healthy. However, Tooth 7 is missing.",
        "Tooth 14 has slight mobility, categorized as mobility level 1.",
        "Probing depths for Tooth 14 are 3 millimeters on both the distal and mesial surfaces.",
        "Tooth 18 has gingival recession noted at 2 millimeters on the central surface.",
        "Probing depths for Tooth 18 are recorded at 4 millimeters on the mesial surface.",
        "Tooth 19 shows some plaque buildup."
    ]

    for chunk in chunks:
        print(f"\nProcessing chunk: {chunk}")
        response = await process_transcript_chunk(chunk)
        
        if response:
            print("Extracted JSON:")
            print(json.dumps(response, indent=2))
            
            # Here you would send the JSON to your frontend
            # Example: websocket.send(json.dumps(response))
            
        # Simulate real-time delay between chunks
        await asyncio.sleep(0.1)

# Run the processing
if __name__ == "__main__":
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
    asyncio.run(live_processing(transcript))