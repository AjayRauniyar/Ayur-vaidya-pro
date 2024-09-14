from flask import render_template, request, Blueprint
from PIL import Image
import pytesseract as pt
import os
import google.generativeai as genai
import textwrap
import markdown
from dotenv import load_dotenv

load_dotenv()

app1 = Blueprint('app1', __name__, template_folder='templates')

# Configure Tesseract
pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure Generative AI
generation_config = {
    "temperature": 0.3,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]



genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


model = genai.GenerativeModel('gemini-1.0-pro', generation_config=generation_config, safety_settings=safety_settings)


def to_markdown(text):
    text = text.replace('-', ' *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

def get_prescription(image_path):
    text = pt.image_to_string(Image.open(image_path))
    lines = text.splitlines()
    structured_text = "\n".join(lines)
    return structured_text

def generate(prompt):
    template = f'''
    Act as a specialized doctor and pharmacist with extensive knowledge of medicines. Below is an OCR-extracted text from a printed prescription:
    {prompt}

    1. Extract data about the Doctor, Clinic, and Patient information.
    2. Extract the names of the medicines, their dosages, timings, and duration from the prescription. Present this information clearly and neatly in normal paragraph format.
    3. Provide a brief description of each medicine mentioned, including its uses and any important information.
    4. Offer suggestions on how each medicine should be taken, including any precautions or tips.

    If the text provided does not appear to be text from a prescription or seems out of context, respond with: "Please provide me with your prescription so that I can assist you." Generate your response accordingly.
    '''
    response = model.generate_content([template])
    return markdown.markdown(response.text)

@app1.route('/')
def medsign():
    return render_template('index2.html')

@app1.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return render_template('index2.html', result="Error: No file part")

    file = request.files['image']
    if file.filename == '':
        return render_template('index2.html', result="Error: No selected file")

    if file:
        # Create an uploads directory if it doesn't exist
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Save the uploaded file
        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        # Process the file
        pres = get_prescription(filepath)
        res = generate(pres)
        os.remove(filepath)  

        return render_template('index2.html', result=res)

    return render_template('index2.html', result="Error: File upload failed")
