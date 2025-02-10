import random
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Google Gemini API
genai.configure(api_key="AIzaSyDKBHIS9J3Jhji77cH1I-p62jBsny6S-T4")

# Create a model instance
model = genai.GenerativeModel('gemini-pro')

# Define Question Types & Difficulty Levels
QUESTION_TYPES = ["multiple_choice", "true_false", "short_answer"]
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# Add a home route
@app.route('/')
def home():
    return render_template('index.html')

# Add a test route
@app.route('/api/test')
def test():
    return jsonify({
        "status": "success",
        "message": "API is working correctly"
    })

def generate_question(topic, subtopic, difficulty, question_type):
    """
    Uses Google Gemini API to generate a quiz question
    """
    system_prompt = """You are an expert educator. Generate a question following these rules:
    1. Make the question clear and concise
    2. Ensure it's appropriate for the specified difficulty level
    3. Focus on understanding rather than memorization
    4. For multiple choice, provide exactly 4 options
    5. Make sure the correct answer is clearly indicated
    
    Return the response in a structured format that can be easily parsed.
    """
    
    topic_str = f"{topic} ({subtopic})" if subtopic else topic
    user_prompt = f"""{system_prompt}

    Create a {difficulty} level {question_type.replace('_', ' ')} question about {topic_str}.
    
    Format your response exactly as follows:

    For multiple choice:
    QUESTION: [Question text]
    OPTIONS:
    A) [First option]
    B) [Second option]
    C) [Third option]
    D) [Fourth option]
    CORRECT_ANSWER: [A/B/C/D]
    EXPLANATION: [Brief explanation of why this is the correct answer]

    For true/false:
    QUESTION: [Question text]
    OPTIONS:
    A) True
    B) False
    CORRECT_ANSWER: [A/B]
    EXPLANATION: [Brief explanation of why this is the correct answer]

    For short answer:
    QUESTION: [Question text]
    CORRECT_ANSWER: [Brief correct answer]
    EXPLANATION: [Brief explanation of why this is the correct answer]
    """

    try:
        response = model.generate_content(user_prompt)
        return parse_question_response(response.text.strip(), question_type)
    except Exception as e:
        error_msg = str(e)
        print(f"API Error: {error_msg}")
        if "invalid_api_key" in error_msg.lower():
            raise Exception("Invalid API key. Please check your Google API key.")
        else:
            raise Exception(f"Failed to generate question: {error_msg}")

def parse_question_response(response_text, question_type):
    """
    Parse the AI response into a structured format
    """
    lines = response_text.split('\n')
    result = {'type': question_type}
    
    for line in lines:
        if line.startswith('QUESTION:'):
            result['question'] = line.replace('QUESTION:', '').strip()
        elif line.startswith('OPTIONS:'):
            result['options'] = []
        elif line.startswith(('A)', 'B)', 'C)', 'D)')):
            result['options'].append(line.strip())
        elif line.startswith('CORRECT_ANSWER:'):
            result['correct_answer'] = line.replace('CORRECT_ANSWER:', '').strip()
        elif line.startswith('EXPLANATION:'):
            result['explanation'] = line.replace('EXPLANATION:', '').strip()
    
    return result

@app.route('/api/generate-assessment', methods=['GET', 'POST'])
def generate_assessment():
    if request.method == 'GET':
        return jsonify({
            "status": "success",
            "message": "Send a POST request to generate an assessment"
        })

    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        topic = data.get("topic", "").strip()
        subtopic = data.get("subtopic", "").strip()
        num_questions = min(int(data.get("num_questions", 5)), 10)  # Limit to 10 questions
        difficulty = data.get("difficulty", "medium").lower()
        question_types = data.get("question_types", ["multiple_choice"])

        # Validate inputs
        if not topic:
            return jsonify({"status": "error", "message": "Topic is required"}), 400
        if difficulty not in DIFFICULTY_LEVELS:
            return jsonify({"status": "error", "message": "Invalid difficulty level"}), 400
        question_types = [qt for qt in question_types if qt in QUESTION_TYPES]
        if not question_types:
            return jsonify({"status": "error", "message": "No valid question types selected"}), 400

        # Generate questions
        questions = []
        successful_questions = 0

        for _ in range(num_questions):
            try:
                question_type = random.choice(question_types)
                question_data = generate_question(topic, subtopic, difficulty, question_type)
                questions.append({
                    **question_data,
                    "difficulty": difficulty,
                    "error": False
                })
                successful_questions += 1
            except Exception as e:
                questions.append({
                    "question": str(e),
                    "type": question_type,
                    "difficulty": difficulty,
                    "error": True
                })

        if successful_questions == 0:
            return jsonify({
                "status": "error",
                "message": "Failed to generate any questions. Please try again or check your API key."
            }), 500

        return jsonify({
            "status": "success",
            "assessment": {
                "topic": topic,
                "subtopic": subtopic,
                "questions": questions,
                "successful_questions": successful_questions
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 to make the server externally visible
    app.run(debug=True, host='0.0.0.0', port=5000)
