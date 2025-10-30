from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the Flask app
app = Flask(__name__)

# Load the fast DistilBART model for summarization
try:
    logging.info("Loading DistilBART summarization model...")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    logging.info("DistilBART model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    summarizer = None

@app.route('/')
def index():
    """Renders the main web page."""
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    """API endpoint that handles the text summarization request."""
    if summarizer is None:
        return jsonify({'error': 'Summarization model is not available.'}), 500

    try:
        data = request.get_json()
        input_text = data.get('text', '')
        summary_length_option = data.get('summary_length', 'medium') # Default to medium if not provided

        if not isinstance(input_text, str) or len(input_text.strip()) == 0:
            return jsonify({'error': 'Input text must be a non-empty string.'}), 400
        
        # --- Set summary length parameters based on user's choice ---
        length_map = {
            "short": {"min_length": 20, "max_length": 60},
            "medium": {"min_length": 50, "max_length": 130},
            "long": {"min_length": 100, "max_length": 200}
        }
        selected_length = length_map.get(summary_length_option, length_map["medium"])
        min_len = selected_length["min_length"]
        max_len = selected_length["max_length"]

        # --- Chunking logic to handle long texts ---
        max_chunk = 750 # Roughly 750 words to stay within 1024 token limit for most models
        words = input_text.split()
        text_chunks = [' '.join(words[i:i + max_chunk]) for i in range(0, len(words), max_chunk)]
        
        logging.info(f"Input text split into {len(text_chunks)} chunk(s). Summarizing with length: {summary_length_option}")

        # Summarize each chunk of text
        summaries = summarizer(text_chunks, max_length=max_len, min_length=min_len, do_sample=False)

        # Combine the summaries from all chunks
        final_summary = ' '.join([summary['summary_text'] for summary in summaries])
        
        logging.info("Summary generated successfully.")
        
        return jsonify({'summary': final_summary})

    except Exception as e:
        logging.error(f"An error occurred during summarization: {e}")
        return jsonify({'error': 'Failed to generate summary.'}), 500

if __name__ == '__main__':
    # Runs the Flask development server
    app.run(debug=True) 