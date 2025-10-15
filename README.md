# Smart Resume Screener

An AI-powered resume ranking application built with Streamlit, leveraging SBERT (Sentence-BERT) for semantic matching to compare resumes against job descriptions. Optionally integrates OpenAI's GPT for advanced scoring and justifications.
<img width="1920" height="1080" alt="Demo SS2" src="https://github.com/user-attachments/assets/bd1fe049-e189-4cf0-835a-967385866a9a" />
<img width="1920" height="1080" alt="Demo SS1" src="https://github.com/user-attachments/assets/5019f859-a79b-46c3-a5e6-83eeeb3c0f37" />

## Features

- **Semantic Matching**: Uses SBERT to compute similarity scores between job descriptions and resumes.
- **LLM Integration**: Optional use of OpenAI GPT for detailed scoring (1-10 scale) with justifications.
- **Resume Parsing**: Extracts skills, experience, education, and email from uploaded resumes (PDF/TXT).
- **Database Storage**: Stores parsed resumes in an in-memory SQLite database for persistence.
- **Ranking and Download**: Ranks candidates and allows downloading results as CSV.
- **Multi-page UI**: Includes ranker and database viewer pages.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Abhinayr006/Smart-Resume-Screener.git
   cd Smart-Resume-Screener
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .myenv
   .myenv\Scripts\activate  # On Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables** (Optional, for LLM features):
   - Create a `.env` file in the root directory.
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Ranker Page**:
   - Enter or upload a job description.
   - Upload resumes (PDF or TXT files).
   - Toggle LLM usage if API key is configured.
   - Click "Rank Resumes" to get similarity scores and rankings.
   - Download selected results as CSV.

2. **Database Viewer Page**:
   - View all stored parsed resumes.
   - Download the full database as CSV.

## Dependencies

- streamlit
- pandas
- numpy
- sentence-transformers
- scikit-learn
- PyPDF2
- pdfplumber
- nltk
- openai
- python-dotenv

## Project Structure

- `app.py`: Main Streamlit application.
- `utils.py`: Utility functions for model loading, ranking, parsing, and database operations.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Files to ignore in version control.
- `test/`: Sample resumes for testing.
- `Demo Video/`: Demo video file.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
