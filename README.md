
# SahayakGraph

SahayakGraph is an AI-powered research project designed to help users explore Indian government welfare schemes and understand potential eligibility based on their profile.

## Planned Components

- **RAG:** Retrieve relevant information from scheme documents.
- **Knowledge Graph:** Represent relationships between citizens, eligibility criteria, and schemes.
- **AI Agents:** Analyze retrieved information and coordinate eligibility checks.
- **Gemini:** Generate natural-language explanations using Google's Gemini API.
- **Evaluation:** Test the accuracy and reliability of results.

## Project Structure

- `data/` — Source documents and datasets
- `kg/` — Knowledge graph
- `rag/` — Retrieval pipeline
- `agents/` — AI agents
- `eval/` — Evaluation
- `app/` — User interface
- `paper/` — Research paper
- `tests/` — Automated tests
- `contract/` — Shared data contracts

## Setup

1. Install Python 3.11.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your Gemini API key:

   ```text
   GEMINI_API_KEY=your_api_key_here
   ```

**Security:** Never commit `.env` or expose your API key publicly.

## Project Status

Initial environment setup and Gemini API connectivity are complete.
