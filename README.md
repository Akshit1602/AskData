# Shell GenAI Demo

This Streamlit application allows you to ask questions about your Shell retail data in natural language. The application uses a Large Language Model to convert your questions into SQL queries, executes them against a local SQLite database, and then generates business insights based on the results.

## Setup

1.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up your OpenAI API credentials:**
    This application uses Streamlit's secrets management for API keys. Create a file named `.streamlit/secrets.toml` and add your OpenAI credentials in the following format:
    ```toml
    OPENAI_API_TYPE = "azure"
    OPENAI_API_KEY = "your_openai_api_key"
    OPENAI_API_BASE = "your_openai_api_base"
    OPENAI_API_VERSION = "2024-05-01-preview"
    OPENAI_DEPLOYMENT_NAME = "gpt-4o"
    ```

## How to Run the Application

Once you have installed the dependencies and set up your API keys, you can run the Streamlit app with the following command:

```bash
streamlit run app.py
```

The application will open in your web browser. You can then ask questions in the chat interface to get insights from your data.
