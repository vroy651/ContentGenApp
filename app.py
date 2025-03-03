import os
import sys

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_path)

from src.chat_history import ChatHistory

# Initialize chat history at the start of the application
chat_history = ChatHistory()

# Import the Streamlit UI
from chatbot import main

# Run the application
if __name__ == "__main__":
    main()
