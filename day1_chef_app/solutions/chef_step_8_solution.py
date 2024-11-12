import logging
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Optional
import textstat

# Set up logging configuration to save interactions in llm_logs.log with timestamps
logging.basicConfig(filename="llm_logs.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Define API key and model version (replace 'your_actual_api_key' with your API key)
API_KEY = "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"
MODEL_VERSION = "meta/llama-3.2-3b-instruct"

# Initialize conversation memory and recipe history in Streamlit session
if 'recipe_history' not in st.session_state:
    st.session_state.recipe_history = []  # List to store user and AI messages for chat history

# Connect to NVIDIA API using ChatNVIDIA with specified parameters
def connect_to_nvidia(api_key: str = API_KEY, model: str = MODEL_VERSION, temperature: float = 0.5, top_p: float = 0.7) -> ChatNVIDIA:
    """
    Establishes a connection to the NVIDIA LLM API.

    Args:
        api_key (str): The API key for NVIDIA access.
        model (str): The model version to use.
        temperature (float): Sampling temperature for creativity.
        top_p (float): Nucleus sampling parameter.

    Returns:
        ChatNVIDIA: An instance of ChatNVIDIA connected to the specified model.
    """
    return ChatNVIDIA(
        model=model,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=1024
    )

# Function to log LLM interactions, including model parameters and evaluation metrics
def log_llm_interaction(prompt: str, response: str, temperature: float, top_p: float, coverage_score: float, diversity_score: float, readability: float):
    """
    Logs each interaction with the LLM, including the prompt, response, model parameters, and evaluation metrics.

    Args:
        prompt (str): The prompt sent to the LLM.
        response (str): The LLM's response.
        temperature (float): Model's temperature parameter used for this interaction.
        top_p (float): Model's top_p parameter used for this interaction.
        coverage_score (float): Ingredient coverage score.
        diversity_score (float): Lexical diversity score.
        readability (float): Readability score of the response.
    """
    logging.info("Prompt: %s", prompt)
    logging.info("Model Parameters: Temperature=%.2f, Top_p=%.2f", temperature, top_p)
    logging.info("Response: %s", response)
    logging.info("Coverage: %.2f, Diversity: %.2f, Readability: %.2f", coverage_score, diversity_score, readability)

# Main recipe generation function with conversation history
def generate_recipe(client, ingredients: List[str], dietary_restrictions: Optional[List[str]] = None, 
                    course_type: str = "main dish", preference: str = "easy", additional_request: str = "") -> str:
    """
    Generates a recipe from the NVIDIA LLM model based on input and conversation history.

    Args:
        client (ChatNVIDIA): The LLM client instance.
        ingredients (List[str]): List of ingredients for the recipe.
        dietary_restrictions (Optional[List[str]]): Any dietary restrictions.
        course_type (str): Type of course (e.g., main dish).
        preference (str): User's recipe preference (e.g., easy).
        additional_request (str): Additional user input or request.

    Returns:
        str: Generated recipe text along with evaluation scores.
    """
    # Prepare conversation history
    conversation_history = [
        HumanMessage(content=msg['content']) if msg['role'] == "user" else SystemMessage(content=msg['content'])
        for msg in st.session_state.recipe_history
    ]

    # Build prompt with conversation history
    prompt_content = f"Create a {preference} recipe for a {course_type} using these ingredients: {', '.join(ingredients)}. "
    prompt_content += f"Ensure the recipe is {', '.join(dietary_restrictions) if dietary_restrictions else 'no specific dietary restrictions'}. {additional_request}"
    conversation_history.append(HumanMessage(content=prompt_content))

    # Set up ChatPromptTemplate with conversation history
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI Chef assistant creating personalized recipes."),
        MessagesPlaceholder(variable_name="messages")
    ])
    chain = prompt_template | client
    ai_response = chain.invoke({"messages": conversation_history}).content  # Get response content

    # Save AI response to recipe history
    st.session_state.recipe_history.append({"role": "assistant", "content": ai_response})

    # Evaluate and log the recipe
    coverage_score = ingredient_coverage_score(ai_response, ingredients)
    diversity_score = lexical_diversity_score(ai_response)
    readability = readability_score(ai_response)
    log_llm_interaction(prompt_content, ai_response, client.temperature, client.top_p, coverage_score, diversity_score, readability)

    return ai_response, coverage_score, diversity_score, readability

# Define evaluation functions
def ingredient_coverage_score(recipe: str, ingredients: List[str]) -> float:
    """
    Calculates the ingredient coverage score based on the number of ingredients used.

    Args:
        recipe (str): The generated recipe text.
        ingredients (List[str]): List of ingredients provided in the prompt.

    Returns:
        float: Proportion of specified ingredients found in the recipe text.
    """
    recipe_lower = recipe.lower()
    matches = sum(1 for ingredient in ingredients if ingredient.lower() in recipe_lower)
    return matches / len(ingredients) if ingredients else 0

def lexical_diversity_score(recipe: str) -> float:
    """
    Calculates the lexical diversity of the recipe text.

    Args:
        recipe (str): The generated recipe text.

    Returns:
        float: Lexical diversity score, indicating word variety.
    """
    words = recipe.split()
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0

def readability_score(recipe: str) -> float:
    """
    Calculates the readability score of the recipe using Flesch Reading Ease.

    Args:
        recipe (str): The generated recipe text.

    Returns:
        float: Readability score (higher values indicate easier readability).
    """
    return textstat.flesch_reading_ease(recipe)

# Streamlit UI setup
st.title("AI Chef: Your Personalized Recipe Generator")

# Input fields for recipe customization
ingredients = st.text_input("Enter ingredients (comma-separated):").split(", ")
dietary_restrictions = st.multiselect("Select dietary restrictions", ["gluten-free", "vegan", "vegetarian", "dairy-free"])
course_type = st.selectbox("Select course type", ["Main dish", "Dessert", "Appetizer"])
preference = st.selectbox("Select preference", ["easy", "gourmet", "quick"])
temperature = st.slider("Select creativity level (temperature)", 0.0, 1.0, 0.5)
top_p = st.slider("Select top_p sampling", 0.0, 1.0, 0.7)

# Initialize ChatNVIDIA client with configured temperature and top_p
if "client" not in st.session_state:
    st.session_state.client = connect_to_nvidia(temperature=temperature, top_p=top_p)

# Display chat history
for message in st.session_state.recipe_history:
    if message["role"] == "user":
        st.chat_message("user").markdown(f"**You:** {message['content']}")
    else:
        st.chat_message("assistant").markdown(f"**AI Chef:** {message['content']}")

# User input for additional requests
user_input = st.chat_input("Enter additional requests or adjustments:")
if user_input:
    st.session_state.recipe_history.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(f"**You:** {user_input}")

    # Generate recipe with conversation context
    with st.spinner():
        recipe_text, coverage_score, diversity_score, readability = generate_recipe(
            st.session_state.client, ingredients, dietary_restrictions, course_type, preference, user_input
        )
        st.chat_message("assistant").markdown(f"**AI Chef:** {recipe_text}")
