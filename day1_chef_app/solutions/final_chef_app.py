import streamlit as st
import logging
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from typing import List, Optional
import textstat

# Set up logging for LLMOps
logging.basicConfig(filename="llm_logs.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Model and dataset versions
MODEL_VERSION = "meta/llama-3.2-3b-instruct"
DATASET_VERSION = "v1.0"

# Define API key (replace with your actual key)
API_KEY = "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"

# Initialize conversation memory and recipe history in Streamlit session
if 'memory' not in st.session_state:
    st.session_state.memory = []  # Using a list to track messages
if 'recipe_history' not in st.session_state:
    st.session_state.recipe_history = []  # Store both user and AI messages

# Connect to NVIDIA API using ChatNVIDIA with specific parameters
def connect_to_nvidia(model: str = MODEL_VERSION, api_key: str = API_KEY, temperature: float = 0.5, top_p: float = 0.7) -> ChatNVIDIA:
    """
    Establishes a connection to the NVIDIA LLM API with dynamic temperature and top_p values.

    Args:
        model (str): The model version to use.
        api_key (str): The API key for NVIDIA access.
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

def log_llm_interaction(temperature: float, top_p: float, response: str, prompt: str, coverage_score: float, diversity_score: float, readability: float):
    """
    Logs each interaction with the LLM, including the prompt, response, model version,
    and evaluation metrics (coverage, diversity, readability).

    Args:
        prompt (str): The prompt sent to the LLM.
        response (str): The LLM's response.
        coverage_score (float): Ingredient coverage score.
        diversity_score (float): Lexical diversity score.
        readability (float): Readability score of the response.
    """
    logging.info(f"Model Version: {MODEL_VERSION}, Dataset Version: {DATASET_VERSION}")
    logging.info(f"Model Temperature: {temperature}, Model Top_p: {top_p}")
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Response: {response}")
    logging.info(f"Evaluation Scores: Coverage={coverage_score:.2f}, Diversity={diversity_score:.2f}, Readability={readability:.2f}")

def generate_recipe(client, temperature: float, top_p: float, ingredients: List[str], dietary_restrictions: Optional[List[str]] = None, 
                    course_type: str = "main dish", preference: str = "easy", additional_request: str = ""
                    ) -> str:
    """
    Generates a recipe from the NVIDIA LLM model based on a structured prompt.

    Args:
        client (ChatNVIDIA): The LLM client instance.
        ingredients (List[str]): List of ingredients for the recipe.
        dietary_restrictions (Optional[List[str]]): Any dietary restrictions.
        course_type (str): Type of course (e.g., main dish).
        preference (str): User's recipe preference (e.g., easy).
        additional_request (str): Additional user input or request.

    Returns:
        str: Generated recipe or error message if generation fails.
    """
    # Construct initial message list with full conversation history
    conversation_history = [
        HumanMessage(content=msg['content']) if msg['role'] == "user" else SystemMessage(content=msg['content'])
        for msg in st.session_state.recipe_history
    ]

    # Add the new user request to the conversation history
    conversation_history.append(HumanMessage(content=f"Create a {preference} recipe for a {course_type} using the following ingredients: "
                                                     f"{', '.join(ingredients)}. Ensure the recipe is "
                                                     f"{', '.join(dietary_restrictions) if dietary_restrictions else 'no specific dietary restrictions'}. "
                                                     f"{additional_request}"))

    # Define the chat prompt template with a placeholder for dynamic messages
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a helpful AI Chef assistant creating personalized recipes."),
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    # Define the chain with the prompt template and client
    chain = prompt_template | client

    # Invoke the chain with the constructed messages using the full conversation history
    ai_response = chain.invoke({"messages": conversation_history})
    response = ai_response.content  # Get the content from the response message

    # Append AI response to recipe history
    st.session_state.recipe_history.append({"role": "assistant", "content": response})

    # Evaluate the recipe and log interaction with scores
    coverage_score, diversity_score, readability = evaluate_recipe(response, ingredients)
    log_llm_interaction(temperature, top_p, conversation_history[-1].content, response, coverage_score, diversity_score, readability)

    return response


def ingredient_coverage_score(recipe: str, ingredients: List[str]) -> float:
    """
    Calculates the ingredient coverage score based on the number of ingredients used.

    Args:
        recipe (str): The generated recipe.
        ingredients (List[str]): List of ingredients to check in the recipe.

    Returns:
        float: The proportion of ingredients covered in the recipe.
    """
    recipe_lower = recipe.lower()
    ingredient_matches = sum(1 for ingredient in ingredients if ingredient.lower() in recipe_lower)
    return ingredient_matches / len(ingredients) if ingredients else 0

def lexical_diversity_score(recipe: str) -> float:
    """
    Calculates the lexical diversity of the recipe text.

    Args:
        recipe (str): The generated recipe.

    Returns:
        float: The lexical diversity score.
    """
    words = recipe.split()
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0

def readability_score(recipe: str) -> float:
    """
    Calculates the readability score of the recipe using Flesch Reading Ease.

    Args:
        recipe (str): The generated recipe.

    Returns:
        float: Readability score (higher means easier to read).
    """
    return textstat.flesch_reading_ease(recipe)

def evaluate_recipe(recipe: str, ingredients: List[str]):
    """
    Evaluates the generated recipe based on ingredient coverage, lexical diversity, and readability.

    Args:
        recipe (str): The generated recipe.
        ingredients (List[str]): List of ingredients used in evaluation.

    Returns:
        tuple: Coverage score, diversity score, and readability score.
    """
    coverage_score = ingredient_coverage_score(recipe, ingredients)
    diversity_score = lexical_diversity_score(recipe)
    readability = readability_score(recipe)
    return coverage_score, diversity_score, readability

# Streamlit UI setup
st.title("AI Chef: Your Personalized Recipe Generator")

# User input fields
ingredients = st.text_input("Enter ingredients (comma-separated):").split(", ")
dietary_restrictions = st.multiselect("Select dietary restrictions", ["gluten-free", "vegan", "vegetarian", "dairy-free"])
course_type = st.selectbox("Select course type", ["Main dish", "Dessert", "Appetizer"])
preference = st.selectbox("Select preference", ["easy", "gourmet", "quick"])
# Streamlit UI setup for temperature and top_p
temperature = st.slider("Select creativity level (temperature)", 0.0, 1.0, 0.5)
top_p = st.slider("Select top_p sampling", 0.0, 1.0, 0.7)

# Initialize ChatNVIDIA client with UI-configured temperature and top_p
if "client" not in st.session_state:
    st.session_state.client = connect_to_nvidia(temperature=temperature, top_p=top_p)

# Show recipe history in chat format
for message in st.session_state.recipe_history:
    if message["role"] == "user":
        st.chat_message("user").markdown(f"**You:** {message['content']}")
    else:
        st.chat_message("assistant").markdown(f"**AI Chef:** {message['content']}")

# Generate recipe upon user input in chat input box
user_input = st.chat_input("Enter any additional requests or adjustments:")
if user_input:
    # Save user input to session history
    st.session_state.recipe_history.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(f"**You:** {user_input}")

    # Generate and display recipe based on user input
    if "client" in st.session_state:
        with st.spinner():
            response = generate_recipe(
                st.session_state.client,
                temperature, 
                top_p,
                ingredients,
                dietary_restrictions,
                course_type,
                preference,
                user_input
            )
            st.chat_message("assistant").markdown(f"**AI Chef:** {response}")
