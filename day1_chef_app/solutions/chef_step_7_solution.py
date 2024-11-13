import logging
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage
from typing import List, Optional
import textstat

# Set up logging configuration to save interactions in llm_logs.log with timestamps
logging.basicConfig(filename="llm_logs.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Define API key and model version (replace 'your_actual_api_key' with your API key)
API_KEY = "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"  
MODEL_VERSION = "meta/llama-3.2-3b-instruct"

# Connect to NVIDIA API using ChatNVIDIA with specific parameters
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

# Define function to calculate ingredient coverage score
def ingredient_coverage_score(recipe: str, ingredients: list) -> float:
    """
    Calculates the proportion of specified ingredients found in the recipe.
    
    Args:
        recipe (str): The generated recipe text.
        ingredients (list): List of ingredients provided in the prompt.

    Returns:
        float: Proportion of specified ingredients found in the recipe text,
               ranging from 0 to 1, where 1 means all ingredients are present.
    """
    recipe_lower = recipe.lower()
    ingredient_matches = sum(1 for ingredient in ingredients if ingredient.lower() in recipe_lower)
    return ingredient_matches / len(ingredients) if ingredients else 0

# Define function to calculate lexical diversity score
def lexical_diversity_score(recipe: str) -> float:
    """
    Calculates the lexical diversity of the recipe text.
    
    Args:
        recipe (str): The generated recipe text.

    Returns:
        float: Lexical diversity score, which is the ratio of unique words
               to total words. A higher score indicates greater word variety.
    """
    words = recipe.split()
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0

# Define function to calculate readability score
def readability_score(recipe: str) -> float:
    """
    Calculates the readability score of the recipe using Flesch Reading Ease.
    
    Args:
        recipe (str): The generated recipe text.

    Returns:
        float: Readability score; higher values indicate easier-to-read text.
    """
    return textstat.flesch_reading_ease(recipe)

# Define function to log LLM interactions, including model parameters and evaluation metrics
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
    logging.info("Evaluation Scores: Coverage=%.2f, Diversity=%.2f, Readability=%.2f", coverage_score, diversity_score, readability)
    logging.info("Response: %s", response)

# Main recipe generation function
def generate_recipe(client, ingredients: List[str], dietary_restrictions: Optional[List[str]] = None, 
                    course_type: str = "main dish", preference: str = "easy") -> str:
    """
    Generates a recipe from the NVIDIA LLM model based on a structured prompt.

    Args:
        client (ChatNVIDIA): The LLM client instance.
        ingredients (List[str]): List of ingredients for the recipe.
        dietary_restrictions (Optional[List[str]]): Any dietary restrictions.
        course_type (str): Type of course (e.g., main dish).
        preference (str): User's recipe preference (e.g., easy).

    Returns:
        str: Generated recipe text along with evaluation scores.
    """
    # Build prompt
    prompt_content = f"Create a {preference} recipe for a {course_type} using the following ingredients: {', '.join(ingredients)}. " \
                     f"Ensure the recipe is {', '.join(dietary_restrictions) if dietary_restrictions else 'no specific dietary restrictions'}."
    prompt = HumanMessage(content=prompt_content)
    
    # Send prompt to the model and get response
    response = client.invoke([prompt])
    recipe_text = response.content

    # Calculate evaluation metrics
    coverage_score = ingredient_coverage_score(recipe_text, ingredients)
    diversity_score = lexical_diversity_score(recipe_text)
    readability = readability_score(recipe_text)

    # Log interaction and evaluation metrics
    log_llm_interaction(prompt_content, recipe_text, client.temperature, client.top_p, coverage_score, diversity_score, readability)
    
    return recipe_text, coverage_score, diversity_score, readability

# Streamlit UI setup
st.title("AI Chef: Recipe Generator")


# Input fields for recipe customization
ingredients = st.text_input("Enter ingredients (comma-separated):").split(", ")  # Text input for ingredients list
dietary_restrictions = st.multiselect("Select dietary restrictions", ["gluten-free", "vegan", "vegetarian", "dairy-free"])  # Multi-select for dietary preferences
course_type = st.selectbox("Select course type", ["Main dish", "Dessert", "Appetizer"])  # Dropdown for course type
preference = st.selectbox("Select preference", ["easy", "gourmet", "quick"])  # Dropdown for recipe preference
temperature = st.slider("Select creativity level (temperature)", 0.0, 1.0, 0.5)  # Slider to adjust temperature
top_p = st.slider("Select top_p sampling", 0.0, 1.0, 0.7)  # Slider for top_p parameter

# Button to generate recipe
if st.button("Generate Recipe"):  # Button to trigger recipe generation
    # Connect to NVIDIA client with specified temperature and top_p
    client = connect_to_nvidia(temperature=temperature, top_p=top_p)
    
    # Generate recipe based on user inputs and log the interaction
    recipe_text, coverage_score, diversity_score, readability = generate_recipe(
        client, ingredients, dietary_restrictions, course_type, preference
    )

    # Display generated recipe and evaluation scores
    st.write("### Recipe:")  # Display section title for recipe
    st.write(recipe_text)  # Display generated recipe text
    st.write("### Evaluation Scores:")  # Display section title for evaluation scores
    st.write(f"Ingredient Coverage Score: {coverage_score:.2f} (0 to 1 scale)")  # Display ingredient coverage score
    st.write(f"Lexical Diversity Score: {diversity_score:.2f} (0 to 1 scale)")  # Display lexical diversity score
    st.write(f"Readability Score (Flesch Reading Ease): {readability:.2f} (0 to 100 scale)")  # Display readability score
