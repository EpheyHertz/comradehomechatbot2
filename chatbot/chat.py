import os
import requests
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv


from app.config import GEMINI_API_KEY, HUGGINGFACE_API_KEY

# Load API keys from environment variables
load_dotenv()
gemini_api_key = GEMINI_API_KEY
huggingface_api_key = HUGGINGFACE_API_KEY


# Load embeddings model
try:
    hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=huggingface_api_key,
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
except Exception as e:
    print(f"Error loading embeddings model: {e}")
    hf_embeddings = None

# Load ChromaDB vector database
persist_directory = "db"

try:
    # Check if the database already exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Loading existing ChromaDB database...")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=hf_embeddings)
    else:
        print("Creating a new ChromaDB database...")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=hf_embeddings)
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

except Exception as e:
    print(f"Error loading ChromaDB: {e}")
    vectordb = None
    retriever = None

# Define chatbot prompt
real_estate_prompt = PromptTemplate(
    template="""You are a highly knowledgeable and professional real estate expert. 
    Your task is to provide expert insights on real estate topics, including property buying, selling, rentals, and investments. 
    You are also a chatbot for the Angel Housing platform created by Ephey Nyaga, a Generative AI Engineer and a Computer Science student from EMBU University, Kenya. 
    You assist users with matters related to the website (angelhouslistingwebsite.vercel.app).
    
    Always respond in a professional and engaging manner. 

    Here is some relevant information that may help:
    {context}

    Question: {question}
    """,
    input_variables=["context", "question"]
)

# Define the LLM model (Gemini AI)
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        api_key=gemini_api_key,
        temperature=0.3
    )
except Exception as e:
    print(f"Error initializing Gemini AI: {e}")
    llm = None

# Initialize memory to keep track of conversations
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create a Conversational Retrieval Chain
if llm and retriever:
    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": real_estate_prompt}
        )
    except Exception as e:
        print(f"Error creating Conversational Retrieval Chain: {e}")
        qa_chain = None
else:
    qa_chain = None

# **Step 1: LLM Determines Whether to Fetch Listings or Use Knowledge Base**
action_classification_prompt = PromptTemplate(
    template="""
    You are an AI assistant helping users with real estate inquiries. Your task is to classify the user's query into one of two categories:

    **Categories:**
    1. **FETCH_LISTINGS**: The user is explicitly looking for property listings (e.g., houses, apartments, villas, rentals) and provides specific details such as location, price range, or property type.
    2. **USE_KNOWLEDGE**: The user is seeking general real estate advice, such as investment tips, mortgage guidance, legal matters, or market trends.

    **Rules for Classification:**
    - Classify as **FETCH_LISTINGS** ONLY if:
      - The query explicitly mentions **property types** (e.g., house, apartment, villa, rental) AND
      - Includes at least **one specific detail** such as location, price range, or amenities.
    - Classify as **USE_KNOWLEDGE** if:
      - The query is about general real estate topics (e.g., "Should I invest in real estate?", "How do mortgages work?") OR
      - The query lacks specific details about properties (e.g., "Tell me about the real estate market").
    - If the query is ambiguous or lacks sufficient information, ask the user for clarification instead of making assumptions.

    **Examples:**
    1. "Find me a 2-bedroom apartment in New York under $3000" → FETCH_LISTINGS
    2. "What are the best neighborhoods to invest in?" → USE_KNOWLEDGE
    3. "How do I apply for a mortgage?" → USE_KNOWLEDGE
    4. "Show me houses in Miami" → FETCH_LISTINGS
    5. "Is it a good time to buy a house?" → USE_KNOWLEDGE
    6. "Tell me about the real estate market in California" → USE_KNOWLEDGE

    **Response Format:**
    - Reply with ONLY "FETCH_LISTINGS" or "USE_KNOWLEDGE". Do not include any additional text or explanations.

    User query: "{query}"
    """,
    input_variables=["query"]
)


# **Step 2: Extract Relevant Filters for Listings**
extract_details_prompt = PromptTemplate(
    template="""Extract relevant details from the user's request.

    User query: "{query}"
    
    Extracted details:
    - Location: (if mentioned, otherwise return "None")
    - Price: (if mentioned, otherwise return "None")
    - Bedrooms: (if mentioned, otherwise return "None")
    - Type: (e.g., Apartment, Villa, etc., otherwise return "None")
    - Amenities: (if mentioned, otherwise return "None")
    remember the type should be in lowercase and without spaces and the price should be in numbers only without any currency symbol
    Provide only the extracted details in the format:
    Location: <location> eg.embu
    Price: <price>eg.500
    Type: <type>eg.apartment

    """,
    input_variables=["query"]
)

# **Step 3: Fetch Houses from API**
import requests

def fetch_houses_from_api(location=None, price=None, house_type=None, amenities=None, limit=10, internal_call=False):
    """
    Fetch houses from the API based on user-provided filters.
    Prevents calling itself when the request originates from within the API.
    """
    if internal_call:
        return {"error": "Internal API call detected. Fetching is disabled to prevent recursion."}

    base_url = "https://angelhouslistingbackendapis.onrender.com/houses/"
    params = {}

    if location and location.lower() != "none":
        params["location"] = location
    if price and price.lower() != "none":
        params["max_price"] = price
    if house_type and house_type.lower() != "none":
        params["type"] = house_type
    if amenities and amenities.lower() != "none":
        params["amenities"] = amenities

    params["limit"] = limit  # Default limit

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()  # Return the list of houses
    except requests.exceptions.RequestException as e:
        print(f"Error fetching houses from API: {e}")
        return {"error": "Failed to fetch houses"}


# **Step 4: Format API Response for Chatbot**
def format_housing_response(houses):
    response = "🏡 **Available Properties Based on Your Search** 🏡\n\n"
    
    unique_houses = []
    seen = set()
    
    for house in houses:
        house_key = (house["title"], house["location"], house["price"], house["type"])
        if house_key not in seen:
            seen.add(house_key)
            unique_houses.append(house)

    for house in unique_houses:
        response += f"🏠 **{house['title']}**\n"
        response += f"📍 *Location:* {house['location']}\n"
        response += f"💰 *Price:* ${house['price']}/month\n"
        response += f"🏠 *Type:* {house['type']}\n"
        response += f"✨ *Amenities:* {', '.join(house['amenities'])}\n"
        
        owner = house["owner"]
        if owner.get("is_verified"):
            response += "✔️ *Verified Listing*\n"
        
        response += f"\n🔗 **Listed by:** {owner['full_name']}\n"
        response += f"![Profile Image]({owner['profile_image']})\n\n"
    
    response += "Would you like more details or assistance in booking a visit? 😊"
    
    return response






# **Main Chatbot Function**
def real_estate_chatbot(user_input: str):
    """
    Handles user queries related to real estate, deciding dynamically whether to fetch house listings or provide general advice.
    """
    if not llm:
        return "Error: Language model is not initialized."

    try:
        # **Step 1: Ask the LLM if we need to fetch listings or use knowledge**
        action_decision = llm.invoke(action_classification_prompt.format(query=user_input))
        print(f"Action Decision: {action_decision}")

        if action_decision.content == "FETCH_LISTINGS":
            # Extract details only if house listings are required
            details = llm.invoke(extract_details_prompt.format(query=user_input))
            extracted_details = details.content
            print(f"Extracted Details: {extracted_details}")

            # Parse extracted details
            details = {
                "location": extracted_details.split("Location: ")[1].split("\n")[0],
                "price": extracted_details.split("Price: ")[1].split("\n")[0],
                "house_type": extracted_details.split("Type: ")[1].split("\n")[0],
            }

            # Fetch and return listings
            house_list = fetch_houses_from_api(**details)
            print("House Lists Fetched",house_list)
            return format_housing_response(house_list['houses'])

        else:
            # Use knowledge base for answering general real estate questions
            if qa_chain:
                response = qa_chain.invoke({"question": user_input})
                return response["answer"]
            else:
                return "Error: Conversational Retrieval Chain is not initialized."
    except Exception as e:
        return f"An error occurred while processing your request: {e}"

