import time
import os
import uuid
import logging
import json
import requests
from typing import Annotated, TypedDict, Optional, Union, Tuple
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, message_to_dict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("real_estate_chatbot")

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Rate limiting
GEMINI_RETRY_DELAY = 2.0  # seconds between retries
GEMINI_MAX_RETRIES = 3

# Define state with annotated messages
class RealEstateState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    classification: Optional[str]
    details: Optional[dict]

# Helper function to serialize messages for logging
def serialize_messages(messages):
    return [
        {
            "type": msg.__class__.__name__,
            "content": msg.content,
            "timestamp": getattr(msg, "timestamp", None)
        }
        for msg in messages
    ]

# Helper function to format conversation history
def format_conversation_history(messages, max_messages=5):
    # Get the last few messages or all if less than max_messages
    history_messages = messages[-min(max_messages, len(messages)):]
    history_context = "\n".join([
        f"{'Human:' if isinstance(msg, HumanMessage) else 'AI:'} {msg.content}" 
        for msg in history_messages
    ])
    return history_context

# ==== Initialize Components ====
# Embeddings
hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACE_API_KEY,
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)

# Vector Database
vectordb = Chroma(
    persist_directory="db",
    embedding_function=hf_embeddings
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    api_key=GEMINI_API_KEY,
    temperature=0.3
)

# ==== Prompt Templates ====
real_estate_prompt = """You are a professional real estate expert for Comrade Homes. 
Provide expert insights on real estate topics. Always be professional and helpful.Crated by Ephey Nyaga. He is a Computer Science student at the University of Embu.

Relevant Context:
{context}

Previous Conversation:
{history}

Question: {question}
"""

action_classification_prompt = """Classify the query into FETCH_LISTINGS, USE_KNOWLEDGE, or REFER_TO_HISTORY:

Previous conversation:
{history}

Current query: {query}

If the query is asking about something mentioned in the previous conversation or requires clarification about previous messages, classify as REFER_TO_HISTORY.
If the query is asking for property listings or searching for properties, classify as FETCH_LISTINGS.
Otherwise, classify as USE_KNOWLEDGE.

Respond ONLY with either FETCH_LISTINGS, USE_KNOWLEDGE, or REFER_TO_HISTORY:"""

extract_details_prompt = """Extract these details from the query:
- Location (or None)
- Price (number only or None)
- Property type (lowercase or None)

Query: {query}
Respond in format:
Location: <value>
Price: <value>
Type: <value>"""

history_based_response_prompt = """
You are a professional real estate expert for Comrade Homes.

The user has asked: {question}

Based ONLY on the conversation history below, provide a response:
{history}

Your response should reference specific information from the conversation history while maintaining a professional and helpful tone.
If the conversation history does not contain information needed to answer the question, acknowledge this but still attempt to provide the most relevant response based on the available context.
"""

# ==== Graph Nodes ====
def classify_query(state: RealEstateState):
    last_message = state["messages"][-1]
    # logger.info(f"Classifying query: {last_message.content}")
    
    # Build history context from previous messages
    history_context = ""
    if len(state["messages"]) > 1:
        history_messages = state["messages"][:-1]  # All but the last message
        history_context = format_conversation_history(history_messages, max_messages=4)
    
    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            response = llm.invoke(action_classification_prompt.format(
                history=history_context,
                query=last_message.content
            ))
            classification = response.content.strip()
            # logger.info(f"Query classified as: {classification}")
            return {**state, "classification": classification}
        except Exception as e:
            if "429" in str(e) and attempt < GEMINI_MAX_RETRIES - 1:
                time.sleep(GEMINI_RETRY_DELAY)
                continue
            # logger.error(f"Classification error: {str(e)}")
            raise

def extract_details(state: RealEstateState):
    last_message = state["messages"][-1]
    # logger.info(f"Extracting details from: {last_message.content}")
    
    response = llm.invoke(extract_details_prompt.format(query=last_message.content))
    
    # Parse response
    details = {"location": "None", "price": "None", "type": "None"}
    for line in response.content.split("\n"):
        if "Location:" in line:
            details["location"] = line.split(": ")[1].strip()
        elif "Price:" in line:
            details["price"] = line.split(": ")[1].strip()
        elif "Type:" in line:
            details["type"] = line.split(": ")[1].strip().lower()
    
    # logger.info(f"Extracted details: {json.dumps(details)}")
    return {**state, "details": details}

def fetch_listings(state: RealEstateState):
    details = state["details"]
    # logger.info(f"Fetching listings with details: {json.dumps(details)}")
    
    houses = fetch_houses_from_api(
        location=details["location"],
        price=details["price"],
        house_type=details["type"]
    )
    
    response = format_housing_response(houses["houses"]) if "houses" in houses else "No listings found."
    # logger.info(f"Generated listings response: {response[:100]}...")  # Log first 100 chars
    
    new_state = {**state, "messages": state["messages"] + [AIMessage(content=response)]}
    # logger.info(f"Updated message history, now has {len(new_state['messages'])} messages")
    return new_state

def answer_from_history(state: RealEstateState):
    last_message = state["messages"][-1]
    # logger.info(f"Answering from conversation history: {last_message.content}")
    
    # Get conversation history excluding the current message
    history_context = format_conversation_history(state["messages"][:-1], max_messages=6)
    
    prompt = history_based_response_prompt.format(
        question=last_message.content,
        history=history_context
    )
    
    # logger.info("Using history-based response prompt")
    
    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            response = llm.invoke(prompt)
            logger.info(f"Generated history-based response: {response.content[:100]}...")
            
            new_state = {**state, "messages": state["messages"] + [AIMessage(content=response.content)]}
            return new_state
        except Exception as e:
            if "429" in str(e) and attempt < GEMINI_MAX_RETRIES - 1:
                time.sleep(GEMINI_RETRY_DELAY)
                continue
            logger.error(f"History response error: {str(e)}")
            raise

def answer_knowledge(state: RealEstateState):
    last_message = state["messages"][-1]
    # logger.info(f"Answering knowledge query: {last_message.content}")
    
    # FIRST: Check if the answer is in conversation history
    if len(state["messages"]) > 2:  # If we have previous exchanges
        history_context = format_conversation_history(state["messages"][-6:-1])  # Last 5 messages excluding current
        
        # Create a prompt that first checks conversation history
        history_check_prompt = f"""
        The user has asked: {last_message.content}
        
        Recent conversation history:
        {history_context}
        
        Does the conversation history contain information relevant to answering this question?
        Answer only YES or NO.
        """
        
        try:
            history_check = llm.invoke(history_check_prompt)
            if "yes" in history_check.content.lower():
                # logger.info("History contains relevant information, using history-based response")
                return answer_from_history(state)
        except Exception as e:
            logger.error(f"History check error: {str(e)}")
            # Continue with RAG if history check fails
    
    # THEN: If not in history or check failed, proceed with RAG
    docs = retriever.invoke(last_message.content)
    context = "\n".join([d.page_content for d in docs])
    # logger.info(f"Retrieved {len(docs)} relevant documents")
    
    # Include conversation history in the prompt
    history_context = ""
    if len(state["messages"]) > 1:
        history_context = format_conversation_history(state["messages"][:-1], max_messages=3)
    
    response = llm.invoke(real_estate_prompt.format(
        context=context, 
        history=history_context,
        question=last_message.content
    ))
    # logger.info(f"Generated knowledge response: {response.content[:100]}...")  # Log first 100 chars
    
    new_state = {**state, "messages": state["messages"] + [AIMessage(content=response.content)]}
    # logger.info(f"Updated message history, now has {len(new_state['messages'])} messages")
    return new_state

# ==== Graph Construction ====
def create_real_estate_graph():
    # Create a persistent memory saver
    memory = MemorySaver()
    builder = StateGraph(RealEstateState)
    
    builder.add_node("classify", classify_query)
    builder.add_node("extract_details", extract_details)
    builder.add_node("fetch_listings", fetch_listings)
    builder.add_node("answer_knowledge", answer_knowledge)
    builder.add_node("answer_from_history", answer_from_history)

    builder.add_edge(START, "classify")
    
    builder.add_conditional_edges(
        "classify",
        lambda state: {
            "FETCH_LISTINGS": "extract_details",
            "USE_KNOWLEDGE": "answer_knowledge",
            "REFER_TO_HISTORY": "answer_from_history"
        }.get(state.get("classification", "USE_KNOWLEDGE"), "answer_knowledge")
    )
    
    builder.add_edge("extract_details", "fetch_listings")
    builder.add_edge("fetch_listings", END)
    builder.add_edge("answer_knowledge", END)
    builder.add_edge("answer_from_history", END)

    # logger.info("Real estate graph created and compiled with memory saver")
    return builder.compile(checkpointer=memory)

# ==== API Functions ====
def fetch_houses_from_api(location=None, price=None, house_type=None):
    params = {
        "location": location if location != "None" else None,
        "max_price": price if price != "None" else None,
        "type": house_type if house_type != "None" else None,
        "limit": 5
    }
    params = {k: v for k, v in params.items() if v}
    
    try:
        # logger.info(f"Fetching houses with params: {json.dumps(params)}")
        response = requests.get("https://digitaloceanapis.comradehomes.me/houses/", params=params)
        results = response.json()
        # logger.info(f"API returned {len(results.get('houses', []))} houses")
        return results
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return {"error": str(e)}

def format_housing_response(houses):
    if not houses:
        # logger.info("No houses found to format")
        return "No matching properties found."
    
    response = ["🏡 Available Properties:"]
    for house in houses[:3]:  # Show top 3
        response.append(
            f"\n🏠 {house['title']}\n"
            f"📍 {house['location']}\n"
            f"💰 ${house['price']}/month\n"
            f"🛏️ {house.get('bedrooms', 'N/A')} bedrooms\n"
            f"🔑 {house['type'].title()}\n"
            f"⭐ Amenities: {', '.join(house['amenities'][:3])}"
        )
    return "\n".join(response) + "\n\nNeed more details or want to schedule a visit?"

# Create a single instance of the graph
real_estate_graph = create_real_estate_graph()

def inspect_state(thread_id):
    """Inspect and log the current state for a thread"""
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state_tuple = real_estate_graph.get_state(config)
        if state_tuple and len(state_tuple) > 0:
            current_state = state_tuple[0]
            
            # Log the message history
            message_history = serialize_messages(current_state["messages"])
            # logger.info(f"Thread {thread_id} state inspection:")
            # logger.info(f"Message count: {len(message_history)}")
            # logger.info(f"Classification: {current_state.get('classification')}")
            # logger.info(f"Details: {json.dumps(current_state.get('details', {}))}")
            # logger.info(f"Message history: {json.dumps(message_history)}")
            
            return message_history
        else:
            # logger.warning(f"No state found for thread {thread_id}")
            return []
    except Exception as e:
        # logger.error(f"Error inspecting state for thread {thread_id}: {str(e)}")
        return []

def real_estate_chatbot(user_input: str, thread_id: str = None):
    # Generate thread ID if not provided
    thread_id = thread_id or str(uuid.uuid4())
    # logger.info(f"Starting new conversation with thread_id: {thread_id}")
    # logger.info(f"User input: {user_input}")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Create input with proper message structure
        input_state = {"messages": [HumanMessage(content=user_input)]}
        # logger.info("Initial state created with user message")
        
        # Process through the graph
        final_state = None
        for event in real_estate_graph.stream(input_state, config, stream_mode="values"):
            final_state = event
            # logger.info(f"Graph processing event received, state has {len(event.get('messages', []))} messages")
        
        # Log final state
        if final_state and "messages" in final_state:
            message_history = serialize_messages(final_state["messages"])
            # logger.info(f"Final state has {len(message_history)} messages")
            
            # Return the response and thread_id with full message history
            return {
                "thread_id": thread_id, 
                "response": final_state["messages"][-1].content,
                "message_history": message_history
            }
        
        # logger.warning("No final state or messages found")
        return {
            "thread_id": thread_id, 
            "response": "No response generated",
            "message_history": []
        }
    
    except Exception as e:
        # logger.error(f"Error in chatbot: {str(e)}")
        return {
            "thread_id": thread_id, 
            "response": f"Error: {str(e)}",
            "message_history": []
        }

def continue_conversation(user_input: str, thread_id: str):
    # logger.info(f"Continuing conversation with thread_id: {thread_id}")
    # logger.info(f"User input: {user_input}")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Inspect current state before modification
        # logger.info("Inspecting current state before adding new message")
        current_messages = inspect_state(thread_id)
        
        # Get existing state
        state_tuple = real_estate_graph.get_state(config)
        
        # If no existing state, start a new conversation
        if not state_tuple or len(state_tuple) == 0:
            # logger.warning(f"No existing state found for thread {thread_id}, starting new conversation")
            return real_estate_chatbot(user_input, thread_id)
            
        # Extract state from tuple and add new message
        existing_state = state_tuple[0]
        # logger.info(f"Retrieved existing state with {len(existing_state.get('messages', []))} messages")
        
        # Add new message to existing state
        input_state = {
            "messages": existing_state["messages"] + [HumanMessage(content=user_input)],
            "classification": None,  # Reset classification for new input
            "details": existing_state.get("details")
        }
        # logger.info(f"Created input state with {len(input_state['messages'])} messages")
        
        # Process through the graph
        final_state = None
        for event in real_estate_graph.stream(input_state, config, stream_mode="values"):
            final_state = event
            # logger.info(f"Graph processing event received, state has {len(event.get('messages', []))} messages")
        
        # Log final state
        if final_state and "messages" in final_state:
            message_history = serialize_messages(final_state["messages"])
            # logger.info(f"Final state has {len(message_history)} messages")
            
            # Return the response and thread_id with full message history
            return {
                "thread_id": thread_id, 
                "response": final_state["messages"][-1].content,
                "message_history": message_history
            }
        
        # logger.warning("No final state or messages found")
        return {
            "thread_id": thread_id, 
            "response": "No follow-up response generated",
            "message_history": current_messages  # Return the original messages in case of failure
        }
    
    except Exception as e:
        # logger.error(f"Error in continuing conversation: {str(e)}")
        return {
            "thread_id": thread_id, 
            "response": f"Error: {str(e)}",
            "message_history": []
        }

# Example usage
# if __name__ == "__main__":
#     # Example of starting a new conversation
#     result = real_estate_chatbot("Hi, I'm looking for apartments in New York")
#     print(f"Response: {result['response']}")
#     print(f"Thread ID: {result['thread_id']}")
    
#     # Example of continuing a conversation
#     thread_id = result['thread_id']
#     follow_up = continue_conversation("What's your name?", thread_id)
#     print(f"Follow-up response: {follow_up['response']}")
    
#     # Example of asking for clarification about something mentioned earlier
#     clarification = continue_conversation("Can you tell me more about the amenities you mentioned?", thread_id)
#     print(f"Clarification response: {clarification['response']}")



