import os
import json
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from .env file!")

# Load PDFs from folder
pdf_folder = r"C:\Users\Shardul More\OneDrive\Desktop\miniprojecct\clgminiproject\pdf"
pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith(".pdf")]

data = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    data.extend(loader.load())

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
docs = text_splitter.split_documents(data)

# Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store in ChromaDB
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
    persist_directory="./vector_db"
)

# Define retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, max_tokens=1500, response_format="json")

# MCQ Prompt Template with Dynamic Question Count
mcq_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI MCQ generator. Your response MUST be STRICT JSON."),
    ("human", 
        "Generate exactly {num_questions} multiple-choice questions (MCQs) based on the following content:\n\n"
        "Topic: {topic}\n\n"
        "{context}\n\n"
        "**Strict Difficulty Assignment:**\n"
        "- If the requested difficulty is 'Easy', all MCQs **must** be in range (1-3).\n"
        "- If 'Medium', all MCQs **must** be exactly in range (4-6).\n"
        "- If 'Hard', all MCQs **must** be in range (7-10).\n\n"
        "**Do NOT mix difficulty levels. Follow the requested difficulty precisely!**\n\n"
        "**Format Response as JSON:**\n"
        "[\n"
        "  {{\n"
        "    \"question\": \"What is AI?\",\n"
        "    \"options\": {{\"A\": \"Artificial Intelligence\", \"B\": \"Automated Input\", \"C\": \"Analog Information\", \"D\": \"None\"}},\n"
        "    \"answer\": \"A\",\n"
        "    \"difficulty\": 7  # Hard level example\n"
        "  }}\n"
        "]\n\n"
        "⚠️ **Return JSON ONLY. No explanations or extra text.**"
    )
])

def generate_mcqs(topic, text, num_questions=10):
    """ Generate exactly 'num_questions' MCQs using LLM based on the specified topic. """
    for attempt in range(3):  # Retry up to 3 times
        try:
            response = llm.invoke(mcq_prompt.invoke({
                "topic": topic, 
                "context": text, 
                "num_questions": num_questions
            }))
            response_text = response.content if hasattr(response, "content") else str(response)

            # Clean JSON response
            response_text = re.sub(r"```json\n|\n```", "", response_text).strip()
            mcqs = json.loads(response_text)

            if isinstance(mcqs, list) and len(mcqs) == num_questions:
                return mcqs  # Success

            print(f"⚠️ Invalid MCQ response format: {mcqs}")
        except json.JSONDecodeError:
            print("⚠️ JSON Parsing Error. Retrying...")
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            break
    return []

def create_mcq_json():
    topic_input = input("Enter the topic for MCQ generation: ")
    topic_parts = topic_input.split(",")
    topic = topic_parts[0].strip()
    num_questions = int(topic_parts[1].strip()) if len(topic_parts) > 1 and topic_parts[1].strip().isdigit() else 10
    
    relevant_docs = retriever.invoke(topic)
    if not relevant_docs:
        print(f"❌ No relevant content found for topic: {topic}")
        return []
    
    extracted_text = " ".join([doc.page_content for doc in relevant_docs])
    mcq_list = generate_mcqs(topic, extracted_text, num_questions)
    
    if not mcq_list:
        print("❌ No valid MCQs generated. Try modifying the topic.")
        return []
    
    filename = f"mcqs_{topic.replace(' ', '_')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(mcq_list, f, indent=4, ensure_ascii=False)
    
    print(f"✅ {num_questions} MCQs successfully generated for topic '{topic}' and saved as '{filename}'.")
    return mcq_list

if __name__ == "__main__":
    create_mcq_json()
