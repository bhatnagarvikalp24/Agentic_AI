import os
from dotenv import load_dotenv
from openai import OpenAI
from vanna.remote import VannaDefault
from langchain_community.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_PATH = './Actuarial_Data (2).db'

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VANNA_API_KEY = os.getenv("vanna_api_key")
VANNA_MODEL_NAME = os.getenv("vanna_model_name")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# LLM Configuration
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 500

# Vanna Configuration
vn_model = VannaDefault(model=VANNA_MODEL_NAME, api_key=VANNA_API_KEY)
vn_model.connect_to_sqlite(DATABASE_PATH)

# OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# Embeddings
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Keywords that usually indicate monetary columns
MONEY_KEYWORDS = ["loss", "premium", "amount", "cost", "ibnr", "ult", "total", "claim", "reserve", "payment"]

# FAISS Configuration
FAISS_INDEX_PATH = "faiss_index/"

# Documentation for database schema
DOCUMENTATION = """
PnC_Data Table:
- Reserve Class contains insurance business lines such as 'Property', 'Casualty', 'Marine', 'Motor', etc.
- Exposure Year refers to the year in which the insured risk was exposed to potential loss.
- RI Type identifies whether the record is 'Gross' or one of the reinsurance types such as 'Ceded - XOL', 'Ceded - QS', 'Ceded - CAP', 'Ceded - FAC', or 'Ceded - Others'.
- Branch indicates the geographical business unit handling the contract, e.g., 'Europe', 'LATAM', 'North America'.
- Loss Type captures the nature of the loss, and may be one of: 'ATT', 'CAT', 'LARGE', 'THREAT', or 'Disc'.
- Underwriting Year represents the year in which the policy was underwritten or originated.
- Incurred Loss represents the total loss incurred to date, including paid and case reserves.
- Paid Loss is the portion of the Incurred Loss that has already been settled and paid out.
- IBNR is calculated as the difference between Ultimate Loss and Incurred Loss.
- Ultimate Loss is the projected final value of loss.
- Ultimate Premium refers to the projected premium expected to be earned.
- Loss Ratio is calculated as Ultimate Loss divided by Ultimate Premium.
- AvE Incurred = Expected - Actual Incurred.
- AvE Paid = Expected - Actual Paid.
- Budget Premium is the forecasted premium for budgeting.
- Budget Loss is the projected loss for budgeting.
- Earned Premium is the portion of the premium that has been earned.
- Case Reserves = Incurred Loss - Paid Loss.
"""

# State keys that are set at entry
STATE_KEYS_SET_AT_ENTRY = [
    "user_prompt", 
    "doc_loaded", 
    "document_path", 
    "vanna_prompt", 
    "fuzzy_prompt",
    "route",
    "sql_result",
    "sql_query",
    "web_links",
    "updated_doc_path",
    "chart_info",
    "comparison_summary",
    "general_summary",
    "faiss_summary", 
    "faiss_sources",
    "faiss_images"
]

# Insurance keywords for search enhancement
INSURANCE_KEYWORDS = [
    "insurance", "insurer", "claim", "premium", "underwriting",
    "policy", "fraud", "broker", "actuary", "reinsurance", "coverage", 
    "Actuarial", "reserving", "P&L", "Profit and Loss"
]

# Domain filters for search
SEARCH_DOMAIN_FILTER = "site:deloitte.com OR site:irdai.gov.in OR site:insurancebusinessmag.com OR site:swissre.com"

# LLM call function
def call_llm(prompt: str) -> str:
    """Call OpenAI LLM with the given prompt"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an intelligent AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI call failed: {e}"

# Utility function for state pruning
def prune_state(state, exclude: list) -> dict:
    """Remove specified keys from state dictionary"""
    return {k: v for k, v in state.items() if k not in exclude} 