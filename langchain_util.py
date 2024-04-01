import os
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import FewShotPromptTemplate

from few_shot_learning import few_shots

def setup(verbose = False):
    #load environment variables
    load_dotenv(find_dotenv())
    # create llm object
    #llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key = os.getenv('GOOGLE_API_KEY'), temperature=0.1)
    llm2 = GoogleGenerativeAI(model="models/text-bison-001", google_api_key = os.getenv('GOOGLE_API_KEY'), temperature=0.1)

    # connect with SQL database
    db_user = "root"
    db_password = "2526"
    db_host = "localhost"
    db_name = "atliq_tshirts"

    try:
        db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", sample_rows_in_table_info = 3)
    except:
        print("Critical Error: can't establish connection with the SQL server..")
        return
    
    # using HuggingFace for computing embedding of queries
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    e = embeddings.embed_query("How many t-shirts do we have left for Nike in XS size and white color?")

    to_vectorize = [" ".join(example.values()) for example in few_shots]

    vectorStore = Chroma.from_texts(to_vectorize, embedding = embeddings, metadatas = few_shots)

    # select k similar examples in vectorstore
    exampleSelector = SemanticSimilarityExampleSelector(vectorstore=vectorStore, k = 2)

    example_prompt = PromptTemplate(
        input_variables = ["Question", "SQLQuery", "SQLResult", "Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}"
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector = exampleSelector, 
        example_prompt = example_prompt,
        prefix=_mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "tok_k"], # variables used in the prompt
    )

    #db_chain = create_sql_agent(llm=llm, db=db, verbose = True, prompt=few_shot_prompt)
    db_chain_fs = SQLDatabaseChain(llm=llm2, database=db, verbose=verbose, prompt=few_shot_prompt)
    
    return db_chain_fs
