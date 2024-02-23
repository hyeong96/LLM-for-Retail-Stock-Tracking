import os
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_experimental.sql import SQLDatabaseChain

def main():
    #load environment variables
    load_dotenv(find_dotenv())
    # create llm object
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key = os.getenv('GOOGLE_API_KEY'), temperature=0.2)
    llm2 = GoogleGenerativeAI(model="models/text-bison-001", google_api_key = os.getenv('GOOGLE_API_KEY'), temperature=0.2)

    # connect with SQL database
    db_user = "root"
    db_password = "2526"
    db_host = "localhost"
    db_name = "atliq_tshirts"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", sample_rows_in_table_info = 3)
    #print(db.table_info)

    db_chain = create_sql_agent(llm=llm, db=db, verbose = False)
    db_chain_2 = SQLDatabaseChain(llm=llm2, database=db, verbose=False)
    question = input("Ask me a question!: ")

    try:
        qns1 = db_chain(question)
        print("Answer from gemini: ", qns1['output'])
    except:
        print("Couldn't retrieve the answer..")
    try:
        qns2 = db_chain_2(question)
        print("Answer from google palm: ", qns2['result'])
    except:
        print("Couldn't retrieve the answer..")

if __name__ == "__main__":
    while True:
        main()