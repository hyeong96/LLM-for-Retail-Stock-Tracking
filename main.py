from langchain_google_genai import GoogleGenerativeAI

api_key = 'AIzaSyC053_xqmQqVFgWYOMPoaC8kwnhPGI0SrI'

llm = GoogleGenerativeAI(model="models/gemini-pro", google_api_key = api_key, temperature=0.2)
poem = llm("what model are you")
print(poem)