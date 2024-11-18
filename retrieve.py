import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from pinecone import Pinecone, ServerlessSpec
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from jira import JIRA

# Load environment variables
load_dotenv()

def format_jira_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_jira_issue(summary, description):
    """
    Creates a Jira issue with the provided summary and description.

    :param summary: A concise summary of the issue.
    :param description: Detailed description of the issue.
    :return: The created Jira issue object.
    """
    jira = JIRA(server=JIRA_SERVER, basic_auth=(JIRA_USER_EMAIL, JIRA_API_TOKEN))
    issue = jira.create_issue(
        project=PROJECT_KEY,
        summary=summary,
        description=description,
        issuetype={'name': 'Task'},  # Change to 'Bug', 'Story', etc., as needed
    )
    return issue

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "embedding-ollama"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
JIRA_SERVER = os.getenv("JIRA_SERVER")  # e.g., "https://yourdomain.atlassian.net"
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_USER_EMAIL = os.getenv("JIRA_USER_EMAIL")
PROJECT_KEY = os.getenv("PROJECT_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)  # Connect to the index

if __name__ == "__main__":
    print("Retrieving...")

    embeddings_model = OllamaEmbeddings(model="llama3")
    llm = ChatOllama(model="llama3")

    query = "Fetch issue summary and details of the recent issue"
    # chain = PromptTemplate.from_template(template=query) | llm

    # Corrected usage of PineconeVectorStore
    vectorstore = PineconeVectorStore(
        index=index,  # Pass the name of the index as a string
        embedding=embeddings_model
    )

    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # retrival_chain = create_retrieval_chain(
    #     retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    # )

    # result = retrival_chain.invoke(input={"input": query})

    # print(result)

    jira_template = """You are a system designed to fetch and summarize error-related information.

    Use the provided context to identify and extract the following:
    1. Issue Summary: Provide a concise summary of the issue.
    2. Issue Details: Provide a brief description or explanation of the issue.

    Give response starting with Isuue Summary followed by Issue Details don't include any special characters at starting and ending.

    Context:
    {context}

    Question: {question}

    Response:"""

    custom_jira_prompt = PromptTemplate.from_template(jira_template)

    jira_chain = (
        {"context": vectorstore.as_retriever() | format_jira_docs, "question": RunnablePassthrough()}
        | custom_jira_prompt
        | llm
    )

    res = jira_chain.invoke(query)

    llm_response = res.content.strip()
    lines = llm_response.split("\n")
    issue_summary = lines[0].strip() if len(lines) > 0 else "No summary provided"
    issue_details = "\n".join(lines[1:]).strip() if len(lines) > 1 else "No details provided"

    # Print LLM response
    print("LLM Response:")
    print(llm_response)

    # Create Jira issue
    jira_issue = create_jira_issue(issue_summary, issue_details)
    print(f"Jira issue created: {jira_issue.key}")

    # print(llm_response.split("\n"))