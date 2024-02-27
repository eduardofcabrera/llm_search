import requests
import os

from tqdm import tqdm

from operator import itemgetter

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chat_models.base import BaseChatModel
from langchain_core.runnables import (
    RunnableLambda,
    RunnableBranch,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain.chains.base import Chain

URL = os.getenv("GOOGLE_SEARCH_API_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")


def if_search(model_return: str) -> bool:

    if model_return == "Use Google Search.":
        return True
    else:
        return False


def get_if_search_chain(model: BaseChatModel):

    search_prompt = SystemMessage(
        content="""
        Based on the question above, do you need to make a search on Google Search to help you answer it? Or you think that you can answer by your own?
        If you need to make a search on Google search return only: Use Google Search.
        If you don't need to make a search on Google search return only: Don't use Google Search.
        Please only return what is required.
        """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            search_prompt,
        ]
    )

    str_output_parser = StrOutputParser()

    if_search_runnable = RunnableLambda(if_search)

    chain = prompt | model | str_output_parser | if_search_runnable

    return chain


def get_chain_without_search(model: BaseChatModel):

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    str_output_parser = StrOutputParser()

    chain_without_search = RunnableParallel(
        model_return=prompt | model | str_output_parser, prompt=prompt
    )

    return chain_without_search


def get_search_query(model_return: str) -> str:
    search_query = model_return.replace("Google Query: ", "")
    return search_query


def get_chain_search_query(model: BaseChatModel):

    search_query_prompt = SystemMessage(
        content="""
        Based on the user question, what would be a good query to search on Google Search to find useful information to answer it?
        Just return:
        
        Google Query: <query>
        """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            search_query_prompt,
        ]
    )

    str_output_parser = StrOutputParser()

    search_query_chain = (
        prompt | model | str_output_parser | RunnableLambda(get_search_query)
    )

    return search_query_chain


def google_api_call(query: str) -> list:
    r = requests.get(
        url=URL,
        params={
            "key": GOOGLE_API_KEY,
            "q": query,
            "cx": SEARCH_ENGINE_ID,
        },
    )

    data = r.json()
    search_items = data["items"]
    links = [item["link"] for item in search_items]

    return links


def api_response_to_retrieval_chain(
    model,
    links: list,
):

    document_vectors = links_to_documents(links)

    web_result_prompt = """
        Below are web results from Google search that may be useful in answering the user's question. Use the results extracted from the web to help you with your answer.

        <web results>
        {context}
        </web results>

        Question: {input}"""

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("system", web_result_prompt),
        ]
    )

    str_output_parser = StrOutputParser()

    prompt_chain = create_stuff_documents_chain(
        RunnablePassthrough(), prompt, output_parser=RunnablePassthrough()
    )

    document_chain = create_stuff_documents_chain(
        model, prompt, output_parser=str_output_parser
    )

    document_and_prompt_chain = RunnableParallel(
        model_return=document_chain, prompt=prompt_chain
    )

    retriever = document_vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_and_prompt_chain)

    return retrieval_chain


def links_to_documents(links: list):

    docs = []

    for link in tqdm(links):
        # loader = WebBaseLoader(link)
        loader = RecursiveUrlLoader(
            url=link, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
        )

        try:
            doc = loader.load()
            docs.extend(doc)
        except:
            pass

    embeddings = OpenAIEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    return vector


def get_chain_with_search(model: BaseChatModel):

    query_search_chain = RunnableParallel(
        input=itemgetter("input"),
        chat_history=itemgetter("chat_history"),
        query=get_chain_search_query(model),
    )

    api_call_chain = RunnableParallel(
        input=itemgetter("input"),
        chat_history=itemgetter("chat_history"),
        query=itemgetter("query"),
        links=RunnableLambda(lambda x: google_api_call(x["query"])),
    )

    question_answer_chain = RunnableParallel(
        chat_history=itemgetter("chat_history"),
        query=itemgetter("query"),
        links=itemgetter("links"),
        question_return=lambda x: api_response_to_retrieval_chain(model, x["links"]),
    )

    search_chain = query_search_chain | api_call_chain | question_answer_chain
    return search_chain


def get_chain(model: BaseChatModel):

    if_search_chain = get_if_search_chain(model)
    with_search_chain = get_chain_with_search(model)
    without_search_chain = get_chain_without_search(model)
    branch_chain = RunnableBranch(
        (lambda x: x["if_search"], with_search_chain),
        (lambda x: not (x["if_search"]), without_search_chain),
        without_search_chain,
    )

    chain = RunnableParallel(
        input=itemgetter("input"),
        chat_history=itemgetter("chat_history"),
        if_search=if_search_chain,
    ) | RunnableParallel(
        input=itemgetter("input"),
        chat_history=itemgetter("chat_history"),
        if_search=itemgetter("if_search"),
        branch=branch_chain,
    )

    return chain


def main():

    llm = ChatOpenAI(model="gpt-4-0125-preview")

    question_ = "Who was Napoleon Bonaparte?"
    question = "Can you tell me what happened in first of July of 2023?"
    question = "Who is Marcos Cabrera from Brazil? What he does?"

    system_message = SystemMessage(
        content="You are a helpful assistant that can make queries on Google Search API to help you answer questions. You can use internet documents to help you search for informations needed to answer questions."
    )

    chat_history = [system_message]

    chain = get_chain(llm)

    while True:
        question = input("Your question: ")
        if question == "EXIT":
            break
        response = chain.invoke({"input": question, "chat_history": chat_history})

        if response["if_search"]:
            model_return = response["branch"]["question_return"]["answer"][
                "model_return"
            ]
            chat_history = response["branch"]["question_return"]["answer"]["prompt"]
        else:
            model_return = response["branch"]["model_return"]
            chat_history = response["branch"]["prompt"]

        chat_history = chat_history.to_messages()
        chat_history.append(AIMessage(content=model_return))
        print(model_return)


if __name__ == "__main__":
    main()
