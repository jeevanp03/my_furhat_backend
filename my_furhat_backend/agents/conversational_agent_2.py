import os
import time
import logging
import uuid
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing_extensions import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults

# Import your custom LLM/chatbot classes.
from my_furhat_backend.models.chatbot_factory import Chatbot_HuggingFace, Chatbot_LlamaCpp, create_chatbot
from my_furhat_backend.utils.util import clean_output
from my_furhat_backend.config.settings import config

# Import your RAG class.
from my_furhat_backend.RAG.rag_flow import RAG

# ---------------------------
# Define external chains as in the article:
# ---------------------------

# Assume llm is available from your chatbot instance:
llm = create_chatbot("llama", model_id="my_furhat_backend/ggufs_models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf").llm

# Implement the Router
router_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, 
    prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
    in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
    or 'vectorstore' based on the question. Return a JSON with a single key 'datasource' and 
    no preamble or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)
question_router = router_prompt | llm | JsonOutputParser()

# Implement the Generate Chain
generate_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {document} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = generate_prompt | llm | StrOutputParser()

# Implement the Retrieval Grader
retrieval_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)
retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()

# Implement the Hallucination Grader
hallucination_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)
hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()

# Implement the Answer Grader
answer_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)
answer_grader = answer_grader_prompt | llm | JsonOutputParser()

# ---------------------------
# Set up our global instances:
# ---------------------------
rag_instance = RAG(
    hf=True,
    persist_directory="my_furhat_backend/db",
    path_to_document="my_furhat_backend/CMRPublished.pdf"
)
search_tool = TavilySearchResults(api_key=config["TAVILY_API_KEY"])
chatbot = create_chatbot("llama", model_id="my_furhat_backend/ggufs_models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf")

# ---------------------------
# Define the State.
# ---------------------------
class State(TypedDict):
    messages: Annotated[List[BaseMessage], "add_messages"]
    question: str
    documents: list         # Retrieved documents.
    web_search: str         # "Yes" or "No" flag indicating if web search is needed.
    generation: str         # Generated answer.
    websearch_docs: str     # Web search results.

# ---------------------------
# Define Workflow Node Functions.
# ---------------------------

def capture_input(state: State) -> dict:
    state.setdefault("messages", [])
    # Here we expect state["question"] is already set.
    human_msg = HumanMessage(content=state["question"])
    state["messages"].append(human_msg)
    return {"messages": state["messages"]}

def route_question(state: State) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source['datasource'])
    if source['datasource'] == 'web_search':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "retrieve"

def retrieve(state: State) -> dict:
    print("---RETRIEVE---")
    question = state["question"]
    documents = rag_instance.retrieve_similar(question, rerank=True)
    state["documents"] = documents
    return {"documents": documents, "question": question}

def grade_documents(state: State) -> dict:
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
    state["documents"] = filtered_docs
    state["web_search"] = web_search
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def decide_to_generate(state: State) -> str:
    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    if web_search == "Yes":
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def web_search(state: State) -> dict:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    docs = search_tool.invoke({"query": question})
    from langchain.schema import Document
    web_results = "\n".join([d["content"] for d in docs])
    web_results_doc = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results_doc)
    else:
        documents = [web_results_doc]
    state["documents"] = documents
    return {"documents": documents, "question": question}

def generate(state: State) -> dict:
    print("---GENERATE---")
    question = state["question"]
    context = format_docs(state["documents"])
    generation = rag_chain.invoke({"question": question, "document": context})
    state["generation"] = generation
    return {"documents": state["documents"], "question": question, "generation": generation}

def grade_generation_v_documents_and_question(state: State) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    context = format_docs(state["documents"])
    generation = state["generation"]
    score = hallucination_grader.invoke({"documents": context, "generation": generation})
    grade = score['score']
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score['score']
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

# ---------------------------
# Build the Graph Workflow.
# ---------------------------
workflow = StateGraph(State)

workflow.add_node("capture_input", capture_input)
workflow.add_node("route_question", route_question)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("decide_to_generate", decide_to_generate)
workflow.add_node("websearch", web_search)
workflow.add_node("generate", generate)
workflow.add_node("grade_generation", grade_generation_v_documents_and_question)

# Set the conditional entry point based on route_question.
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("capture_input", "route_question")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)

# Compile the graph with a MemorySaver checkpoint.
memory = MemorySaver()
compiled_graph = workflow.compile(checkpointer=memory)

# ---------------------------
# Run the Conversation.
# ---------------------------
def run_conversation_streaming():
    config = {"configurable": {"thread_id": "1"}}
    print("Welcome. This assistant uses RAG, web search, and grading routing.")
    query = input("Enter your query about the document (or 'exit' to quit): ")
    if query.strip().lower() == "exit":
        return
    system_prompt = SystemMessage(content=(
        "You are a friendly and knowledgeable assistant. "
        "Answer based on the provided document content and web search results if needed."
    ))
    state: State = {
        "question": query,
        "messages": [HumanMessage(content=system_prompt.content)],
        "documents": [],
        "web_search": "No",
        "generation": "",
        "websearch_docs": ""
    }
    print("Streaming response...")
    for step in workflow.stream(state, config, stream_mode="values"):
        pass
    final_generation = state.get("generation", "")
    if final_generation:
        print("Assistant:", clean_output(final_generation))
    else:
        print("No AI response found.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        state["question"] = user_input
        state["messages"].append(HumanMessage(content=user_input))
        for step in workflow.stream(state, config, stream_mode="values"):
            pass
        final_generation = state.get("generation", "")
        if final_generation:
            print("Assistant:", clean_output(final_generation))
        else:
            print("No AI response found.")

if __name__ == "__main__":
    run_conversation_streaming()
