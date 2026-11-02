
import asyncio
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage


ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "data" / "part4" / "faiss_openai_index"

# Load the existing OpenAI API key.
load_dotenv(ROOT / "part3" / ".env")

# Separate conversation histories for each user.
user_histories = {}
user_locks = {}


@lru_cache(maxsize=1)
def get_rag_chain():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing")

    if not (INDEX_PATH / "index.faiss").exists():
        raise FileNotFoundError(f"Missing FAISS index: {INDEX_PATH}")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model="gpt-6-luna",
        reasoning_effort="none",
        use_responses_api=False,
        max_tokens=300,
    )

    rewrite_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given the chat history and latest question, "
            "rewrite the question to be self-contained. "
            "Do not answer the question.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_retriever = create_history_aware_retriever(
        llm, retriever, rewrite_prompt
    )

    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the EE542 Cloud Computing Study Assistant. "
            "Answer using only the provided lecture content. "
            "If the answer is not in the notes, say so. "
            "Keep answers clear and concise.\n\n"
            "Lecture content:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    document_prompt = PromptTemplate.from_template(
        "Source: {source}, Page: {page}\n{page_content}"
    )

    answer_chain = create_stuff_documents_chain(
        llm,
        answer_prompt,
        document_prompt=document_prompt,
    )

    return create_retrieval_chain(
        history_retriever, answer_chain
    )


async def ask_cloud_rag(question: str, user_id: int) -> str:
    lock = user_locks.setdefault(user_id, asyncio.Lock())

    async with lock:
        history = user_histories.get(user_id, [])

        # Run synchronous LangChain work in a thread.
        result = await asyncio.to_thread(
            lambda: get_rag_chain().invoke({
                "input": question,
                "chat_history": history,
            })
        )

        answer = result["answer"]

        user_histories[user_id] = (
            history + [
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
        )[-12:]

        sources = []
        seen = set()

        for doc in result["context"]:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            key = (source, page)

            if key not in seen:
                seen.add(key)
                sources.append(f"{source}, p.{page}")

        if sources:
            answer += "\n\nRetrieved sources: " + "; ".join(sources)

        return answer
