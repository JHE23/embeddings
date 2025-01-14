from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from harry_potter_facts import *
from api_token import *
import transformers
import torch

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
model_id = "meta-llama/Llama-3.2-3B-Instruct"

try:
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    print("database loaded")

except Exception:
    vector_store = FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatL2(),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    print("new database created")
    print(vector_store.add_texts(texts=harry_potter_facts))
    vector_store.save_local("faiss_index")

# llm = HuggingFaceEndpoint(
#     repo_id=model_id,
#     task="question-answering",
#     max_new_tokens=50,
#     do_sample=False,
#     repetition_penalty=1.03,
#     huggingfacehub_api_token=api_token,
#     temperature=0.01
# )

# chat = ChatHuggingFace(llm=llm, verbose=True)

# messages = [
#     ("system", "You are the creator of the children's book Harry Potter. Answer the user question about Harry Potter using only the following context: Ron has ginger hair"),
#     ("human", "What colour is ron's hair?"),
# ]

# answer = chat.invoke(messages)
# print(answer.content)

#retriever = vector_store.as_retriever()

question = "Who wrote Harry Potter?"
relevant_texts = vector_store.similarity_search_with_relevance_scores(question)
context = ""
for text in relevant_texts:
    score = text[1]
    content = text[0].page_content
    print(score, content)
    if score > 0.1:
        context = context + content

if len(context) == 0:
    context = "No relevant information in database"
print(context)

prompt = f"""
- You are a helpful assistant that answers questions based only on the children's book *Harry Potter*.
- Limit your answer to 3 sentences or less.
- Ensure your answer focuses only on *Harry Potter* and does not include unrelated information.
- Respond using only the context provided below, without incorporating any outside knowledge. 
- If the context contains no relevant information required to answer the user question, reply with I'm sorry my database does not contain this information.

Context: {context}
"""

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": question},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])

messages = [
    {"role": "user", "content": question},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
