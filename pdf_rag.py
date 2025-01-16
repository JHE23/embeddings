from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import transformers
import torch
import PyPDF2 

embeddings_model_id = "sentence-transformers/all-mpnet-base-v2"
llm_model_id = "meta-llama/Llama-3.2-3B-Instruct"

class rag_llm:
    def __init__(
        self,
        vector_db_dir: str = None,
        llm_model_id: str = llm_model_id,
        embedding_model_id: str = embeddings_model_id
    ):
        self.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_id),
            persist_directory=vector_db_dir,
        )
        self.llm = transformers.pipeline(
            "text-generation",
            model=llm_model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.llm_prompt = f"""
        - You are a helpful assistant that answers questions based only on the context provided.
        - Limit your answer to 6 sentences or less.
        - Respond using only the context provided below, without incorporating any outside knowledge. 
        - If the context contains no relevant information required to answer the user question, reply with I'm sorry my database does not contain this information.

        Context: 
        """

    def add_pdf_to_vector_store(self, file_path: str):
        text = self.extract_text_from_pdf(file_path)
        text_list = text.splitlines()
        chunks = self.get_chunks_from_docs(docs=text_list, chunk_size=800, overlap=200)
        self.vector_store.add_texts(chunks)
    
    def ask(self, question: str, prompt: str = None):
        if not prompt:
            prompt = self.llm_prompt
        context = ""

        relevant_texts = self.vector_store.similarity_search_with_relevance_scores(question)
        for text in relevant_texts:
            score = text[1]
            content = text[0].page_content
            print(score, content)
            print("\n\n")
            if score > 0.1:
                context = context + "\n\n" + content
        if len(context) == 0:
            context = "No relevant information in database"

        prompt = prompt + context
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
        outputs = self.llm(
            messages,
            max_new_tokens=256,
        )
        return outputs[0]["generated_text"][-1]

    @staticmethod
    def extract_text_from_pdf(pdf_file):
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages[3:]:
            text += page.extract_text()
        return text

    @staticmethod
    def get_chunks_from_docs(docs: list[str], chunk_size: int, overlap: int):
        chunks = []
        chunk = ""
        buffer = ""
        for doc in docs:
            #print(len(chunk))
            if len(chunk) < chunk_size:
                chunk = chunk + doc
                if len(chunk) > (chunk_size - overlap):
                    buffer = buffer + doc
            else:
                chunks.append(chunk)
                chunk = buffer + doc
                buffer = ""
        return chunks

if __name__ == "__main__":
    doc_path = "English ESPP FAQ.pdf"
    question = "How does ESPP work?"

    rag = rag_llm()
    rag.add_pdf_to_vector_store(file_path=doc_path)
    response = rag.ask(question=question)
    print(response)
