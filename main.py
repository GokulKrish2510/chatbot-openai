from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory,FileChatMessageHistory
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import pandas as pd
pd.options.display.max_rows = 4000
import warnings
warnings.filterwarnings("ignore")

class Chatbot:
    def __init__(self, pdf : str):
        """
        Initializes the Chatbot class with a PDF filename.

        Args:
            pdf (str): Path to the PDF file.
        """

        # Load environment variables (Open AI)
        load_dotenv()

        # Store the PDF filename for later use
        self.pdf = pdf

        # Initialize embedding model for converting text to vectors
        self.embeddings_model = OpenAIEmbeddings()

        # Initialize ChatOpenAI object for generating text responses
        self.chat_bot = ChatOpenAI()

        # Call internal methods for further setup
        self.__embedding()
        self.__LLMChain()

    def __chunking(self):
        """
        Splits the PDF document into smaller chunks for efficient processing.

        Returns:
            list: List of text chunks extracted from the PDF.
        """

        # Load pages from the PDF using PDFPlumberLoader
        loader = PDFPlumberLoader(self.pdf)
        pages = loader.load()

        # Create a text splitter object with chunk size and overlap configuration
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=400,
        length_function=len,
        is_separator_regex=False,
        )

        # Split the loaded pages into text chunks using the splitter
        text_chunks = text_splitter.split_documents(pages)
        return text_chunks
    
    def __embedding(self):

        """
        Creates a FAISS vector database for storing and searching text embeddings.
        """
        
        # Define a local file store for caching embeddings
        store = LocalFileStore('./cache/')

        # Create a cache-backed embedding object using the store and model
        cache_embed = CacheBackedEmbeddings.from_bytes_store(
                        underlying_embeddings=self.embeddings_model,
                        document_embedding_cache = store,
                        namespace=self.embeddings_model.model
                        )
        
        # Extract text chunks from the PDF using the chunking function
        text_chunks = self.__chunking()

        # Create a FAISS vector database from the text chunks and embeddings
        self.db = FAISS.from_documents(text_chunks,cache_embed)
    
    def __prompting_and_memory(self):

        """
        Defines the prompting template and conversation memory for the chatbot.
        """

        # Define the prompting template with input variables and messages
        self.chat_prompt = ChatPromptTemplate(
                        input_variables = ['content','question','memory_messages', 'human_input'],
                        messages=[
                            MessagesPlaceholder(variable_name='memory_messages'),
                            HumanMessagePromptTemplate.from_template(
                                '''
                                You are helpful analyst chatbot interacting with human,
                                Answer the question by analysing the {content}
                                Human question {question}
                                {human_input}
                                '''
                            )
                            ]
                            )
        # Define the conversation memory with buffer size and key names
        self.chat_memory = ConversationBufferMemory(
            memory_key='memory_messages',
            input_key="human_input",
            return_messages=True)
    
    def __LLMChain(self):

        """
        Combines the prompting template, memory, and ChatOpenAI object into an LLMChain.
        """

        # Call the prompting and memory setup function  
        self.__prompting_and_memory()

        # Create the LLMChain with the chat model, prompt, and memory
        self.chatbot_chain = LLMChain(
            llm=self.chat_bot,
            prompt= self.chat_prompt,
            memory=self.chat_memory,
            )
        
    def chat_bot(self,question:str):

        """
        Answers a question based on the PDF content and provided question.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The generated response to the question.
        """
        
        # Search for similar content in the vector database from the question
        content_db = self.db.similarity_search(question)

        # Passing the content to chatbot to get answer for the user and return 
        result = self.chatbot_chain({'content':str(content_db[0].page_content),'question':question, 'human_input':''})
        return result['text']
