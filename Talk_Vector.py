import os
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from datetime import datetime
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential

from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from Prompts import Search_Prompt

AUTOMOTIVE_QUERY_PROMPT = Search_Prompt()

@dataclass
class DocumentMetadata:
    """Structured container for document metadata"""
    source_file: str
    file_type: str
    timestamp: str
    chunk_id: Optional[int] = None
    
class RAGConversationAgent:
    def __init__(
        self, 
        db_path: str = "./vector_db",
        collection_name: str = "my_documents",
        model_name: str = "gpt-4",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 0.7,
        max_history_length: int = 10
    ):
        self.console = Console()
        self.max_history_length = max_history_length
        
        try:
            self._initialize_embeddings(embedding_model)
            self._initialize_database(db_path, collection_name)
            self._initialize_retriever()
            self._initialize_llm(model_name, temperature)
            self._initialize_chain()
            
            self.chat_history: List = []
            
        except Exception as e:
            self.console.print(f"[red]Initialization Error: {str(e)}[/red]")
            raise

    def _initialize_embeddings(self, model_name: str) -> None:
        try:
            self.embeddings = OpenAIEmbeddings(model=model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")

    def _initialize_database(self, db_path: str, collection_name: str) -> None:
        try:
            self.db = Chroma(
                persist_directory=db_path,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            collection_stats = self.db._collection.count()
            if collection_stats == 0:
                raise ValueError("Database is empty")
                
            self.console.print(f"[green]Successfully loaded database with {collection_stats} documents[/green]")
            
        except Exception as e:
            raise RuntimeError(f"Database initialization failed: {str(e)}")

    def _initialize_retriever(self) -> None:
        """Initialize the retriever using the latest LangChain patterns"""
        self.retriever = self.db.as_retriever(
            search_type="similarity",  # Can be "similarity", "mmr", or "similarity_score_threshold"
            search_kwargs={
                "k": 4,  # Number of documents to return
                # If using similarity_score_threshold, you can add:
                # "score_threshold": 0.5
                # If using mmr, you can add:
                # "fetch_k": 20,
                # "lambda_mult": 0.5
            }
        )

    def _initialize_llm(self, model_name: str, temperature: float) -> None:
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )

    def _initialize_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a direct and efficient Slovenian insurance advisor. 
            Answer questions in 2-3 sentences maximum while being informative and professional.
            
            Core Instructions:
            - Use formal Slovenian ("vikanje")
            - Only use information from the provided context
            - Always suggest complementary insurance products from same package ("paket")
            - Highlight package discounts and combined benefits
            - Explain why suggested products work well together
            - Acknowledge if specific information is missing
            
            Response Format:
            1. Direct answer about requested insurance (1 sentence)
            2. Related package products suggestion with benefit (1-2 sentences)
            3. Package discount mention if applicable (1 sentence)"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
            ("system", "Context information: {context}")
        ])

        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "chat_history": lambda x: self._format_chat_history(),
                "question": RunnablePassthrough()
            }
            | prompt 
            | self.llm 
            | StrOutputParser()
        )

        # Define the RAG chain
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "chat_history": lambda x: self._format_chat_history(),
                "question": RunnablePassthrough()
            }
            | prompt 
            | self.llm 
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _manage_chat_history(self) -> None:
        if len(self.chat_history) > self.max_history_length * 2:
            self.chat_history = self.chat_history[-self.max_history_length * 2:]
            self.console.print("[yellow]Chat history truncated to prevent context overflow[/yellow]")

    def _format_chat_history(self) -> List:
        if not self.chat_history:
            return []
        
        formatted_messages = []
        for msg in self.chat_history:
            if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                formatted_messages.append(msg)
        
        return formatted_messages

    def display_retrieved_documents(self, documents, search_query: str = None):
        if search_query:
            self.console.print(f"\n[bold purple]Search Query:[/bold purple] {search_query}")
            
        if not documents:
            self.console.print("[yellow]No relevant documents were retrieved.[/yellow]")
            return
            
        for idx, doc in enumerate(documents, 1):
            metadata = DocumentMetadata(
                source_file=doc.metadata.get('source_file', 'Unknown'),
                file_type=doc.metadata.get('file_type', 'Unknown'),
                timestamp=doc.metadata.get('timestamp', 'Unknown'),
                chunk_id=doc.metadata.get('chunk_id')
            )
            
            self._display_document_panel(idx, doc, metadata)

    def _display_document_panel(self, idx: int, doc: Any, metadata: DocumentMetadata):
        content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        
        panel_content = (
            f"[bold]Document {idx}[/bold]\n"
            f"[yellow]File:[/yellow] {os.path.basename(metadata.source_file)}\n"
            f"[yellow]Type:[/yellow] {metadata.file_type}\n"
            f"[yellow]Timestamp:[/yellow] {metadata.timestamp}\n"
            f"[yellow]Chunk ID:[/yellow] {metadata.chunk_id}\n"
            f"\n[green]Content Preview:[/green]\n{content_preview}"
        )
        
        self.console.print(Panel(panel_content, border_style="blue"))

    def chat(self, message: str) -> str:
        self.console.print("\n[bold blue]Processing query...[/bold blue]")
        
        try:
            self._manage_chat_history()
            
            # Get response using the chain
            response = self.chain.invoke(message)
            
            # Get the retrieved documents for display - updated to use new pattern
            docs = self.retriever.invoke(message)
            self.display_retrieved_documents(docs, message)
            
            # Update chat history
            self.chat_history.extend([
                HumanMessage(content=message),
                AIMessage(content=response)
            ])
            
            return response
            
        except Exception as e:
            error_msg = f"Error during chat processing: {str(e)}"
            self.console.print(f"[red]{error_msg}[/red]")
            return f"I encountered an error while processing your request: {str(e)}"

    def start_interactive_chat(self):
        self.console.print("[bold green]Enhanced RAG Conversational Agent[/bold green]")
        self.console.print("[bold green]Commands: 'exit' to end, 'debug' for system info, 'clear' to reset history[/bold green]")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == "exit":
                    self.console.print("\n[bold green]Session ended. Goodbye![/bold green]")
                    break
                    
                if user_input.lower() == "debug":
                    self._display_debug_info()
                    continue
                    
                if user_input.lower() == "clear":
                    self.chat_history.clear()
                    self.console.print("[green]Chat history cleared[/green]")
                    continue
                
                response = self.chat(user_input)
                self.console.print(f"\n[bold cyan]Assistant:[/bold cyan] {response}")
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Session interrupted by user[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Unexpected error: {str(e)}[/red]")

    def _display_debug_info(self):
        self.console.print("\n[bold blue]System Status:[/bold blue]")
        self.console.print(f"Documents in DB: {self.db._collection.count()}")
        self.console.print(f"Chat history length: {len(self.chat_history)}")
        self.console.print(f"Max history length: {self.max_history_length}")
        self.console.print(f"Retriever settings: {self.retriever.search_kwargs}")
        formatted_history = self._format_chat_history()
        self.console.print(f"[blue]Chat History:[/blue]\n{formatted_history}")

def main():
    console = Console()
    
    try:
        load_dotenv()
        
        agent = RAGConversationAgent(
            db_path="./vector_db_MD",
            collection_name="Car_stuff",
            model_name="gpt-4",
            temperature=0.7,
            max_history_length=10
        )
        
        agent.start_interactive_chat()
        
    except Exception as e:
        console.print(f"[red]Failed to initialize agent: {str(e)}[/red]")
        console.print_exception()

if __name__ == "__main__":
    main()