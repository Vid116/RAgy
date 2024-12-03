import os
from typing import Optional, List
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def delete_file_from_db(file_name: str) -> Optional[List[str]]:
    if not file_name or not isinstance(file_name, str):
        raise ValueError("Invalid file name provided")
        
    try:
        db = Chroma(
            persist_directory="./vector_db_MD",
            embedding_function=OpenAIEmbeddings(),
            collection_name="Car_stuff"
        )
        
        results = db.get(where={"source_file": file_name})
        if not results or not results['ids']:
            print(f"File not found in database: {file_name}")
            return None
            
        confirm = input(f"Found {len(results['ids'])} chunks from {file_name}. Delete? (y/n): ")
        if confirm.lower() == 'y':
            db._collection.delete(ids=results['ids'])
            print(f"Deleted {len(results['ids'])} chunks from {file_name}")
            return results['ids']
        else:
            print("Deletion cancelled")
            return None
            
    except Exception as e:
        print(f"Error accessing database: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        file_name = input("Enter file path to delete: ")
        delete_file_from_db(file_name)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")