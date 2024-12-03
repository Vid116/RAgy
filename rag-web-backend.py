from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import Dict
import json
import logging
from Talk_Vector import RAGConversationAgent
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_websockets=True
)

# Additional middleware for WebSocket headers
@app.middleware("http")
async def add_websocket_headers(request, call_next):
    response = await call_next(request)
    if not isinstance(response, Response):
        return response
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

active_connections: Dict[str, WebSocket] = {}

rag_agent = RAGConversationAgent(
    db_path="./vector_db_MD",
    collection_name="Car_stuff",
    model_name="gpt-4",
    temperature=0.7,
    max_history_length=10
)

@app.websocket("/chat/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_connections[session_id] = websocket
    logger.info(f"Client connected: {session_id}")
    
    try:
        # Send welcome message
        await websocket.send_json({
            "message": "Welcome! You're connected to the chat server.",
            "type": "system"
        })
        
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            logger.info(f"Message received from {session_id}: {message}")
            
            # Echo back user's message first
            await websocket.send_json({
                "message": message,
                "type": "user"
            })
            
            try:
                # Get response from RAG agent
                response = rag_agent.chat(message)
                
                # Send agent's response
                await websocket.send_json({
                    "message": response,
                    "type": "agent"
                })
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({
                    "message": f"Error processing your request: {str(e)}",
                    "type": "system"
                })
                
    except WebSocketDisconnect:
        active_connections.pop(session_id, None)
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
        active_connections.pop(session_id, None)

@app.get("/")
async def root():
    return {"message": "WebSocket server is running"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)