import operator
from typing import Annotated, List, TypedDict
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_community.tools import tool 
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Hoax Buster AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

search = DuckDuckGoSearchRun()

@tool 
def cari_berita_terkini(query: str):
    """Gunakan tool ini untuk mencari berita, fakta, atau informasi terbaru di internet."""
    return search.run(query)

tools = [cari_berita_terkini]

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    search_data: str
    analysis: str
    final_answer: str
    steps_log: Annotated[List[str], operator.add]

def researcher_node(state: AgentState):
    researcher_llm = model.bind_tools(tools)
    sys_msg = SystemMessage(
        content="""Kamu adalah Peneliti Senior. Tugasmu adalah mencari fakta keras (hard facts) untuk memverifikasi klaim pengguna.
                   Gunakan tool pencarian jika diperlukan. Jangan menyimpulkan dulu, cukup kumpulkan data."""
    )
    messages = [sys_msg] + state["messages"]
    response = researcher_llm.invoke(messages)

    return {"messages": [response], "steps_log": ["🕵️ Researcher sedang bekerja..."]}

def analyst_node(state: AgentState):
    messages = state["messages"]
    sys_msg = """Kamu adalah Verifikator Fakta (Fact Checker) yang kritis.
    Tugasmu:
    1. Baca pertanyaan pengguna.
    2. Baca hasil pencarian dari tool messages sebelumnya.
    3. Tentukan apakah klaim tersebut: BENAR (FACT), SALAH (HOAX), atau MENYESATKAN (MISLEADING).
    4. Jelaskan alasan logismu berdasarkan data yang ada.
    
    PENTING: Jangan membuat format laporan akhir dulu. Fokus pada analisis fakta.
    """
    analysis_prompt = [SystemMessage(content=sys_msg)] + messages
    response = model.invoke(analysis_prompt)
    return {"analysis": response.content, "steps_log": ["⚖️ Verifikator sedang memverifikasi data..."]}

def writer_node(state: AgentState):
    analysis_content = state["analysis"]
    original_question = state["messages"][0].content
    sys_msg = f"""
    Kamu adalah Editor Berita yang ramah dan jelas.
    Tugasmu adalah menjawab pertanyaan user berdasarkan analisis verifikator.
    
    Pertanyaan User: {original_question}
    Analisis Verifikator: {analysis_content}
    
    Format jawabanmu:
    1. <b>Status</b>: [HOAX / FAKTA / DISINFORMASI] (Gunakan Huruf Kapital dan Bold)
    2. <b>Penjelasan</b>: Rangkuman singkat dan padat.
    3. <b>Kesimpulan</b>: Saran untuk pengguna.
    
    Gunakan Bahasa Indonesia yang baik dan tidak kaku.
    """
    response = model.invoke([SystemMessage(content=sys_msg)])
    return {"final_answer": response.content, "messages": [response], "steps_log": ["✍️ Writer sedang menulis laporan akhir..."]}

def should_continue(state:AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "analyst"

# workflow graph
workflow = StateGraph(
    AgentState
)

workflow.add_node("researcher", researcher_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "researcher")
workflow.add_conditional_edges(
    "researcher",
    should_continue,
    {
        "tools": "tools",
        "analyst": "analyst"
    }
)
workflow.add_edge("tools", "analyst")
workflow.add_edge("analyst", "writer")
workflow.add_edge("writer", END)

graph = workflow.compile()

# API endpoints
class QueryRequest(BaseModel):
    question: str

@app.post("/analyze")
async def analyze_claim(req: QueryRequest):
    """
    Endpoint utama untuk memproses pertanyaan user
    """
    try:
        inputs = {"messages": [HumanMessage(content=req.question)], "steps_log": []}

        result = await graph.ainvoke(inputs)
        return {
            "status": "success",
            "logs": result.get("steps_log", []),
            "final_answer": result.get("final_answer", "Maaf, gagal memproses")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# arahkan ke index,html
app.mount("/", StaticFiles(directory="static", html=True), name="static")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)