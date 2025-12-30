import operator
import json
import os
from typing import Annotated, List, TypedDict
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_community.tools import tool 
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

app = FastAPI(title="Hoax Buster AI Agent")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Setup
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Menggunakan APIWrapper agar bisa mengambil URL dan Snippet secara terpisah
ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)

@tool 
def cari_berita_terkini(query: str):
    """
    Gunakan tool ini untuk mencari berita. 
    Mengembalikan data JSON berisi Judul, Snippet, dan Link asli.
    """
    # Mengambil hasil terstruktur (list of dict)
    try:
        results = ddg_wrapper.results(query, max_results=5)
        return json.dumps(results) 
    except Exception as e:
        return json.dumps([{"error": str(e)}])

tools = [cari_berita_terkini]

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    analysis: str
    final_answer: str
    steps_log: Annotated[List[str], operator.add]

def researcher_node(state: AgentState):
    researcher_llm = model.bind_tools(tools)
    sys_msg = SystemMessage(
        content="""Kamu adalah Peneliti Senior. 
        Tugasmu adalah mencari fakta keras (hard facts) dan sumber link terpercaya.
        Gunakan tool pencarian untuk mendapatkan data terbaru."""
    )
    messages = [sys_msg] + state["messages"]
    response = researcher_llm.invoke(messages)
    return {"messages": [response], "steps_log": ["🕵️ Researcher sedang mengumpulkan bukti digital..."]}

def analyst_node(state: AgentState):
    messages = state["messages"]
    sys_msg = """Kamu adalah Verifikator Fakta (Fact Checker) yang kritis.
    
    Tugasmu:
    1. Analisis hasil pencarian dari researcher.
    2. Tentukan status: BENAR (FACT), SALAH (HOAX), atau MENYESATKAN (MISLEADING).
    3. Tentukan **Confidence Score (0-100)** berdasarkan kualitas bukti.
       - 90-100: Bukti sangat kuat & banyak sumber mainstream.
       - 50-89: Ada bukti tapi konteks perlu diperjelas.
       - 0-49: Tidak ada bukti valid / sumber abal-abal.
    4. Jelaskan alasan logismu.
    """
    analysis_prompt = [SystemMessage(content=sys_msg)] + messages
    response = model.invoke(analysis_prompt)
    return {"analysis": response.content, "steps_log": ["⚖️ Menganalisis kredibilitas & menghitung skor..."]}

def writer_node(state: AgentState):
    analysis_content = state["analysis"]
    original_question = state["messages"][0].content
    
    # Mencari data JSON dari tool message terakhir untuk diambil link-nya
    search_data_context = "Data tidak ditemukan."
    for msg in reversed(state["messages"]):
        if msg.type == "tool":
            search_data_context = msg.content
            break

    sys_msg = f"""
    Kamu adalah Editor Berita AI & Frontend Developer.
    Tugasmu menyajikan laporan akhir dalam format **HTML MURNI** (tanpa Markdown ```html).
    
    Pertanyaan User: {original_question}
    Analisis Verifikator: {analysis_content}
    Data Pencarian Mentah (JSON): {search_data_context}

    Instruksi Output HTML (Gunakan Tailwind CSS):
    
    1. **Header Status**: 
       Buat badge besar. Jika HOAX pakai bg-red-500, FAKTA bg-green-500, MISLEADING bg-yellow-500.
    
    2. **Confidence Score Bar**:
       Analisis skor dari verifikator. Buat Progress Bar.
       Format:
       <div class="mb-6 p-4 bg-slate-900/50 rounded-lg border border-slate-700">
         <div class="flex justify-between text-sm text-slate-300 mb-2 font-mono">
           <span>TINGKAT KEPERCAYAAN AI</span>
           <span class="text-white font-bold">XX%</span>
         </div>
         <div class="w-full bg-slate-700 rounded-full h-2.5">
           <div class="h-2.5 rounded-full transition-all duration-1000" style="width: XX%; background-color: [WARNA_SESUAI_SKOR]"></div>
         </div>
       </div>

    3. **Penjelasan Singkat**: 
       Tulis rangkuman analisis yang mudah dibaca warga +62. Gunakan paragraf singkat.

    4. **Sumber Referensi (Wajib Valid)**:
       Ambil dari 'link' dan 'title' di Data Pencarian Mentah.
       Format:
       <div class="mt-6 border-t border-slate-700 pt-4">
         <h4 class="text-sm font-semibold text-slate-400 mb-3 flex items-center gap-2">
            🔗 Sumber Terverifikasi
         </h4>
         <ul class="space-y-2 text-sm">
           <li>
             <a href="URL_ASLI" target="_blank" class="flex items-center gap-2 text-blue-400 hover:text-blue-300 hover:underline transition-colors group">
               <span class="w-1.5 h-1.5 rounded-full bg-blue-500 group-hover:animate-ping"></span>
               JUDUL_SUMBER
             </a>
           </li>
         </ul>
       </div>

    HANYA KEMBALIKAN KODE HTML DI DALAM TAG DIV. JANGAN ADA TEKS LAIN.
    """
    
    response = model.invoke([SystemMessage(content=sys_msg)])
    return {"final_answer": response.content, "messages": [response], "steps_log": ["✍️ Menyusun laporan akhir & daftar pustaka..."]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "analyst"

workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "researcher")
workflow.add_conditional_edges("researcher", should_continue, {"tools": "tools", "analyst": "analyst"})
workflow.add_edge("tools", "analyst")
workflow.add_edge("analyst", "writer")
workflow.add_edge("writer", END)

graph = workflow.compile()

class QueryRequest(BaseModel):
    question: str

@app.post("/analyze")
async def analyze_claim(req: QueryRequest):
    try:
        inputs = {"messages": [HumanMessage(content=req.question)], "steps_log": []}
        result = await graph.ainvoke(inputs)
        return {
            "status": "success",
            "logs": result.get("steps_log", []),
            "final_answer": result.get("final_answer", "<p>Gagal memproses data.</p>")
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount folder static agar index.html bisa diakses langsung
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)