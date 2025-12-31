import operator
import json
import base64
from typing import Annotated, List, TypedDict, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
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

app = FastAPI(title="Hoax Buster AI Agent - Multimodal")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# model LLama untuk query teks
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# model LLama untuk query gambar
vision_model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct", 
    temperature=0
)

# DuckDuckGo Search
ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)

@tool 
def cari_berita_terkini(query: str):
    """Mencari berita terkini. Mengembalikan JSON berisi Judul, Snippet, dan Link."""
    try:
        results = ddg_wrapper.results(query, max_results=5)
        return json.dumps(results) 
    except Exception as e:
        return json.dumps([{"error": str(e)}])

tools = [cari_berita_terkini]

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    image_data: Optional[str] # Field baru untuk gambar
    analysis: str
    final_answer: str
    steps_log: Annotated[List[str], operator.add]

def vision_node(state: AgentState):
    image_data = state.get("image_data")
    
    # Jika tidak ada gambar, langsung lanjut ke node berikutnya
    if not image_data:
        return {"steps_log": []}

    message = HumanMessage(
        content=[
            {"type": "text", "text": "Tugasmu adalah mengekstrak SEMUA teks yang ada di dalam gambar ini. Jangan tambahkan opini, hanya ekstrak teksnya apa adanya."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                },
            },
        ]
    )
    
    # Invoke model vision
    response = vision_model.invoke([message])
    extracted_text = response.content

    new_query = f"Analisis klaim berikut yang diekstrak dari gambar: '{extracted_text}'"
    
    return {
        "messages": [HumanMessage(content=new_query)], 
        "steps_log": ["👁️‍🗨️ AI Vision selesai membaca teks dalam gambar..."]
    }


def researcher_node(state: AgentState):
    researcher_llm = model.bind_tools(tools)
    last_message_content = state["messages"][-1].content
    
    sys_msg = SystemMessage(
        content=f"""Kamu adalah Peneliti Senior. Tugasmu mencari fakta keras untuk memverifikasi klaim ini: "{last_message_content}".
        Gunakan tool pencarian untuk mendapatkan data terbaru."""
    )
    # Kita reset messages context untuk researcher agar fokus pada query terakhir
    response = researcher_llm.invoke([sys_msg, HumanMessage(content=last_message_content)])
    return {"messages": [response], "steps_log": ["🕵️ Researcher sedang mengumpulkan bukti digital..."]}

def analyst_node(state: AgentState):
    tool_output = "Tidak ada data."
    for msg in reversed(state["messages"]):
        if msg.type == "tool":
            tool_output = msg.content
            break
            
    sys_msg = f"""Kamu adalah Verifikator Fakta Kritis.
    Data Pencarian: {tool_output}
    
    Tugas:
    1. Tentukan status: BENAR (FACT), SALAH (HOAX), atau MENYESATKAN (MISLEADING).
    2. Tentukan Confidence Score (0-100).
    3. Jelaskan alasan logismu berdasarkan data.
    """
    response = model.invoke([SystemMessage(content=sys_msg)])
    return {"analysis": response.content, "steps_log": ["⚖️ Menganalisis kredibilitas & menghitung skor..."]}

def writer_node(state: AgentState):
    analysis_content = state["analysis"]
    original_question = state["messages"][0].content

    search_data_context = "Data tidak ditemukan."
    for msg in reversed(state["messages"]):
        if msg.type == "tool":
            search_data_context = msg.content
            break

    # Kita berikan template HTML yang SPESIFIK agar bar persentase muncul
    sys_msg = f"""
    Kamu adalah Frontend Developer expert. Tugasmu render laporan dalam **HTML MURNI**.
    
    Analisis: {analysis_content}
    Data JSON: {search_data_context}

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
    return {"final_answer": response.content, "messages": [response], "steps_log": ["✍️ Menyusun laporan akhir..."]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "analyst"

# Graph Setup
workflow = StateGraph(AgentState)

# Tambahkan node
workflow.add_node("vision", vision_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)

# Alur: START -> vision -> researcher -> ...
workflow.add_edge(START, "vision")
workflow.add_edge("vision", "researcher") 
workflow.add_conditional_edges("researcher", should_continue, {"tools": "tools", "analyst": "analyst"})
workflow.add_edge("tools", "analyst")
workflow.add_edge("analyst", "writer")
workflow.add_edge("writer", END)

graph = workflow.compile()

# Endpoint baru menerima Form Data (teks dan/atau file)
@app.post("/analyze")
async def analyze_claim(
    question: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    try:
        image_base64 = None
        initial_message = ""

        # Proses Gambar jika ada
        if image:
            print(f"Menerima gambar: {image.filename}")
            contents = await image.read()
            image_base64 = base64.b64encode(contents).decode('utf-8')
            initial_message = "Analisis gambar yang diupload."
        
        # Proses Teks jika ada
        elif question:
             initial_message = question
        
        else:
             raise HTTPException(status_code=400, detail= "Harus mengirim teks atau gambar.")

        inputs = {
            "messages": [HumanMessage(content=initial_message)], 
            "image_data": image_base64, 
            "steps_log": []
        }

        result = await graph.ainvoke(inputs)
        
        return {
            "status": "success",
            "logs": result.get("steps_log", []),
            "final_answer": result.get("final_answer", "<p>Gagal memproses data.</p>")
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)