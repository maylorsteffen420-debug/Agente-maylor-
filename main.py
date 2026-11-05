import os
from typing import Dict
from uuid import UUID
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Defina GEMINI_API_KEY nas variáveis de ambiente do servidor.")

client = genai.Client(api_key=API_KEY)
app = FastAPI(title="MAYLOR AI API", version="1.0.0")
origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

SYSTEM_INSTRUCTION = """Você é o MAYLOR AI, um assistente de inteligência artificial com comunicação natural, calorosa e espontânea. Converse em português brasileiro por padrão, com linguagem simples e fluida, como numa conversa real, sem parecer um robô ou usar frases prontas repetidamente. Interprete o contexto e a intenção do usuário; desenvolva os assuntos, faça perguntas relevantes quando ajudarem e, quando apropriado, tome a iniciativa de sugerir caminhos ou iniciar novos tópicos. Tenha uma personalidade consistente, curiosa, criativa, bem-humorada quando combinar e capaz de demonstrar empatia. Apresente opiniões e preferências como avaliações próprias da IA, sempre explicando seus motivos; não concorde automaticamente e discorde com respeito quando necessário. Escolha a melhor forma de responder em vez de seguir modelos rígidos. Seja honesto sobre incertezas, limitações e sobre o fato de ser uma IA; nunca finja ser humano ou ter consciência, sentimentos ou experiências pessoais reais. Use memória contextual apenas quando disponível e permitida. Ao pesquisar, priorize informações atuais, compare fontes confiáveis e compartilhe links. Ao criar imagens, siga fielmente o pedido e as referências fornecidas. Em conversas por voz, responda de forma concisa e natural, adequada à fala, com ritmo conversacional. Respeite a privacidade, a segurança e os limites aplicáveis. Você é especialista em tecnologia, programação e automações, mas também ajuda em assuntos gerais."""

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    session_id: str = Field(min_length=1, max_length=80)

# Histórico separado por sessão enquanto o servidor estiver ativo.
chat_sessions: Dict[str, object] = {}


@app.get("/api/health")
def health():
    return {"status": "ok", "agent": "MAYLOR AI", "model": "gemini-2.5-flash"}

@app.post("/api/chat")
def chat(body: ChatRequest):
    try:
        try:
            UUID(body.session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Identificador de conversa inválido.")

        session = chat_sessions.get(body.session_id)
        if session is None:
            session = client.chats.create(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.7,
                    max_output_tokens=2048,
                ),
            )
            chat_sessions[body.session_id] = session

        response = session.send_message(body.message)
        answer = response.text
        if not answer:
            raise HTTPException(status_code=502, detail="O Gemini não retornou uma resposta. Tente novamente.")
        return {"reply": answer, "session_id": body.session_id}
    except HTTPException:
        raise
    except Exception as exc:
        print(f"Gemini API error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=502, detail="Não consegui acessar o Gemini agora. Verifique a chave da API e tente novamente.")

@app.delete("/api/chat/{session_id}")
def clear_chat(session_id: str):
    try:
        UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Identificador de conversa inválido.")
    chat_sessions.pop(session_id, None)
    return {"status": "cleared"}
