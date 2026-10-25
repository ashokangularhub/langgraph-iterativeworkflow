from fastapi import FastAPI
from fastapi.responses import Response
from src.IterativeLLMWorkflow import iterativeWorkFlow
from src.TopicState import TopicState
from pydantic import BaseModel
import uvicorn


api = FastAPI(title="Iterative LLM Workflow API",
              description="API for managing iterative LLM workflows", version="1.0.0")


class TopicRequest(BaseModel):
    topic: str
    max_iteration: int = 3
    quality_threshold: int = 8
    min_iteration: int = 2


class TopicResponse(BaseModel):
    draft: str
    summary: str
    final_status: str
    score: int
    history: list[dict]
    topic: str


@api.get("/topic/get_graph")
async def get_graph():
    png_bytes = iterativeWorkFlow.get_graph().draw_mermaid_png()
    return Response(content=png_bytes, media_type="image/png")


@api.post("/topic/finaldraft", response_model=TopicResponse)
async def final_draft(request: TopicRequest):
    topic = request.topic
    max_iteration = request.max_iteration
    quality_threshold = request.quality_threshold
    min_iteration = request.min_iteration

    initial_state = TopicState(topic=topic, max_iteration=max_iteration, quality_threshold=quality_threshold,
                               min_iteration=min_iteration
                               )
    final_state = iterativeWorkFlow.invoke(initial_state)
    final_state = TopicState(**final_state)
    response = TopicResponse(
        draft=final_state.draft,
        summary=final_state.summary,
        final_status=final_state.final_status,
        score=final_state.score,
        history=final_state.history,
        topic=final_state.topic
    )
    return response

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8000)
