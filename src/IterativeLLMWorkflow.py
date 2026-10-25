from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from src.TopicState import TopicState
from typing import Literal, Annotated
from pydantic import BaseModel, Field
import operator
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("iterative_llm_workflow")

load_dotenv()


class FeedBackSchema(BaseModel):
    feedback: str
    score: int
    decision: Literal["optimise", "approve"] = Field(
        ..., description="The decision based on the feedback")


model = ChatOpenAI()
structuredFeedbackModel = model.with_structured_output(FeedBackSchema)


def get_draft(state: TopicState):
    topic = state.topic
    logger.info("Generating draft for topic: %s", topic)
    prompt = f"create a draft with some good precise content for the topic: {topic}"
    response = model.invoke(prompt).content
    logger.info("Draft generated for topic: %s", response)
    return {"draft": response}


def get_feedback(state: TopicState):
    draft = state.draft
    logger.info("Getting feedback for draft: %s", draft)
    prompt = (
        f"You are a strict, critical editor reviewing a draft on the topic: '{state.topic}'.\n"
        f"Draft:\n{draft}\n\n"
        f"Score the draft from 1 to 10 on clarity, depth, structure and accuracy. "
        f"A score of {state.quality_threshold} or higher should only be given if the draft is "
        f"genuinely polished and needs no further changes - do not be lenient or round up. "
        f"This is iteration {state.iteration + 1} of up to {state.max_iteration}, so unless the draft "
        f"is already excellent, favor giving concrete, actionable feedback for improvement over approving it. "
        f"Set decision to 'approve' only if the score meets or exceeds {state.quality_threshold} and no "
        f"meaningful improvements remain; otherwise set decision to 'optimise' and explain exactly what to fix."
    )
    response: FeedBackSchema = structuredFeedbackModel.invoke(prompt)
    logger.info("Feedback for the draft : %s", response)
    next_iteration = state.iteration+1
    history = {
        "iteration": next_iteration,
        "draft": draft,
        "score": response.score,
        "feeddback": response.feedback,
        "decision": response.decision
    }
    return {
        "history": [history],
        "final_status": response.decision,
        "iteration": next_iteration,
        "draft": draft,
        "score": response.score,
        "feedback": response.feedback
    }


def route_after_feedback(state: TopicState) -> str:
    if state.iteration >= state.max_iteration:
        return "approve"
    if state.iteration < state.min_iteration:
        return "optimise"
    if state.final_status == "approve":
        return "approve"
    elif state.score >= state.quality_threshold:
        return "approve"
    else:
        return "optimise"


def do_optimise(state: TopicState):
    draft = state.draft
    topic = state.topic
    feedback = state.feedback
    logger.info("Optimise the draft for topic: %s", topic)
    logger.info("Optimise the draft : %s", draft)
    prompt = (
        f"Earlier draft did not meet the quality standards for the topic: {topic}.\n"
        f"Here is the draft:\n{draft}\n\n"
        f"Here is the reviewer's feedback that must be addressed:\n{feedback}\n\n"
        f"Rewrite and improve the draft to fully address this feedback and meet the quality standards."
    )
    response: str = model.invoke(prompt).content
    # iteration count is tracked solely in get_feedback (one feedback round = one iteration);
    # incrementing it here too would burn the iteration budget twice per loop.
    return {
        "draft": response
    }


def do_approve(state: TopicState):
    draft = state.draft
    logger.info("final approved draft: %s", draft)
    prompt = f"Here is the final approved draft : {draft} for the topic : {state.topic}. generate a summary of response "
    response: str = model.invoke(prompt).content

    return {"summary": response, "final_status": "approve"}


graph = StateGraph(TopicState)

graph.add_node("get_draft", get_draft)
graph.add_node("get_feedback", get_feedback)
graph.add_node("do_optimise", do_optimise)
graph.add_node("do_approve", do_approve)

graph.add_edge(START, "get_draft")
graph.add_edge("get_draft", "get_feedback")
graph.add_conditional_edges(
    "get_feedback",
    route_after_feedback,
    {
        "approve": "do_approve",
        "optimise": "do_optimise"
    }
)
graph.add_edge("do_optimise", "get_feedback")
graph.add_edge("do_approve",END)

iterativeWorkFlow = graph.compile()
