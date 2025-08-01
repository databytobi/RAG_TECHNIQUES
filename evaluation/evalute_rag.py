"""
RAG Evaluation Script

This script evaluates the performance of a Retrieval-Augmented Generation (RAG) system
using various metrics from the deepeval library.

Dependencies:
- deepeval
- langchain_core
- json

Custom modules:
- helper_functions (for RAG-specific operations)
"""

import json
from typing import List, Tuple, Dict, Any
import os
import sys

from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.gemini_model import GeminiModel

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from custom_gemini_wrapper import GeminiChatWrapper  # 🟢 Your wrapper

# Use Gemini wrapper for LLM
llm: BaseChatModel = GeminiChatWrapper(model="gemini-2.0-flash", temperature=0)

# Path setup for helper functions
current_dir = os.path.dirname(os.path.abspath(_file_))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from helper_functions import (
    create_question_answer_from_context_chain,
    answer_question_from_context,
    retrieve_context_per_question
)

def create_deep_eval_test_cases(
    questions: List[str],
    gt_answers: List[str],
    generated_answers: List[str],
    retrieved_documents: List[str]
) -> List[LLMTestCase]:
    return [
        LLMTestCase(
            input=question,
            expected_output=gt_answer,
            actual_output=generated_answer,
            retrieval_context=retrieved_document
        )
        for question, gt_answer, generated_answer, retrieved_document in zip(
            questions, gt_answers, generated_answers, retrieved_documents
        )
    ]

# Define evaluation metrics
correctness_metric = GEval(
    name="Correctness",
    model="gemini-2.0-flash",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model="gemini-2.0-flash",
    include_reason=False
)

relevance_metric = ContextualRelevancyMetric(
    threshold=1,
    model="gemini-2.0-flash",
    include_reason=True
)

def evaluate_rag(retriever, num_questions: int = 5) -> Dict[str, Any]:
    """
    Evaluates a RAG system using predefined test questions and metrics.
    """
    eval_prompt = PromptTemplate.from_template("""
    Evaluate the following retrieval results for the question.
    
    Question: {question}
    Retrieved Context: {context}
    
    Rate on a scale of 1-5 (5 being best) for:
    1. Relevance: How relevant is the retrieved information to the question?
    2. Completeness: Does the context contain all necessary information?
    3. Conciseness: Is the retrieved context focused and free of irrelevant information?
    
    Provide ratings in JSON format:
    """)

    eval_chain = eval_prompt | llm | StrOutputParser()

    # Generate test questions
    question_gen_prompt = PromptTemplate.from_template(
        "Generate {num_questions} diverse test questions about climate change:"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()

    questions = question_chain.invoke({"num_questions": num_questions}).split("\n")

    results = []
    for question in questions:
        context = retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in context])

        eval_result = eval_chain.invoke({
            "question": question,
            "context": context_text
        })
        results.append(eval_result)

    return {
        "questions": questions,
        "results": results,
        "average_scores": calculate_average_scores(results)
    }

def calculate_average_scores(results: List[Dict]) -> Dict[str, float]:
    # Stub: fill in based on your expected output format (likely a list of JSON strings)
    pass

if _name_ == "_main_":
    # Example call: evaluate_rag(your_retriever)
    pass
