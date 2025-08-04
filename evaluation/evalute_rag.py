"""
RAG Evaluation Script

This script evaluates the performance of a Retrieval-Augmented Generation (RAG) system
using various metrics from the deepeval library.

Dependencies:
- deepeval
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
from deepeval.models import GeminiModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from helper_functions import (
    create_question_answer_from_context_chain,
    answer_question_from_context,
    retrieve_context_per_question
)

# Initialize Gemini LLM for DeepEval and LangChain
llm = GeminiModel(
    model_name="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# Define evaluation metrics
correctness_metric = GEval(
    name="Correctness",
    model=llm,
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
    model=llm,
    include_reason=False
)

relevance_metric = ContextualRelevancyMetric(
    threshold=1,
    model=llm,
    include_reason=True
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

def calculate_average_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
    total_scores = {"relevance": 0, "completeness": 0, "conciseness": 0}
    for result in results:
        try:
            scores = json.loads(result)
            total_scores["relevance"] += scores.get("Relevance", 0)
            total_scores["completeness"] += scores.get("Completeness", 0)
            total_scores["conciseness"] += scores.get("Conciseness", 0)
        except Exception as e:
            print("Error parsing result:", result)
            continue

    n = len(results)
    if n == 0:
        return {key: 0.0 for key in total_scores}
    return {key: val / n for key, val in total_scores.items()}

def evaluate_rag(retriever, num_questions: int = 5) -> Dict[str, Any]:
    """
    Evaluates a RAG system using retrieval scores and deep evaluation metrics.
    """
    # Step 1: Generate questions
    question_gen_prompt = PromptTemplate.from_template(
        "Generate {num_questions} diverse and challenging test questions about climate change:"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()
    raw_questions = question_chain.invoke({"num_questions": num_questions})
    questions = [q.strip("- ").strip() for q in raw_questions.split("\n") if q.strip()]

    gt_answers = []
    generated_answers = []
    retrieved_contexts = []

    for question in questions:
        retrieved_docs = retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in retrieved_docs])
        answer = answer_question_from_context(question, context_text, llm)
        gt_answer = "Answer the question based strictly on the provided context."

        retrieved_contexts.append(context_text)
        generated_answers.append(answer)
        gt_answers.append(gt_answer)

    # Step 2: Create test cases
    test_cases = create_deep_eval_test_cases(
        questions, gt_answers, generated_answers, retrieved_contexts
    )

    # Step 3: Run evaluation
    evaluate(test_cases, [correctness_metric, faithfulness_metric, relevance_metric])

    # Step 4: Custom scoring (optional)
    eval_prompt = PromptTemplate.from_template("""
    Evaluate the following retrieval results for the question.

    Question: {question}
    Retrieved Context: {context}

    Rate on a scale of 1-5 (5 being best) for:
    1. Relevance
    2. Completeness
    3. Conciseness

    Return ratings in this JSON format:
    {{ "Relevance": x, "Completeness": y, "Conciseness": z }}
    """)

    eval_chain = eval_prompt | llm | StrOutputParser()
    eval_results = []

    for question, context_text in zip(questions, retrieved_contexts):
        try:
            result = eval_chain.invoke({
                "question": question,
                "context": context_text
            })
            eval_results.append(result)
        except Exception as e:
            print("Error during custom scoring:", e)
            eval_results.append("{}")

    avg_scores = calculate_average_scores(eval_results)

    return {
        "questions": questions,
        "generated_answers": generated_answers,
        "retrieved_contexts": retrieved_contexts,
        "average_scores": avg_scores,
        "raw_custom_scores": eval_results
    }

if _name_ == "_main_":
    from helper_functions import chunks_query_retriever
    results = evaluate_rag(chunks_query_retriever)
    print(json.dumps(results, indent=2))

