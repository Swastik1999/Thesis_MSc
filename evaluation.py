import os
import traceback
from typing import Dict

import numpy as np
import pandas as pd
from config import GlobalConfig
from datasets import Dataset
from pipeline import TimingTracker
from ragas import evaluate
from ragas.metrics import AnswerAccuracy, ContextRelevance, ResponseGroundedness
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama


def evaluate_rag_results(
    df: pd.DataFrame,
    trial_number: int,
    eval_model: str,
    temperature: float = 0.0,
) -> Dict[str, float]:
    timer = TimingTracker()
    timer.start("evaluation")

    valid_answers = sum(
        1 for a in df["answer"] if a and str(a).strip() and str(a) != ""
    )
    valid_contexts = sum(1 for c in df["contexts"] if c and len(c) > 0)

    if valid_answers == 0 or valid_contexts == 0:
        return {
            "nv_accuracy": 0.0,
            "nv_response_groundedness": 0.0,
            "nv_context_relevance": 0.0,
            "evaluation_time": 0.0,
        }

    try:
        llm = ChatOllama(
            model=eval_model,
            temperature=temperature,
            num_gpus=GlobalConfig.QA_WORKERS,
            keep_alive=False,
        )
    except Exception as e:
        print(f"Failed to initialize evaluation LLM: {e}")
        return {
            "nv_accuracy": 0.0,
            "nv_response_groundedness": 0.0,
            "nv_context_relevance": 0.0,
            "evaluation_time": 0.0,
        }

    df["contexts"] = df["contexts"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    eval_df = df[["question", "contexts", "answer", "short_answers"]].copy()
    eval_df.columns = [
        "user_input",
        "retrieved_contexts",
        "response",
        "reference",
    ]
    eval_df["user_input"] = eval_df["user_input"].astype(str)
    eval_df["response"] = eval_df["response"].apply(
        lambda x: str(x) if x else "No answer"
    )
    eval_df["reference"] = eval_df["reference"].astype(str)

    eval_df = eval_df[
        (eval_df["response"] != "")
        & (eval_df["response"] != "No answer")
        & (eval_df["retrieved_contexts"].apply(lambda x: len(x) > 0))
    ].reset_index(drop=True)

    if len(eval_df) == 0:
        return {
            "nv_accuracy": 0.0,
            "nv_response_groundedness": 0.0,
            "nv_context_relevance": 0.0,
            "evaluation_time": 0.0,
        }

    try:
        dataset = Dataset.from_pandas(
            eval_df[
                ["user_input", "retrieved_contexts", "response", "reference"]
            ]
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return {
            "nv_accuracy": 0.0,
            "nv_response_groundedness": 0.0,
            "nv_context_relevance": 0.0,
            "evaluation_time": 0.0,
        }

    try:
        score = evaluate(
            dataset,
            metrics=[
                AnswerAccuracy(),
                ContextRelevance(),
                ResponseGroundedness(),
            ],
            llm=llm,
            run_config=RunConfig(
                timeout=100000,
                max_retries=20,
                max_wait=50,
                log_tenacity=False,
                max_workers=GlobalConfig.QA_WORKERS,
            ),
        )
        per_sample_df = score.to_pandas()
        per_sample_df.to_csv(
            os.path.join(
                GlobalConfig.BASE_OUTPUT_DIR,
                f"trial_{trial_number}",
                "evaluation_per_sample.csv",
            ),
            index=False,
        )
        timer.end("evaluation")
        return {
            "nv_accuracy": float(np.nanmean(score["nv_accuracy"])),
            "nv_response_groundedness": float(
                np.nanmean(score["nv_response_groundedness"])
            ),
            "nv_context_relevance": float(
                np.nanmean(score["nv_context_relevance"])
            ),
            "evaluation_time": timer.get_duration("evaluation"),
        }

    except Exception as e:
        print(f"Evaluation error: {e}")
        traceback.print_exc()
        return {
            "nv_accuracy": 0.0,
            "nv_response_groundedness": 0.0,
            "nv_context_relevance": 0.0,
            "evaluation_time": 0.0,
        }