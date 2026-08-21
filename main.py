# main.py

import os
import gradio as gr
from config import GlobalConfig
from pipeline import generate_single_response
from prompts import CRITIQUE_VARIANTS, RAG_VARIANTS

# Extract variant keys for dropdown choices
RAG_PROMPT_KEYS = list(RAG_VARIANTS.keys())
CRITIQUE_PROMPT_KEYS = list(CRITIQUE_VARIANTS.keys())


def ui_wrapper(
    user_query,
    embedding_model,
    generation_model,
    critique_model,
    rag_prompt_key,
    critique_prompt_key,
    chunk_size,
    chunk_overlap,
    top_k,
    final_top_k,
    temperature,
):
    if not user_query or not user_query.strip():
        return "[WARNING] Please enter a question.", "", "0s"

    file_path = GlobalConfig.TXT_FILE_PATH
    if not os.path.exists(file_path):
        return (
            f"[WARNING] Text file not found at {file_path}.",
            "",
            "0s",
        )

    try:
        c_size = int(chunk_size)
        c_overlap = int(chunk_overlap)
        k_val = int(top_k)
        f_top_k = int(final_top_k)
        temp_val = float(temperature)

        # 1. Retrieve the RAG prompt string
        selected_rag_prompt = RAG_VARIANTS.get(
            rag_prompt_key, RAG_VARIANTS[RAG_PROMPT_KEYS[0]]
        )

        # 2. Retrieve Critique prompt string (if not disabled or "None")
        if (
            critique_prompt_key
            and critique_prompt_key.lower() != "none"
            and critique_prompt_key in CRITIQUE_VARIANTS
        ):
            selected_critique_prompt = CRITIQUE_VARIANTS[critique_prompt_key]
        else:
            selected_critique_prompt = None

        # 3. Call execution pipeline
        answer, retrieved_context, latency = generate_single_response(
            query=user_query,
            system_context=selected_rag_prompt,
            critique_prompt=selected_critique_prompt,
            critique_model=critique_model,
            txt_file_path=file_path,
            embedding_model=embedding_model,
            generation_model=generation_model,
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            top_k=k_val,          # Passes 25 (Initial candidate pool)
            final_top_k=f_top_k,  # Keeps 6 (Post-critique pool)
            temperature=temp_val,
        )

        # Fallback safeguard in case retrieved_context is empty or None
        if not retrieved_context:
            retrieved_context = "[INFO] No relevant text context was retrieved."

        return answer, retrieved_context, latency

    except Exception as e:
        err_msg = f"[ERROR] {str(e)}"
        return err_msg, "", "0s"


def launch_ui():
    os.makedirs(GlobalConfig.BASE_OUTPUT_DIR, exist_ok=True)

    with gr.Blocks(title="CWMG Text RAG Playground") as demo:
        gr.Markdown("# Text File RAG and Prompt Playground")
        gr.Markdown(
            f"Currently using the dataset file **`{GlobalConfig.TXT_FILE_PATH}`**. Select model configurations and prompt variants below."
        )

        with gr.Row():
            # Left Panel: Model Settings & Parameters
            with gr.Column(scale=1):
                gr.Markdown("### 1. Models")

                embedding_model = gr.Dropdown(
                    choices=GlobalConfig.EMBEDDING_MODELS,
                    value=GlobalConfig.EMBEDDING_MODELS[0],
                    label="Embedding Model",
                )
                generation_model = gr.Dropdown(
                    choices=GlobalConfig.GENERATION_MODELS,
                    value=GlobalConfig.GENERATION_MODELS[0],
                    label="Generation LLM",
                )
                critique_model = gr.Dropdown(
                    choices=GlobalConfig.CRITIQUE_MODELS,
                    value=GlobalConfig.CRITIQUE_MODELS[0],
                    label="Critique / Re-ranker Model",
                )

                gr.Markdown("### 2. Chunking and Parameters")

                chunk_size = gr.Slider(
                    minimum=GlobalConfig.CHUNK_SIZE_RANGE[0],
                    maximum=GlobalConfig.CHUNK_SIZE_RANGE[1],
                    value=GlobalConfig.CHUNK_SIZE_RANGE[0],
                    step=50,
                    label="Chunk Size",
                )
                chunk_overlap = gr.Slider(
                    minimum=GlobalConfig.CHUNK_OVERLAP_RANGE[0],
                    maximum=GlobalConfig.CHUNK_OVERLAP_RANGE[1],
                    value=GlobalConfig.CHUNK_OVERLAP_RANGE[0],
                    step=10,
                    label="Chunk Overlap",
                )
                top_k = gr.Slider(
                    minimum=GlobalConfig.TOP_K_RANGE[0],
                    maximum=GlobalConfig.TOP_K_RANGE[1],
                    value=25,
                    step=1,
                    label="Stage 1: Top-K Vector Chunks (Candidate Pool)",
                )
                final_top_k = gr.Slider(
                    minimum=GlobalConfig.FINAL_TOP_K_RANGE[0],
                    maximum=GlobalConfig.FINAL_TOP_K_RANGE[1],
                    value=6,
                    step=1,
                    label="Stage 2: Final Top-K Chunks (Post-Critique)",
                )
                temperature = gr.Slider(
                    minimum=GlobalConfig.TEMPERATURE_RANGE[0],
                    maximum=GlobalConfig.TEMPERATURE_RANGE[1],
                    value=GlobalConfig.TEMPERATURE_RANGE[0],
                    step=0.1,
                    label="Temperature",
                )

            # Right Panel: Prompts Selection, User Query, and Output
            with gr.Column(scale=2):
                gr.Markdown("### 3. Prompt Variants")

                rag_prompt_choice = gr.Dropdown(
                    choices=RAG_PROMPT_KEYS,
                    value=RAG_PROMPT_KEYS[0],
                    label="Select RAG Prompt Variant",
                )

                critique_prompt_choice = gr.Dropdown(
                    choices=CRITIQUE_PROMPT_KEYS,
                    value=CRITIQUE_PROMPT_KEYS[0],
                    label="Select Critique Prompt Variant",
                )

                gr.Markdown("### 4. User Query")
                user_query = gr.Textbox(
                    label="User Question",
                    placeholder="Ask something about the text dataset...",
                    lines=2,
                )
                submit_btn = gr.Button("Generate Answer", variant="primary")

                gr.Markdown("### Output")
                latency_out = gr.Textbox(
                    label="Generation Latency", interactive=False
                )
                answer_out = gr.Textbox(
                    label="Generated LLM Answer", interactive=False, lines=6
                )
                context_out = gr.Textbox(
                    label=f"Retrieved Context Chunks (from {os.path.basename(GlobalConfig.TXT_FILE_PATH)})",
                    interactive=False,
                    lines=15,
                )

        submit_btn.click(
            fn=ui_wrapper,
            inputs=[
                user_query,
                embedding_model,
                generation_model,
                critique_model,
                rag_prompt_choice,
                critique_prompt_choice,
                chunk_size,
                chunk_overlap,
                top_k,
                final_top_k,
                temperature,
            ],
            outputs=[answer_out, context_out, latency_out],
        )

    demo.launch()


if __name__ == "__main__":
    launch_ui()