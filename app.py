import time
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_path = "emozilla/LLongMA-2-7b-storysummarizer"
tokenizer  = AutoTokenizer.from_pretrained(model_path)
model      = AutoModelForCausalLM.from_pretrained(model_path,
                                                  trust_remote_code=True,
                                                  # use_flash_attention=True,
                                                  device_map="auto")

def summary(
    text,
    task,
    temperature=0.8,
    repetition_penalty=1.1,
):
    start_time = time.time()
    pipe       = pipeline(model=model, tokenizer=tokenizer)

    sentence   = text + f'\n ### {task}:'
    max_length = len(text)
    
    outputs    = pipe(
        sentence,temperature=temperature,
        repetition_penalty=repetition_penalty,
        do_sample=True,
    )
    end_time    = time.time()
    total_time  = end_time - start_time 
    return outputs, total_time

def fn(
    text,
    task,
    temperature=0.8,
    repetition_penalty=1.1,
):
    result, tt = summary(text, temperature, repetition_penalty)
    return result, f"{tt:.4f}s"

demo = gr.Interface(
    fn=fn,
    inputs=[
        gr.Textbox(lines=3, placeholder='Enter Text To Summarise',label="Content", info="Enter the text content you wish to summarize"),
        gr.Dropdown(["SUMMARY", "ANALYSIS"], label="Task", info="Choose task type"),
        gr.Slider(minimum=1.0, maximum=15.00, step=0.5, value=1.1, label='Repetition Penalty', info='This parameter controls how much the model penalizes itself for generating repeated words or phrases. A higher value will result in more unique paraphrases, but may also result in less accurate paraphrases.'),
        gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.8, label='Temperature', info='Adjust temperature of the generation'),
        ],
    outputs=[
        gr.Text(label="Summary"),
        gr.Text(label="Inference time in seconds")
    ],
)

demo.launch()
