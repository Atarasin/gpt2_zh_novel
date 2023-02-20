from transformers import (
    BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
)
import gradio as gr


tokenizer = BertTokenizer.from_pretrained("pretrained")
model = GPT2LMHeadModel.from_pretrained("checkpoints/jinyong-52000")
text_generator = TextGenerationPipeline(model, tokenizer)

def novel_generate(prompt: str, max_length: int):
    return text_generator(prompt, max_length=max_length, do_sample=True)[0]['generated_text']

if __name__ == "__main__":
    demo = gr.Interface(
        fn=novel_generate,
        inputs=['text', gr.Slider(1, 1000)],
        outputs=['text']
    )
    demo.launch()