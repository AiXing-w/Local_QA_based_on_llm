import gradio as gr
from utils.load_files import load_Docs
from utils.model_process import *
import argparse
import json


def getParse():
    with open('config.json', 'r') as file:
        data = json.load(file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint_url', default=data["endpoint_url"])
    parser.add_argument("--embedding_model", default=data["embedding_model"])
    parser.add_argument('--llm', default=data['llm'])
    args = parser.parse_args()
    return args


color_map = {
    "harmful": "crimson",
    "neutral": "gray",
    "beneficial": "green",
}


def html_src(harm_level):
    return f"""
<div style="display: flex; gap: 5px;">
  <div style="background-color: {color_map[harm_level]}; padding: 2px; border-radius: 5px;">
  {harm_level}
  </div>
</div>
"""


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(diaghistory, message):
    if len(message["files"]) > 0:
        directory = message["files"][-1]
        diaghistory = []
        documents = load_Docs(directory)
        global db
        db = store_chroma(documents, embeddings, directory)

    for x in message["files"]:
        diaghistory.append(((x,), None))
    if message["text"] is not None:
        diaghistory.append((message["text"], None))
    return diaghistory, gr.MultimodalTextbox(value=None, interactive=False)


def bot(diaghistory, response_type):
    query = diaghistory[-1][0]
    if query != '':
        global db
        info = augment_prompt(db, query)
        prompt = "基于下面给出的资料，回答问题，如果资料不足或者回答不了就回答不知道。下面是资料：\n"
        prompt += info
        prompt += f"下面是问题：{query}"
        global history
        ans, _ = model.chat(tokenizer, prompt, [])
        response = f"文中相关片段:\n{info}\n\n" + ans
    else:
        response = "请描述您的问题"
    diaghistory[-1][1] = response
    return diaghistory


with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        scale=1,
    )
    response_type = gr.Radio(
        [
            "pdf",
            "doc(x)",
            "text",
        ],
        value="text",
        label="Response Type",
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        placeholder="Enter message or upload file...",
        show_label=False,
    )

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )
    bot_msg = chat_msg.then(
        bot, [chatbot, response_type], chatbot, api_name="bot_response"
    )
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    chatbot.like(print_like_dislike, None, None)

demo.queue()
if __name__ == "__main__":
    args = getParse()
    endpoint_url = args.endpoint_url
    embedding_model_params = args.embedding_model
    llm_path = args.llm['model_path'].replace("/", "\\")
    embeddings = load_embedding_mode(embedding_model_params)
    model, tokenizer = load_llm_mode(llm_path)
    history = []
    db = None
    demo.launch()
