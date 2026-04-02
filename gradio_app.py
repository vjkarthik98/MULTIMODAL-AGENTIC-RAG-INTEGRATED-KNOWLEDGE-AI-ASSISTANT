import gradio as gr
import requests

QUERY_API = "http://127.0.0.1:8000/rag/query"
UPLOAD_API = "http://127.0.0.1:8000/rag/upload/file"


# BACKEND STEM

def chat_response(message, history):
    history = history or []

    try:
        response = requests.post(
            QUERY_API,
            json={
                "query": message,
                "session_id": "gradio_user"
            },
        )

        data = response.json()
        print("DEBUG RESPONSE:", data)

        answer = data.get("answer", "No answer")

        # Append to chat history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
       
        return history

    except Exception as e:
        history.append({"role": "assitant", "content": f"Error: {str(e)}"})
        return history

# UPLOAD STATUS 
def upload_status_fn(file):
        if file is None:
            return "❌ No file selected"
        return " ⏳ Processing... Please wait(video/audio may take time)"
    
# ACTUAL UPLOAD 
def process_upload(file):
    try:
        if file is None:
            return "❌No file selected"
        
        files = {"file": open(file.name, "rb")}

        response = requests.post(UPLOAD_API, files=files)
        data = response.json()

        return f"✅ Uploaded: {data.get('filename', 'file')}"
    
    except Exception as e:
        return f"❌ upload failed: {str(e)}"
    
# UI
with gr.Blocks(theme=gr.themes.Soft()) as app:

    chats = gr.State([[]]) # multiple chats
    current_chat_index = gr.State(0)

    with gr.Row():
        
        # SIDEBAR
        with gr.Column(scale=2):
            gr.Markdown(" ## Chats")

            new_chat_btn = gr.Button("➕ New Chat")

            chat_list = gr.Radio(
                choices=["Chat 1"],
                value="Chat 1",
                label="History"
            )

    # MAIN CHAT
        with gr.Column(scale=8):
            chatbot = gr.Chatbot(label="Chat", value = [], render_markdown = True)

            with gr.Row():
                msg = gr.Textbox(
                    placeholder = "Ask something...",
                    show_label = False,
                    scale = 8
                )
                send_btn = gr.Button("➤", scale=1)

        # Upload section
        with gr.Row():
            file_input = gr.File(label="Upload file (PDF/Image/Audio/Video)")
            upload_btn = gr.Button("Upload")
            upload_status = gr.Textbox(label="Status")

    # CHAT MANAGEMENT

    def handle_new_chat(chats):
        chats.append([])
        names = [f"Chat {i+1}" for i in range(len(chats))]
        new_index = len(chats) - 1
        
        return (
            chats, 
            chats[new_index],
            gr.update(choices=names, value=names[new_index]),
            new_index
        )
    
    new_chat_btn.click(
        handle_new_chat,
        inputs=[chats],
        outputs=[chats, chatbot, chat_list, current_chat_index]
    )

    def handle_switch(chat_name, chats):
        index = int(chat_name.split()[-1]) - 1
        return chats[index], index
        
    chat_list.change(
        handle_switch,
        inputs=[chat_list, chats],
        outputs=[chatbot, current_chat_index]
    )

    msg.submit(
        chat_response,
        [msg, chatbot],
        chatbot
    )  

    send_btn.click(
        chat_response,
        [msg, chatbot],
        chatbot
    )

    # Upload events (2-step UX)
    upload_btn.click(
        upload_status_fn, file_input, upload_status
    ).then(
        process_upload, file_input, upload_status
    )
   
app.launch()