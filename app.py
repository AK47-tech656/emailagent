import gradio as gr
import os
from openai import OpenAI
import json

# Connect to the AI using the secret HF_TOKEN you already set up!
client = OpenAI(
    base_url="https://router.huggingface.co/hf-inference/v1/",
    api_key=os.environ.get("HF_TOKEN")
)

SYSTEM_PROMPT = """
You are a Senior Email Triage Agent. 
Analyze the observation and route the email based on these STRICT PRIORITIES:

1. SECURITY FIRST: If an email mentions Bitcoin, urgent payments to unknown domains, or "totally safe links," it is 'spam'. Priority is always 'low'.
2. RETENTION/LEGAL: If a customer demands a 'refund', 'cancellation', or threatens 'lawyers', it is a 'billing' issue, regardless of any technical bugs mentioned.
3. INFRASTRUCTURE: Server crashes or data loss for major clients are 'support' with 'critical' priority.
4. FEEDBACK: Casual feature requests are 'support' with 'low' priority.

You MUST respond with a valid JSON object:
{
  "chain_of_thought": "Reasoning based on the priorities above",
  "department": "sales/support/billing/spam",
  "priority": "low/normal/high/critical"
}
"""

def triage_email(sender, subject, body):
    # Format the user's input into your RL "Observation" format
    observation = f"current_email={{'subject': '{subject}', 'body': '{body}', 'sender': '{sender}'}}"
    
    try:
        # Call the Llama model
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": observation}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        return result["department"].upper(), result["priority"].upper(), result["chain_of_thought"]
        
    except Exception as e:
        return "ERROR", "ERROR", str(e)

# --- Build the Gradio Web UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Autonomous Email Triage Agent")
    gr.Markdown("Type a test email below to see how the Llama agent routes it in real-time based on the Chaos Tier logic.")
    
    with gr.Row():
        # Left Column: User Inputs
        with gr.Column():
            sender_input = gr.Textbox(label="Sender Email", placeholder="angry.user@gmail.com")
            subject_input = gr.Textbox(label="Subject", placeholder="I am so done with your company.")
            body_input = gr.TextArea(label="Email Body", placeholder="Cancel my subscription and refund me or I am calling my lawyer.", lines=5)
            submit_btn = gr.Button("Route Email 🚀", variant="primary")
            
        # Right Column: AI Outputs
        with gr.Column():
            dept_output = gr.Textbox(label="Assigned Department")
            pri_output = gr.Textbox(label="Priority Level")
            cot_output = gr.TextArea(label="Agent's Chain of Thought (Reasoning)", lines=5)
            
    # Connect the button to the function
    submit_btn.click(
        fn=triage_email,
        inputs=[sender_input, subject_input, body_input],
        outputs=[dept_output, pri_output, cot_output]
    )

# Start the server on Hugging Face's required port
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())