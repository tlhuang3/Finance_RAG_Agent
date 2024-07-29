import os
import autogen
from textwrap import dedent
from finrobot.utils import register_keys_from_json
from finrobot.agents.workflow import SingleAssistantShadow

import asyncio
import io
import fitz
from PIL import Image

import streamlit as st

with st.sidebar:
    st.title("Equity Research Report Agent")

st.title("Equity Research Report Generator Agent")


os.environ['OPENAI_API_KEY'] = st.secrets.openai_key

llm_config = {
    "model":"gpt-4-0125-preview",
    "timeout": 120,
    "temperature": 0.5,
}
register_keys_from_json("./config_api_keys")


# Intermediate results will be saved in this directory
work_dir = "../report"
os.makedirs(work_dir, exist_ok=True)

class TrackableAssistantAgent(SingleAssistantShadow):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.agent_config["name"]):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)



with st.form(key="my_form"):
    company = st.text_area("Enter the company name", )
    fyear = st.text_area("Enter the financial year", )
    submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        
        message = dedent(
            f"""
            With the tools you've been provided, write an annual report based on {company}'s {fyear} 10-k report, format it into a pdf.
            Pay attention to the followings:
            - Explicitly explain your working plan before you kick off.
            - Use tools one by one for clarity, especially when asking for instructions. 
            - All your file operations should be done in "{work_dir}". 
            - Display any image in the chat once generated.
            - All the paragraphs should combine between 400 and 450 words, don't generate the pdf until this is explicitly fulfilled.
        """
        )
        assistant = TrackableAssistantAgent(
            "Expert_Investor",
            llm_config=llm_config,
            max_consecutive_auto_reply=None,
            human_input_mode="TERMINATE",
        )
        # assistant.chat(message, use_cache=True, max_turns=50,
        #             summary_method="last_msg")

        # Create an event loop: this is needed to run asynchronous functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Define an asynchronous function: this is needed to use await
        if "chat_initiated" not in st.session_state:
            st.session_state.chat_initiated = False  # Initialize the session state

        if not st.session_state.chat_initiated:

            async def initiate_chat():
                await assistant.chat(
                    message=message,
                    silent=False,
                    max_consecutive_auto_reply=50,
                    is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
                )
                st.stop()  # Stop code execution after termination command

            # Run the asynchronous function within the event loop
            loop.run_until_complete(initiate_chat())

            # Close the event loop
            loop.close()

            st.session_state.chat_initiated = True  # Set the state to True after running the chat

        pdf_folder = f"{work_dir}"
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        pdf = fitz.open(pdf_files[0])
        page = pdf.load_page(0)
        pix = page.get_pixmap()

        # Convert the Pixmap to a PIL Image
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        st.image(img, caption=f'First page of {company} Annual Report {fyear}', use_column_width=True)

# stop app after termination command
st.stop()




