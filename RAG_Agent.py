from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from autogen import ConversableAgent
from autogen import register_function
import os
import streamlit as st

import asyncio

os.environ['OPENAI_API_KEY'] = st.secrets.openai_key
with st.sidebar:
    st.title("Finance Chatbot Agent")

st.title("Finance Chatbot Agent")




# Load Sentence Transformer Embeddings
emb_fn = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

# Load SEC Filings and Earnings Call Databases
sec_filings_md_db = Chroma(persist_directory="./database/sec-filings-md-db_v2",
                                embedding_function=emb_fn,
                                collection_name="sec_filings_md_v2")


earnings_call_db = Chroma(persist_directory="./database/earnings-call-db_v2", embedding_function=emb_fn, collection_name="earnings_call_v2")

# Retrieve all entries for earnings call metadata (this may vary depending on your actual ChromaDB implementation)
entries = earnings_call_db.get(include=["metadatas"])

# Retrieve all speakers for each quarter in earnings call database
speakers_list_1 = []
speakers_list_2 = []
speakers_list_3 = []
speakers_list_4 = []

for meta in entries['metadatas']:
    for key, value in meta.items():
        if key == "quarter" and value == "Q1":
            speakers_list_1.append(meta['speaker'])
        elif key == "quarter" and value == "Q2":
            speakers_list_2.append(meta['speaker'])
        elif key == "quarter" and value == "Q3":
            speakers_list_3.append(meta['speaker'])
        elif key == "quarter" and value == "Q4":
            speakers_list_4.append(meta['speaker'])


quarter_speaker_dict = {
        "Q1":speakers_list_1,
        "Q2":speakers_list_2,
        "Q3":speakers_list_3,
        "Q4":speakers_list_4
    }

# Define the SEC Form Names
sec_form_names = ['10-K', '10-Q4', '10-Q3', '10-Q2', '10-Q1']


# Design Tools for Querying the Databases

global FROM_MARKDOWN
FROM_MARKDOWN = True


def query_database_earnings_call(
        question: str,
        quarter: str
    )->str:
        """This tool will query the earnings call transcripts database for a given question and quarter and it will retrieve
        the relevant text along from the earnings call and the speaker who addressed the relevant documents. This tool helps in answering questions
        from the earnings call transcripts.

        Args:
            question (str): _description_. Question to query the database for relevant documents.
            quarter (str): _description_. the financial quarter that is discussed in the question and possible options are Q1, Q2, Q3, Q4

        Returns:
            str: relevant text along from the earnings call and the speaker who addressed the relevant documents
        """
        assert quarter in ["Q1", "Q2", "Q3", "Q4"], "The quarter should be from Q1, Q2, Q3, Q4"

        req_speaker_list = []
        quarter_speaker_list = quarter_speaker_dict[quarter]

        for sl in quarter_speaker_list:
            if sl in question or sl.lower() in question:
                req_speaker_list.append(sl)
        if len(req_speaker_list) == 0:
            req_speaker_list = quarter_speaker_list

        relevant_docs = earnings_call_db.similarity_search(
            question,
            k=5,
            filter={
                "$and":[
                    {
                        "quarter":{"$eq":quarter}
                    },
                    {
                        "speaker":{"$in":req_speaker_list}
                    }
                ]
            }
        )

        speaker_releavnt_dict = {}
        for doc in relevant_docs:
            speaker = doc.metadata['speaker']
            speaker_text = doc.page_content
            if speaker not in speaker_releavnt_dict:
                speaker_releavnt_dict[speaker] = speaker_text
            else:
                speaker_releavnt_dict[speaker] += " "+speaker_text

        relevant_speaker_text = ""
        for speaker, text in speaker_releavnt_dict.items():
            relevant_speaker_text += speaker + ": "
            relevant_speaker_text += text + "\n\n"

        return relevant_speaker_text




def query_database_markdown_sec(
            question: str,
            sec_form_name: str
    )->str:
  assert sec_form_name in sec_form_names, f'The search form type should be in {sec_form_names}'

  relevant_docs = sec_filings_md_db.similarity_search(
      question,
      k=3,
      filter={
          "filing_type":{"$eq":sec_form_name}
      }
  )
  # print(relevant_docs)
  relevant_section_text = ""
  for relevant_text in relevant_docs:
      relevant_section_text += relevant_text.page_content + "\n\n"

  return relevant_section_text


def query_database_sec(
            question: str,
            sec_form_name: str
    )->str:
        """This tool will query the SEC Filings database for a given question and form name, and it will retrieve
        the relevant text along from the SEC filings and the section names. This tool helps in answering questions
        from the sec filings.

        Args:
            question (str): _description_. Question to query the database for relevant documents
            sec_form_name (str): _description_. SEC FORM NAME that the question is talking about. It can be 10-K for yearly data and 10-Q for quarterly data. For quarterly data, it can be 10-Q2 to represent Quarter 2 and similarly for other quarters.

        Returns:
            str: Relevant context for the question from the sec filings
        """
        if not FROM_MARKDOWN:
          return "No data available"
        elif FROM_MARKDOWN:
          return query_database_markdown_sec(question,sec_form_name)
        

# System Message for the Agent
sec_form_system_msg = ""
for sec_form in sec_form_names:
    if sec_form == "10-K":
        sec_form_system_msg+= "10-K for yearly data, "
    elif "10-Q" in sec_form:
        quarter = sec_form[-1]
        sec_form_system_msg+= f"{sec_form} for Q{quarter} data, "
sec_form_system_msg = sec_form_system_msg[:-2]

earnings_call_system_message = ", ".join(["Q1", "Q2", "Q3", "Q4"])

system_msg = f"""You are a helpful financial assistant and your task is to select the sec_filings or earnings_call to best answer the question.
You can use query_database_sec(question,sec_form) by passing question and relevant sec_form names like {sec_form_system_msg}
or you can use query_database_earnings_call(question,quarter) by passing question and relevant quarter names with possible values {earnings_call_system_message}. When you are ready to end the coversation, reply TERMINATE"""



class TrackableAssistantAgent(ConversableAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)


class TrackableUserProxyAgent(ConversableAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)




with st.container():

    # Streamlit input and button
    task = st.text_area("Enter your question here", value="What is the Net Revenue for AMD on March 30, 2024?")
    if st.button("Submit"):
        if task:
            # Create Proxies for the Agents
            llm_config = {"model":"gpt-3.5-turbo"}

            # Create an User Proxy Agent
            user_proxy = TrackableUserProxyAgent(
                name = "user",
                system_message=system_msg,
                code_execution_config=False,
                llm_config=llm_config,
                human_input_mode="NEVER",
                is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
            )

            tool_proxy = TrackableAssistantAgent(
                name="assistant",
                system_message="Analyze the response from user proxy and decide whether the suggested database is suitable "
                ". Answer in simple yes or no",
                llm_config=False,
                # is_termination_msg=lambda msg: "exit" in msg.get("content",""),
                default_auto_reply="Please select the right database.",
                human_input_mode="ALWAYS",
            )

            tools_dict = {
                    "sec":[query_database_sec,"Tool to query SEC filings database"],
                    "earnings_call": [query_database_earnings_call, "Tool to query earnings call transcripts database"],
                }

            # Register Functions
            for tool_name,tool in tools_dict.items():
                register_function(
                    tool[0],
                    caller=user_proxy,
                    executor=tool_proxy,
                    name = tool[0].__name__,
                    description=tool[1]
                )

            # Create an event loop: this is needed to run asynchronous functions
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Define an asynchronous function: this is needed to use await
            if "chat_initiated" not in st.session_state:
                st.session_state.chat_initiated = False  # Initialize the session state

            if not st.session_state.chat_initiated:

                async def initiate_chat():
                    await user_proxy.a_initiate_chat(
                        recipient=tool_proxy,
                        message=task,
                        max_consecutive_auto_reply=5,
                        is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
                    )
                    st.stop()  # Stop code execution after termination command

                # Run the asynchronous function within the event loop
                loop.run_until_complete(initiate_chat())

                # Close the event loop
                loop.close()

                st.session_state.chat_initiated = True  # Set the state to True after running the chat


# stop app after termination command
st.stop()