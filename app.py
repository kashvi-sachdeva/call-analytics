import streamlit as st
import tempfile
import os
from vertexai_sdk import get_genai_client, transcribe_audio_parallel
from prompt_transcribe import transcript_prompt
from response_schema import response_schema

st.set_page_config(page_title="Gemini Audio Transcription", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Transcription", "Prompt", "Response Schema"])

# Sidebar for config (only on Transcription page)
if page == "Transcription":
    st.sidebar.header("Configuration")
    gcp_api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
    model = st.sidebar.selectbox("Model", [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro"
    ])
    thinking_disabled = (model == "gemini-2.5-flash-lite")

    thinking_tokens = st.sidebar.number_input(
        "Thinking Tokens",
        min_value=0,
        max_value=256,
        value=100 if model == "gemini-2.5-flash" else 128 if model == "gemini-2.5-pro" else 0,
        disabled=thinking_disabled,
        help="Only supported for flash and pro models"
    )
    chunk_length_sec = st.sidebar.number_input("Chunk Length (sec)", min_value=60, max_value=600, value=360)
    max_workers = st.sidebar.number_input("Max Workers", min_value=1, max_value=8, value=4)
    st.title("Google Gemini Audio Transcription")
    st.write("Upload an audio file (mp3, wav, aac)")
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "aac"])
    if uploaded_file and gcp_api_key:
        if st.button("Transcribe Audio"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                client = get_genai_client(mode="sdk", api_key=gcp_api_key)
                transcript_json = transcribe_audio_parallel(tmp_path, client, chunk_length_sec=chunk_length_sec,model=model,thinking_tokens=None if thinking_disabled else thinking_tokens, max_workers=max_workers)
                st.success(f"Transcription complete. {len(transcript_json)} entries.")
                import pandas as pd
                from io import BytesIO

                # Convert to DataFrame
                df = pd.DataFrame(transcript_json)

                # Build Excel file in memory
                output = BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Transcript")

                # Naming format: audio filename + model
                audio_name = os.path.splitext(uploaded_file.name)[0]
                excel_filename = f"{audio_name}_{model}.xlsx"

                # Download button
                st.download_button(
                    label="📥 Download Transcript as Excel",
                    data=output.getvalue(),
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                for entry in transcript_json:
                    st.markdown(f"**{entry['start_time']} - {entry['end_time']}**: {entry['utterance']}")
                    if entry.get("speaker"):
                        st.write(f"Speaker: {entry['speaker']}")
                    if entry.get("classified_agent"):
                        st.write(f"Classified Agent: {entry['classified_agent']}")
                    if entry.get("classified_speaker"):
                        st.write(f"Classified Speaker: {entry['classified_speaker']}")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.remove(tmp_path)
    else:
        st.info("Please provide your API key and upload an audio file.")
    st.markdown("---")
    st.markdown("**Deployment Tip:** For production, use Streamlit Cloud or Docker. Uploaded files are stored temporarily and deleted after transcription.")

elif page == "Prompt":
    st.title("Prompt Used for Transcription")
    prompt_text = st.text_area("Edit the transcription prompt below and click 'Update' to save.", value=transcript_prompt, height=300)
    if st.button("Update Prompt"):
        with open("prompt_transcribe.py", "w") as f:
            f.write(f"transcript_prompt = '''{prompt_text}'''")
        st.success("Prompt updated! Reload the app to use the new prompt.")
    st.info("You can update the prompt to include non-speech classification or other instructions.")

elif page == "Response Schema":
    st.title("Response Schema for Transcription")
    st.json(response_schema)
    st.markdown("#### Add/Remove Schema Fields")
    new_field = st.text_input("Add new field to schema (e.g. non_speech)")
    if st.button("Add Field") and new_field:
        # Only works for dict schema, not full validation
        if isinstance(response_schema, dict):
            response_schema[new_field] = "string"
            with open("response_schema.py", "w") as f:
                f.write(f"response_schema = {response_schema}")
            st.success(f"Field '{new_field}' added! Reload the app to use the new schema.")
        else:
            st.error("Schema editing only supported for dict-based schemas.")
    st.info("You can update the schema to include non-speech classification or other fields.")
