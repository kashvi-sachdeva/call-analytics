import streamlit as st
import tempfile
import os
from vertexai_sdk import get_genai_client, transcribe_audio_parallel
from prompt_transcribe import transcript_prompt
from response_schema import response_schema
from datetime import datetime
from sheet_utils import get_sheets_client
st.set_page_config(page_title="Gemini Audio Transcription", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Transcription", "Prompt", "Response Schema"])
#get speaking time
def get_speaking_times(transcript_json):
    """
    Returns a dict: {speaker_name: total_seconds_spoken}
    """
    from datetime import datetime

    speaker_times = {}
    for entry in transcript_json:
        speaker = entry.get("speaker")
        start = entry.get("start_time")
        end = entry.get("end_time")
        if not speaker or not start or not end:
            continue

        # Convert mm:ss → seconds
        try:
            mm, ss = map(int, start.split(":"))
            start_sec = mm * 60 + ss
            mm, ss = map(int, end.split(":"))
            end_sec = mm * 60 + ss
        except:
            continue

        duration = max(0, end_sec - start_sec)
        speaker_times[speaker] = speaker_times.get(speaker, 0) + duration

    return speaker_times

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

def update_master_index(sheets, spreadsheet_id, index_row, required_columns=None):
    """
    Append a row to Master_Index, creating/updating sheet as needed.
    - sheets: Google Sheets service.spreadsheets()
    - index_row: list of values to append
    - required_columns: list of column headers to enforce
    """
    if required_columns is None:
        required_columns = ["Audio", "Model", "Duration", "Sheet_URL", "Timestamp", "Speaking_Times"]

    # 1️⃣ Get existing sheets
    sheet_metadata = sheets.get(spreadsheetId=spreadsheet_id).execute()
    existing_sheets = {s["properties"]["title"]: s["properties"]["sheetId"] for s in sheet_metadata["sheets"]}

    if "Master_Index" not in existing_sheets:
        # Create the sheet if missing
        sheets.batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": [{"addSheet": {"properties": {"title": "Master_Index"}}}]}
        ).execute()
        # Write header row
        sheets.values().update(
            spreadsheetId=spreadsheet_id,
            range="Master_Index!A1",
            valueInputOption="RAW",
            body={"values": [required_columns]}
        ).execute()
    else:
        # Check headers in case some columns were added/edited
        result = sheets.values().get(
            spreadsheetId=spreadsheet_id, range="Master_Index!A1"
        ).execute()
        existing_headers = result.get("values", [[]])[0]

        # Add missing headers
        missing_cols = [c for c in required_columns if c not in existing_headers]
        if missing_cols:
            updated_headers = existing_headers + missing_cols
            sheets.values().update(
                spreadsheetId=spreadsheet_id,
                range="Master_Index!A1",
                valueInputOption="RAW",
                body={"values": [updated_headers]}
            ).execute()

            # Ensure index_row aligns with updated_headers
            index_row_dict = dict(zip(required_columns, index_row))
            index_row = [index_row_dict.get(c, "") for c in updated_headers]

    # 2️⃣ Append new row
    sheets.values().append(
        spreadsheetId=spreadsheet_id,
        range="Master_Index!A2",
        valueInputOption="RAW",
        body={"values": [index_row]}
    ).execute()

def push_transcript_to_google_sheets(df, spreadsheet_id, audio_name, model_name, duration_sec, speaking_times, latency, type_of_call):
    service = get_sheets_client()
    sheets = service.spreadsheets()

    # Tab Name
    sheet_name = f"{audio_name}_{model_name}".replace(" ", "_")
    existing_sheets = sheets.get(spreadsheetId=spreadsheet_id).execute()["sheets"]
    existing_titles = [s["properties"]["title"] for s in existing_sheets]
    suffix = 1
    while sheet_name in existing_titles:
        sheet_name = f"{audio_name}_{model_name}_{suffix}".replace(" ", "_")
        suffix += 1
    # 1️⃣ Create the sheet/tab
    requests = [{
        "addSheet": {
            "properties": {
                "title": sheet_name,
                "gridProperties": {"rowCount": 2000, "columnCount": 20}
            }
        }
    }]

    response = service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": requests}
    ).execute()

    # Get the new sheetId
    sheet_id = response['replies'][0]['addSheet']['properties']['sheetId']

    # 2️⃣ Write transcript DataFrame
    values = [df.columns.tolist()] + df.values.tolist()
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"{sheet_name}!A1",
        valueInputOption="RAW",
        body={"values": values}
    ).execute()

    # 3️⃣ Set tab color based on model
    MODEL_TAB_COLORS = {
        "gemini-2.5-pro":       {"red": 0.10, "green": 0.60, "blue": 0.90},   # blue
        "gemini-2.5-flash":     {"red": 1.00, "green": 0.70, "blue": 0.20},   # orange
        "gemini-2.5-flash-lite":{"red": 0.80, "green": 0.10, "blue": 0.10},   # red
        "gemini-1.5-pro":       {"red": 0.40, "green": 0.40, "blue": 0.40},   # gray
    }
    tab_color = MODEL_TAB_COLORS.get(model_name)
    if tab_color:
        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={
                "requests": [
                    {
                        "updateSheetProperties": {
                            "properties": {
                                "sheetId": sheet_id,
                                "tabColor": tab_color
                            },
                            "fields": "tabColor"
                        }
                    }
                ]
            }
        ).execute()
    import json
    speaker_times_str = json.dumps(speaking_times) 
    sheet_metadata = sheets.get(spreadsheetId=spreadsheet_id).execute()
    sheet_id = None
    for s in sheet_metadata["sheets"]:
        if s["properties"]["title"] == sheet_name:
            sheet_id = s["properties"]["sheetId"]
            break
    sheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit#gid={sheet_id}"

    index_row = [
        audio_name,
        model_name,
        duration_sec,
        sheet_url,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        speaker_times_str,
        type_of_call,
        latency
    ]

    # 7️⃣ Update Master_Index using the robust function
    required_columns = ["Audio", "Model", "Duration", "Sheet_URL", "Timestamp", "Speaking_Times", "Type_of_Call", "Latency"]
    update_master_index(sheets, spreadsheet_id, index_row, required_columns)

    
    return sheet_url

from pydub import AudioSegment
def get_audio_duration_seconds(path):
    audio = AudioSegment.from_file(path)
    return len(audio) / 1000 

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
    type_of_call= st.sidebar.text_input("Type of Call")
    st.title("Google Gemini Audio Transcription")
    st.write("Upload an audio file (mp3, wav, aac)")
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "aac"])
    if uploaded_file:
        st.audio(uploaded_file, format=uploaded_file.type)
    if uploaded_file and gcp_api_key:
        if st.button("Transcribe Audio"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                total_duration_sec=get_audio_duration_seconds(tmp_path)
                st.info(f"Audio duration: {total_duration_sec} seconds")
                client = get_genai_client(mode="sdk", api_key=gcp_api_key)
                transcript_json, latency = transcribe_audio_parallel(tmp_path, client, chunk_length_sec=chunk_length_sec,model=model,thinking_tokens=None if thinking_disabled else thinking_tokens, max_workers=max_workers)
                st.success(f"Transcription complete. {len(transcript_json)} entries.")
                import pandas as pd
                # from io import BytesIO

                # # Convert to DataFrame
                # df = pd.DataFrame(transcript_json)

                # # Build Excel file in memory
                # output = BytesIO()
                # with pd.ExcelWriter(output, engine="openpyxl") as writer:
                #     df.to_excel(writer, index=False, sheet_name="Transcript")

                # # Naming format: audio filename + model
                # audio_name = os.path.splitext(uploaded_file.name)[0]
                # excel_filename = f"{audio_name}_{model}.xlsx"

                # # Download button
                # st.download_button(
                #     label="📥 Download Transcript as Excel",
                #     data=output.getvalue(),
                #     file_name=excel_filename,
                #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                # )
                if transcript_json:
                    df = pd.DataFrame(transcript_json)
                    
                spreadsheet_id = "1g4pRBIhsXN5fc2-HVnJLmIix1YfKlAWcSVOtApER_i4"  
                audio_name = os.path.splitext(uploaded_file.name)[0]
                speaking_times = get_speaking_times(transcript_json)
                sheet_name = push_transcript_to_google_sheets(
                    df=df,
                    spreadsheet_id=spreadsheet_id,
                    audio_name=audio_name,
                    model_name=model,
                    duration_sec=total_duration_sec,
                    speaking_times=speaking_times,
                    latency=latency,
                    type_of_call=type_of_call
                )

                st.success(f"Uploaded to Google Sheet as tab: {sheet_name}")


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
