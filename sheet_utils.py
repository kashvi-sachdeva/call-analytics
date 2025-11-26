import streamlit as st
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def get_sheets_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service"],
        scopes=SCOPES
    )

    from googleapiclient.discovery import build
    return build("sheets", "v4", credentials=creds)
