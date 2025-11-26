
import json
from google.oauth2.service_account import Credentials
import streamlit as st

SERVICE_INFO = st.secrets["gcp_service"]

creds = Credentials.from_service_account_info(
    SERVICE_INFO,
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)

def get_sheets_client():
    from googleapiclient.discovery import build

    service = build('sheets', 'v4', credentials=creds)
    return service