# sheet_utils.py
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import streamlit as st
from datetime import datetime

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Load service account info from Streamlit secrets
SERVICE_INFO = st.secrets["gcp_service"]  # The section name in secrets.toml


# Create credentials
CREDS = Credentials.from_service_account_info(
    SERVICE_INFO,
    scopes=SCOPES
)

def get_sheets_client():
    """Returns a Google Sheets client using the service account."""
    return build("sheets", "v4", credentials=CREDS)
