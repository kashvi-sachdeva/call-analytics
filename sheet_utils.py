# sheet_utils.py
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import streamlit as st
from datetime import datetime

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Load service account info from Streamlit secrets
SERVICE_INFO = st.secrets["gcp_service"]  # The section name in secrets.toml

# Fix private key newlines
private_key = SERVICE_INFO["private_key"].replace("\\n", "\n")
SERVICE_INFO["private_key"] = private_key

# Create credentials
CREDS = Credentials.from_service_account_info(
    SERVICE_INFO,
    scopes=SCOPES
)

def get_sheets_client():
    """Returns a Google Sheets client using the service account."""
    return build("sheets", "v4", credentials=CREDS)
