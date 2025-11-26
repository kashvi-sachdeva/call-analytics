from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from datetime import datetime

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
CREDS = Credentials.from_service_account_file("/home/kashvi/call/call-analytics/service_acc.json", scopes=SCOPES)

def get_sheets_client():
    return build("sheets", "v4", credentials=CREDS)
