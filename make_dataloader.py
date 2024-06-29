import os
from dotenv import load_dotenv

from process_data.pre_process import ProcessData
from ulti.create_folder import create_folder_if_not_exists, delete_folder_if_exists


load_dotenv(".env")

LOG_DATA_PATH = os.getenv("LOG_DATA_PATH")

delete_folder_if_exists(LOG_DATA_PATH)
create_folder_if_not_exists(LOG_DATA_PATH)


MintRec = ProcessData()
MintRec.setup()
