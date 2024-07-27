import gdown
from zipfile import ZipFile

def download_database(drive_url: str, dst: str)-> None:
    gdown.download(drive_url, output=dst)
    
def unzip_file(file_name:str) -> None:
    with ZipFile('Flickr.zip', 'r') as zip_ref:
                zip_ref.extractall()

             