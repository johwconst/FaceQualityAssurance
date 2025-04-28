import os
import requests
import time

def download_image_from_site_this_person_does_not_exist():
    url = "https://thispersondoesnotexist.com"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            time_string = time.strftime("%Y%m%d%H%M%S")
            file_path = os.path.join(folder_path, f"generated_image_{time_string}.jpg")
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"Image saved to: {file_path}")
        else:
            print(f"Error downloading image: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image: {e}")

if __name__ == "__main__":
    # Create the folder if it doesn't exist
    folder_path = os.path.join(os.path.dirname(__file__), "photos")
    os.makedirs(folder_path, exist_ok=True)
    for i in range(10):
        download_image_from_site_this_person_does_not_exist()  
    