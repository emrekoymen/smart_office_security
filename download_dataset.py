import os
import requests
import time
from PIL import Image
from io import BytesIO

# Create directories if they don't exist
os.makedirs("dataset/images", exist_ok=True)

# List of specific Unsplash image IDs known to contain people
# These are public domain images from Unsplash featuring people
image_urls = [
    "https://images.unsplash.com/photo-1560250097-0b93528c311a",  # businessman
    "https://images.unsplash.com/photo-1513258496099-48168024aec0",  # person walking
    "https://images.unsplash.com/photo-1485217988980-11786ced9454",  # woman working
    "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df",  # office worker
    "https://images.unsplash.com/photo-1514369118554-e20d93546b30",  # person with hard hat
    "https://images.unsplash.com/photo-1463453091185-61582044d556",  # man standing
    "https://images.unsplash.com/photo-1540569876033-6e5d046a1d77",  # people working
    "https://images.unsplash.com/photo-1517245386807-bb43f82c33c4",  # construction site
    "https://images.unsplash.com/photo-1508341591423-4347099e1f19",  # person with vest
    "https://images.unsplash.com/photo-1517732306149-e8f829eb588a"   # crowd of people
]

def download_image(url, save_path):
    try:
        # Add parameters to get a reasonably sized image
        full_url = f"{url}?fit=crop&w=800&h=600&q=80"
        response = requests.get(full_url)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Open the image to verify it's valid
        img = Image.open(BytesIO(response.content))
        
        # Save the image
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded image: {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

# Download images
for i, url in enumerate(image_urls):
    # Format the image filename
    image_path = f"dataset/images/person_{i+1:02d}.jpg"
    
    # If the image already exists, skip it
    if os.path.exists(image_path):
        print(f"Image {image_path} already exists, skipping...")
        continue
    
    # Download the image
    success = download_image(url, image_path)
    
    # Add a small delay to avoid hitting rate limits
    time.sleep(1)

print("\nDataset download complete.")
print(f"Downloaded images are saved in the 'dataset/images' directory.") 