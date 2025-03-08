#!/bin/bash
# Script to download office sample videos for testing

# Create videos directory if it doesn't exist
mkdir -p videos

echo "Downloading office test videos..."

# Download a reliable sample video that can be used for people detection testing
# This is from Google's common test videos collection
wget -O videos/office_test_video.mp4 "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/PeopleOnStreet.mp4"

# Check if download was successful
if [ -s videos/office_test_video.mp4 ]; then
    echo "Download complete. Test video with people saved as videos/office_test_video.mp4"
else
    # Try a backup video
    echo "Error downloading video. Trying backup source..."
    wget -O videos/office_test_video.mp4 "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4"
    
    if [ -s videos/office_test_video.mp4 ]; then
        echo "Download complete. Backup test video saved as videos/office_test_video.mp4"
    else
        echo "Error downloading video. Please check your internet connection."
        exit 1
    fi
fi

echo "Note: This test video is intended for testing the person detection system."
echo "For better results, you can replace it with actual office security footage." 