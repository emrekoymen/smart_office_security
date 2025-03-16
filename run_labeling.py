from GroundingDINO import GroundingDinoLabellingTool

def main():
    print("\n=== Smart Office System Dataset Labeling ===")
    
    # Define input and output folders
    input_folder = "dataset/images"
    output_image_folder = "dataset/output/annotated_images"
    output_xml_folder = "dataset/output/xml_annotations"
    
    # Use multiple text queries to improve detection chances
    text_queries = "person . human . man . woman . people . individual"
    
    print(f"Input folder: {input_folder}")
    print(f"Output folders: {output_image_folder} (images), {output_xml_folder} (annotations)")
    print(f"Using text queries: {text_queries}")
    
    # Create and run the labeling tool
    labeler = GroundingDinoLabellingTool(
        input_folder=input_folder,
        output_image_folder=output_image_folder,
        output_xml_folder=output_xml_folder,
        text_queries=text_queries  # Multiple text queries
    )
    
    labeler.run()
    
    print("\n=== Labeling process completed ===")
    print("Results:")
    print(f"- Annotated images saved to: {output_image_folder}")
    print(f"- XML annotations saved to: {output_xml_folder}")

if __name__ == "__main__":
    main() 