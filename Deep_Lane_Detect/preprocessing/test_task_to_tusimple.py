import json
import os

def generate_json_data(image_directory, output_filename="output.json", h_samples_start=160, h_samples_end=710, h_samples_step=10, run_time_value=1000):
    """
    Generates JSON data based on image files in a directory.

    Args:
        image_directory (str): The path to the directory containing image files.
        output_filename (str): The name of the output JSON file.
        h_samples_start (int): The starting value for h_samples.
        h_samples_end (int): The ending value for h_samples.
        h_samples_step (int): The step value for h_samples.
        run_time_value (int): The value for the run_time field.
    """

    # Generate the h_samples list
    h_samples = list(range(h_samples_start, h_samples_end + h_samples_step, h_samples_step))

    # Initialize an empty list for lanes (as per your example)
    lanes = []

    # Open the output file in write mode
    with open(output_filename, 'w') as f:
        # Iterate through files in the specified directory
        for root, _, files in os.walk(image_directory):
            for filename in files:
                # You might want to add a filter for image file extensions here
                # For example: if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                
                # Construct the raw_file path
                # Replace backslashes with forward slashes for consistent JSON paths
                raw_file_path = os.path.join(root, filename).replace(os.sep, '/')

                # Create the JSON object for the current image
                data = {
                    "h_samples": h_samples,
                    "lanes": lanes,
                    "run_time": run_time_value,
                    "raw_file": raw_file_path
                }

                # Write the JSON object as a single line to the file
                f.write(json.dumps(data) + '\n')

    print(f"JSON data successfully written to {output_filename}")

# --- How to use the script ---
if __name__ == "__main__":
    # Define your image directory here
    # Example: If your images are in a folder named 'my_images' in the same directory as the script
    my_image_directory = "dataset/test/images/img" # Change this to your actual directory

    # Call the function to generate the JSON file
    # You can adjust h_samples_start, h_samples_end, h_samples_step, and run_time_value
    generate_json_data(
        image_directory=my_image_directory,
        output_filename="test_tasks.json",
        h_samples_start=160,
        h_samples_end=710,
        h_samples_step=10,
        run_time_value=1000
    )