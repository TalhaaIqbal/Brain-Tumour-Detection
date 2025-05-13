# Brain Tumor Detection Web Application

This is a web application built with Streamlit that uses deep learning to detect brain tumors from MRI/X-ray images. The application provides a user-friendly interface for uploading medical images and getting instant analysis results.

## Features

- Modern and professional user interface
- Support for MRI and X-ray image uploads
- Real-time tumor detection using deep learning
- Confidence score visualization
- Responsive design
- Clear result presentation

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository or download the files
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Make sure you have activated your virtual environment
2. Ensure that `model.h5` is present in the root directory
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. The application will open in your default web browser at `http://localhost:8501`

## Usage

1. Open the web application in your browser
2. Click on the "Choose an MRI/X-ray image..." button to upload an image
3. Wait for the analysis to complete
4. View the results, including:
   - Uploaded image display
   - Tumor detection result
   - Confidence score
   - Visual probability meter

## Important Notes

- This application is for educational and research purposes only
- Always consult healthcare professionals for medical decisions
- The model's predictions should not be used as a sole basis for medical diagnosis
- The application supports PNG, JPG, and JPEG image formats

## Model Information

The application uses a pre-trained deep learning model (`model.h5`) for tumor detection. The model has been trained on a dataset of brain MRI/X-ray images and can classify images as either containing a tumor or not.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. 