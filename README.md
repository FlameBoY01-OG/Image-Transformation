# NeuroBrush

Neural Style Transfer application that transforms photos into artistic images using deep learning.

## Overview

NeuroBrush applies the artistic style of one image to the content of another using VGG19-based neural style transfer. The application features a modern web interface with drag-and-drop support and a FastAPI backend.

## Features

- **Neural Style Transfer**: VGG19-based implementation with multiple style layers
- **High Quality Output**: 768px processing resolution with total variation loss for smooth results
- **Drag & Drop Interface**: Intuitive file upload with preview
- **Original Size Preservation**: Output maintains input image dimensions
- **Real-time Progress**: Live optimization updates during processing

## Technology Stack

**Backend:**
- Python 3.x
- PyTorch
- FastAPI
- torchvision

**Frontend:**
- HTML5
- CSS3
- Vanilla JavaScript

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Image-Transformation
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Start the backend server:
```bash
python app.py
```

The API will be available at `http://localhost:8000`

4. Open the frontend:
```bash
cd ../frontend
python -m http.server 5500
```

The server will be available at http://localhost:5500

## Usage

1. Upload or drag-drop a content image (your photo)
2. Upload or drag-drop a style image (artwork style)
3. Click "Transform" button
4. Wait approximately 60 seconds for processing
5. Download the resulting styled image

## API Endpoints

### GET /
Health check endpoint

**Response:**
```json
{
  "message": "NeuroBrush API is active"
}
```

### POST /transform
Transform content image with style image

**Parameters:**
- `content`: Image file (multipart/form-data)
- `style`: Image file (multipart/form-data)

**Response:**
- PNG image (image/png)

## Architecture

### Neural Style Transfer Engine

The NST engine uses VGG19 pretrained model with the following components:

- **Content Loss**: Captures high-level content features (conv_4 layer)
- **Style Loss**: Matches style across multiple layers (conv_1 to conv_5)
- **Total Variation Loss**: Reduces noise and creates smoother outputs
- **LBFGS Optimizer**: Efficient optimization with strong Wolfe line search

### Processing Pipeline

1. Load and preprocess images
2. Extract features using VGG19
3. Optimize input image to minimize combined loss
4. Apply total variation regularization
5. Resize to original dimensions
6. Return PNG format for lossless quality

## Configuration

Default parameters in `nst_engine.py`:

- `num_steps`: 500 optimization iterations
- `style_weight`: 1,000,000
- `content_weight`: 1
- `tv_weight`: 10
- `max_size`: 768px processing resolution

Adjust these values to control style strength, content preservation, and smoothness.

## Project Structure

```
Image-Transformation/
├── backend/
│   ├── app.py                 # FastAPI server
│   ├── nst_engine.py          # Neural style transfer implementation
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── index.html            # Main HTML page
│   ├── script.js             # JavaScript logic
│   └── style.css             # Styling
└── README.md
```

## Performance

- **Processing Time**: ~60 seconds per image
- **Resolution**: Processes at 768px, outputs at original size
- **Memory**: Requires ~2GB RAM, GPU recommended for faster processing
- **Output Format**: PNG for lossless quality

## License

MIT

## Acknowledgments

Based on the paper "A Neural Algorithm of Artistic Style" by Gatys et al.
