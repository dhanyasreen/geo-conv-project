# geo-conv-project

**Topic:** Variants of Basic Convolution Function — Standard Convolution vs. Dilated (Atrous) Convolution for multi-scale feature recognition in a geographic-imaging system.

## Problem
A geographic-imaging system (e.g. satellite/aerial image analysis) must recognize features that appear at very different spatial scales in the same scene — small isolated objects (buildings), medium linear structures (roads/rivers), and large area features (lakes/forest patches). A single small convolution kernel struggles to capture the large-scale context needed for the bigger features without either using very deep stacks of layers or losing spatial resolution through pooling.

## Approach
This project implements two convolution variants **from scratch in NumPy** (no black-box deep learning framework required) and applies both to the same synthetic multi-scale geo-image using an identical 3×3, 9-parameter kernel:

1. **Standard Convolution** (dilation = 1) — effective receptive field 3×3.
2. **Dilated / Atrous Convolution** (dilation = 3) — effective receptive field 7×7, same 9 trainable weights.

The two outputs are compared to show how receptive-field size changes what a convolution "sees," and which variant is more appropriate for multi-scale geographic feature recognition.

## Repository Structure
```
geo-conv-project/
├── README.md
├── requirements.txt
├── dataset/
│   ├── generate_dataset.py     # builds the synthetic multi-scale geo-image
│   ├── geo_image.npy
│   └── geo_image.csv
├── src/
│   └── conv_variants.py        # standard vs dilated convolution implementation
├── notebooks/
│   └── Conv_Variants_Analysis.ipynb
├── results/
│   ├── standard_conv_output.png
│   ├── dilated_conv_output.png
│   ├── side_by_side_comparison.png
│   └── receptive_field_diagram.png
└── screenshots/
    └── output.png
```

## How to Run
```bash
pip install -r requirements.txt
python dataset/generate_dataset.py   # creates the synthetic geo-image
python src/conv_variants.py          # runs both convolutions, saves results/
```

## Report
See `Report.docx` for the full write-up (Title Page, Problem Statement, Objective, Concept Used, Methodology, Implementation, Results & Output, Analysis, Conclusion, References).
