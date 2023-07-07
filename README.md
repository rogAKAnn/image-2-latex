# image-2-latex

This is the model I used as part of the thesis project: Building an AI application to generate TeX code of the document images containing mathematical images.
The mathematical images, after being detected and segmented, runs into the model to generate the TeX corresponding to it.

This model achieves 85.77% BLEU Score on the test set of IM2LATEX-100K dataset.

# Component
- CNN Network (taken from "Translating Math Formula Images to LaTeX Sequences Using Deep Neural Networks with Sequence-level Training",  Zelun Wang, 2019).
- Positional Encoding
- LSTM with Addition Attention.

# How to use
- Clone the project, then install the dependencies as specified in requirements.txt
- Fill samples folder with images of your choice.
- Run python main.py
- The result will be outputted in the output folder.
