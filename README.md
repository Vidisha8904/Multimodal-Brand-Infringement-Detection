# Brand-infringement


This project is a **single Jupyter Notebook** that calculates an overall similarity score between two images by comparing both their **visual content** and any **text contained in the images**.

The notebook was designed with logo/image comparison in mind. It first removes the background from the input images, extracts visible text, calculates text similarity and visual similarity separately, and then combines the available scores into an overall similarity score.

## Approach

The pipeline consists of the following steps:

1. **Input images**
   - Two image paths are provided through the `image_paths` list.
   - The current notebook uses two Reliance Jio logo images as example inputs.

2. **Background removal**
   - The notebook uses the Hugging Face **RMBG-2.0** model (`briaai/RMBG-2.0`) to separate the foreground from the background.
   - The processed images are kept in memory as PNG buffers.
   - The processed images are also displayed for visual inspection.

3. **Text extraction**
   - Each processed image is encoded as Base64.
   - The images are sent to the Groq API using the **LLaMA 3.2 90B Vision Preview** model.
   - The prompt instructs the model to return only the visible text.
   - If no text is detected, the model is instructed to return `0`.

4. **Text similarity**
   - Extracted text is converted to lowercase and special characters are removed.
   - Text similarity is calculated using:
     - `SequenceMatcher` for fuzzy string similarity.
     - `Soundex` through the `phonetics` library for phonetic similarity.
   - The fuzzy and phonetic scores are combined when the fuzzy similarity passes the configured threshold.

5. **Visual similarity**
   - The notebook uses OpenAI's **CLIP** model, specifically `ViT-L/14@336px`.
   - CLIP generates image embeddings for the two processed images.
   - The embeddings are L2-normalized and their dot product is used as the visual similarity score.

6. **Overall similarity**
   - When text is detected in both images, the final score is calculated as:

     `Overall Similarity = 0.5 × Text Similarity + 0.5 × Image Similarity`

   - When text is not detected in either image, the image similarity is used as the overall similarity.

## Technologies Used

- Python
- PyTorch
- OpenAI CLIP
- Hugging Face Transformers
- RMBG-2.0
- Groq API
- LLaMA 3.2 90B Vision Preview
- Pillow (PIL)
- Matplotlib
- `difflib.SequenceMatcher`
- `phonetics`
- NumPy

## Installation

The notebook installs the main dependencies directly using pip:

```bash
pip install git+https://github.com/openai/CLIP.git
pip install sentence-transformers transformers kornia groq
pip install phonetics
pip install fuzzy
```

The notebook can be run in an environment such as **Google Colab**, provided the required models and dependencies can be loaded.

## API Key

The notebook requires a **Groq API key**.

The Groq client is initialized using:

```python
client = Groq(api_key=api_key)
```

Before running the text-extraction section, `api_key` must therefore be defined with a valid Groq API key.

## Input Images

Update the `image_paths` variable with the paths to the two images you want to compare:

```python
image_paths = [
    "/path/to/image_1.png",
    "/path/to/image_2.png"
]
```

The current notebook uses:

```python
image_paths = [
    "/content/Reliance_Jio_Logo.svg - Copy.png",
    "/content/Reliance_Jio_Logo.svg.png"
]
```

## Output

The notebook produces three similarity values when text is detected:

```text
Text Similarity: ...
Image Similarity: ...
Overall Similarity: ...
```

If text is not detected, the output contains:

```text
Text Similarity: N/A (Text not detected)
Image Similarity: ...
Overall Similarity: ...
```

The scores are represented as decimal values, where higher values indicate greater similarity according to the corresponding similarity calculation.

## Project Structure

This project currently consists of a single notebook:

```text
Logo_Similarity_Vidisha.ipynb
```

The notebook contains the complete pipeline, including dependency installation, image preprocessing, background removal, text extraction, similarity calculations, and result display.

## Important Notes

- The notebook is currently implemented as a **notebook-based prototype**, rather than a standalone Python package or application.
- The example image paths point to Google Colab's `/content/` directory and may need to be changed when running elsewhere.
- The Groq API is required for the text-extraction stage.
- The RMBG-2.0 and CLIP models can require substantial computational resources. CUDA is used when available.
- The notebook's text-similarity implementation currently uses fuzzy matching and Soundex. Although `SentenceTransformer` is imported, it is not used in the final similarity calculation.
- The notebook's final weighting gives equal importance to text similarity and image similarity when both are available.

## Example Use Case

The project can be used as a starting point for comparing two logos or other images where both **visual appearance** and **textual content** are relevant.

For example, two logos may have similar shapes and visual features but different names, or they may contain similar text while having noticeably different visual designs. Combining the two signals provides a single score based on both aspects.
