# EasyOCR Coding and Testing

**Intro**

Since Tesseract is slower for the real life photos due to the needed multiple tries/rotations, would like to try out "easyocr" instead. This seems to be much faster for "not-so-clean" photos.

TODO:
Would need to find ways to adjust bbox merging algorithm for easyocr

-----------------------------------------

**Installation**

If making an venv, I recommend calling it .venv in order to make use of existing ".gitignore" file entry:
- For Windows:
`python -m venv .venv`

<br>
Then:

`pip install easyocr`

-----------------------------------------