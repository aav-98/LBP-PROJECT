# LBP-PROJECT
This project was conducted for the course Biometric Systems at Sapienza University. The aim was to create a facial recognition system using the local binary pattern operator presented by Ojala, Pietikainen, and Maenpaa in 2002.

## TABLE OF CONTENTS

[DESCRIPTION](https://github.com/aav-98/LBP-PROJECT/edit/main/README.md#description)

[PROJECT STRUCTURE](https://github.com/aav-98/LBP-PROJECT/edit/main/README.md#project-structure)

[DATASETS](https://github.com/aav-98/LBP-PROJECT/edit/main/README.md#datasets)

[EXTENDED YALE B DATASET](https://github.com/aav-98/LBP-PROJECT/edit/main/README.md#extended-yale-b-dataset)  

[RUN PROGRAM](https://github.com/aav-98/LBP-PROJECT/edit/main/README.md#run-program)

[EVALUATION](https://github.com/aav-98/LBP-PROJECT/edit/main/README.md#evaluation)

[REFERENCES](https://github.com/aav-98/LBP-PROJECT/edit/main/README.md#references)

## DESCRIPTION
The following project uses open-source tools and libraries to create a facial recognition system. This includes:
* Scikit-learn 
* OpenCV
* Numpy
* Pandas

## PROJECT STRUCTURE
```
LBP Project
¦   LICENSE
¦   main.py
¦   
+---datasets
¦   +---test
¦   +---training
¦       
+---project_code
    ¦   lbp.py
    ¦   useful_methods.py
```

## DATASETS

### EXTENDED YALE B DATASET

The extended Yale Face Database B contains 16128 images of 28 human subjects under 9 poses and 64 illumination conditions.

The images in the dataset are stored in PGM files. "PGM (Portable Gray Map) files store grayscale 2D images. Each pixel within the image contains only one or two bytes of information (8 or 16 bits)."[^2] 

It can be accessed here: http://vision.ucsd.edu/~iskwak/ExtYaleDatabase/ExtYaleB.html

## RUN PROGRAM

To run the program, provide the necessary arguments in the command line:

```
-d --dataset [help: path to the dataset images]
-r --radius [help: specify the radius to be used for lbp-descriptor]
-n --neighbours [help: specify the number of interval points for lbp-descriptor]
```

## EVALUATION

## REFERENCES

[^1]: Ahonen T, Hadid A, Pietikäinen M. Face description with local binary patterns: application to face recognition. IEEE Trans Pattern Anal Mach Intell. 2006 Dec;28(12):2037-41. doi: 10.1109/TPAMI.2006.244. PMID: 17108377.
[^2]: [https://www.adobe.com/creativecloud/file-types/image/raster/pgm-file.html](https://www.adobe.com/creativecloud/file-types/image/raster/pgm-file.html)


