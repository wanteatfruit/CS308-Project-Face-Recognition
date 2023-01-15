# CS308-Project-Face-Recognition
## File structure
- `vgg.py` includes functions encapsulating the face recognition pipeline. `face_verification()` takes two image paths as inputs and outputs their cosine simiarity and whether they are the same faces. `face_identification()` takes one image path as input and outputs five most probable matches in VGGFace2's training set.
- `evaluation.ipynb` includes pre-processing meta files and test images, and evaluation for both verification and identification.
## Reproducing the results
- Install dependencies
```
pip install tensorflow
pip install git+https://github.com/rcmalli/keras-vggface.git
pip install mtcnn
```
- Prepare dataset

  For evaluating face identification, we use VGGFace2's evaluation set. 
The file structure should be like `./test/n000001/n001_01.jpg`
For evaluating face verification, we will not provide our evaluation set due to privacy issues. However you can build your own evaluation set and put them under `./veri_test/class_name/image_name.jpg`

- Run `evaluation.ipynb`
