# transIQA
Image quality assessment by transfer learning


#### Dependencies:
Dlib 19.10 (for human face detection)
<br>Opencv-python (for imread)
<br>Scipy (scipy.stats for srocc & lcc evaluation)

#### Change log:

**Next Version(0.18)**
1. **Algorithm convergence**
    * BP method
    * Learning rate
    * and so on
1. **Code**
    * Usability
        * log time
        * save model
        * draw pictures
    * Coding redundance 
    * Coding reuse
    * Coding robustness
    * Coding Readable
    * and so on
1. **Model**
    * Layer setting
    * Loss function
    * and so on

**Version_0.17(16/04/18)**
1. **Training function rewrite**
    * One train(), one epoch
    * Per epoch: 5*reload Dataset(unlimited MODE for 16G ROM)
    * Per Dataset: 30 * iteration
1. **Loss**
    * Use L2-loss of pytorch: _torch.nn.functional.mse_loss_
1. **Evaluation**
    * Linear Correlation Coefficient(LCC)
    * Spearman Rank Order Correlation Coefficient(SROCC)
1. **Technique detail**
    1. _./src/tools.py_: write wrappers here for **Integrity**
    1. Stantarize input image: at _./src/tools.standardize_image()_
    1. Multiprocessing: try mtp while loading data. SEE _./src/mtp_using.py_
    1. Argparse: In main.py, control running mode
    1. Two running mode control: Limited and unlimited for pc and lab environment
    1. LCC & SROCC: _scipy.stats_
    1. _python *.py 2>&1 | tee ./src/training_log.txt_

**Version_0.16(12/04/18)**
1. **Dataset** dlib filtered faces
    * Format: _.npy_ format faces(59034 total, 11G).
    * Path and scores: _./data/face_score_generated_dlib.txt_
    * Generated from: _./data/level_score_generator/level_score_generator.py_ and *image_score_generated_dlib.txt*
1. **Source scripts**
    * _tools.py_:
        * Add Function: _prepare_faces(scale=1.2)_
    * _main.py_:
        * Add Function: _test()_
    * _dataset.py:_
        * Change dataset class: loading faces while initialization
        * Add dataset class test: when testing, init only.
1. **Technique detail**
    1. Image reading: _cv2.imread()_, _cv2.cvtColor()_, remove _Glymur_
    2. Face detection: _dlib_
    3. Dataset save and load: _np.save()_, _np.load()_
    4. Memory restore: _del_ and _gc.collect()_
    
**version_0.15(10/04/18)**
1. Dataset: full use:
<br>dataset structure: ./pristine/, ./GB/GB1-5, ./GN/GN1-5,
 ./JP2K/JP2K1-5, ./JPEG/JPEG1-5
2. Source scripts:
<br> dataset.py: dataset, dataloader
<br> main.py: train
<br> tools.py: show image
<br> model.py: net model structure

**version_0.10(09/04/18)**
<br>setup a simple network with dataset