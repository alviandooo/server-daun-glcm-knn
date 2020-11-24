from flask import Flask, request,jsonify,send_file,send_from_directory
from werkzeug.utils import secure_filename

from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import StandardScaler  

import numpy as np 
import cv2 
import os
import re
from PIL import Image
import pickle 
import pandas as pd 
import json


from sklearn.preprocessing import LabelEncoder




app = Flask(__name__, static_url_path='/static')

app.config["IMAGE_UPLOADS"] =  './static'
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]

# if __name__ == '__main__':
#     app.run(port=9000)

def getModel():
    global model
    # load the model from disk
    model = pickle.load(open('model_baru', 'rb'))
    # result = model.predict(X_test) 
    print(' * model loaded...')

# -------------------- Utility function ------------------------
def normalize_label(str_):
    str_ = str_.replace(" ", "")
    str_ = str_.translate(str_.maketrans("","", "()"))
    str_ = str_.split("_")
    return re.sub(r'\d+$', '',''.join(str_[:2]))

def allowed_image(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".",1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else :
        return False

imgs = [] #list image matrix
labels = []

def preprocess_image(image):
    imgs = []
    # img = cv2.imread(image)
    img = cv2.imread(os.path.join(app.config["IMAGE_UPLOADS"], image))  
    # Convert RGB image to grayscale  use cv2.cvtColor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get Height and Weight from gray shape
    h, w = gray.shape
    # Set ymin, ymax, xmin, xmax from each gray shape
    ymin, ymax, xmin, xmax = h//150, h*149//150, w//150, w*149//150           

    # crop region of interest (ROI) to get important part from citra leaf
    crop = gray[ymin:ymax, xmin:xmax]

    # resize 20% use cv2.resize()
    resize = cv2.resize(crop, (0,0), fx=0.2, fy=0.2)

    imgs.append(resize)
    labels.append(normalize_label(os.path.splitext(image)[0]))

    glcm(imgs,labels)


# ----------------- calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ----------------------------------
def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    glcm = greycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    feature.append(label) 
    
    return feature


# ----------------- call calc_glcm_all_agls() for all properties ----------------------------------
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

def glcm(imgs, labels):
    glcm_all_agls = []
    for img, label in zip(imgs, labels): 
        glcm_all_agls.append(
                calc_glcm_all_agls(img, 
                                    label, 
                                    props=properties)
                                )
    
    columns = []
    angles = ['0', '45', '90','135']
    for name in properties :
        for ang in angles:
            columns.append(name + "_" + ang)
            
    columns.append("label")

    # Create the pandas DataFrame for GLCM features data
    glcm_df = pd.DataFrame(glcm_all_agls, 
                        columns = columns)
                        
    #save to csv
    glcm_df.to_csv("1.csv")
    
    knn(glcm_df)

def knn(glcm_df):
    X = glcm_df.iloc[:, 0:-1].values  

    


    global datalatih
    # load the model from disk
    x_train = pickle.load(open('datalatih', 'rb'))
    print(' * scaler loaded...')

    scaler = StandardScaler()  
    scaler.fit(x_train)
    X_test_scaled = scaler.transform(X)


    # result = model.predict(X_test) 

    
    # predict test data
    global pred
    global khasiat
    pred = model.predict(X_test_scaled).tolist()

    if pred[0] == "mangkok":
        khasiat = "Manfaat Mencegah terjadinya infeksi pada luka dan peradangan, Mengatasi lemah, letih, lesu, lunglai, dan sakit kepala, mencegah ganguuan pada mata, Membantu mengatasi sariawan, Mencegah terjadinya kerontokan rambut dan menjaga kesehatan rambut."
    elif pred[0] == "sirsak":
        khasiat = "Manfaat Mencegah dan melawan kanker, Mengatasi kolestrol, Mengobati peradangan, Mengatasi asam urat, Mencegah penyakit paru-paru, Bantu cegah insomnia, Mengobati kista ovarium pada wanita, Melancarkan pencernaan."
    elif pred[0] == "binahongmerah":
        khasiat = "Manfaat  Mencegah Kanker, Baik untuk Kesehatan Jantung, Mencegah Diabetes, Memulihkan Stamina, Mengatasi Anemia, Menangkal Radikal Bebas, Meningkatkan Nafsu Makan"
    elif pred[0] == "urangaring":
        khasiat = "Manfaat Mengatasi rambut rontok, Menghitamkan dan melebatkan rambut, Menghilangkan ketombe, Mencegah bertumbuhnya uban, Mengatasi rambut kering dan kusam"
    elif pred[0] == "tapakdara":
        khasiat = "Menghilangkan kepenatan akibat stres berat, mengatasi batuk dan pilek, Mengatasi demam tinggi yang berakibat kejang-kejang, mengobati bisul, borok, dan gondok."
    elif pred[0] == "sirihmerah":
        khasiat = "Manfaat sebagai antiseptik, Mengurangi Bau Mulut dan Badan, Mengobati Gusi Berdarah, Mengobati Gatal, Mengobati Batuk, Mengatasi Sakit Perut, Mengobati Asam Urat, Mengobati Bronkitis, Mengobati Pneumonia, Mengobati Jantung Koroner"
    elif pred[0] == "seri":
        khasiat = "Manfaat Mengobati sakit kepala, Mengobati sakit kepala, Menstabilkan gula darah"
    elif pred[0] == "salam":
        khasiat = "Manfaat mencegah komplikasi diabetes, mengobati batu ginjal, membakar lemak tubuh, mengobati asam urat, meredakan peradangan, bersifat anti-inflamasi, Mencegah darah tinggi, Meningkatkan kinerja jantung, Mengobati batuk, Mengatasi sembelit"
    elif pred[0] == "kumiskucing":
        khasiat = "Manfaat Mengatasi Rematik, Meredakan Batuk, Mengobati Gusi Bengkak, Mengatasi Masuk Angin, Menurunkan Berat Badan, Mengurangi Gatal Akibat Alergi, Menurunkan Tekanan Darah Tinggi, Mengatasi Batu Empedu"
    elif pred[0] == "kembangsepatu":
        khasiat = "Manfaat Mengobati diabetes, Mencegah infeksi saluran kemih, Mengatasi rambut rontok, Mengatasi uban dini, Mengeluarkan racun, Kesehatan kulit, Menurunkan berat badan"
    elif pred[0] == "katuk":
        khasiat = "Manfaat Melancarkan produksi ASI, Mengatasi anemia,  Meningkatkan daya tahan tubuh, Menjaga kesehatan mata, Meningkatkan vitalitas pria, Menjaga kesehatan tulang.Mengobati luka, Mencegah osteoporosis"
    elif pred[0] == "jarak":
        khasiat = "Manfaat mengatasi sakit gigi dan sariawan, Mengatasi Konstipasi (Sembelit), Mengobati Luka,  Mengobati Radang, Mengatasi Keputihan pada Mulut Bayi, Mengatasi Ketombe, Merawat Kesehatan Rambut"
    elif pred[0] == "jambubiji":
        khasiat = "Manfaat mengobati penyakit gusi, obat diare, menurunkan kolestrol, mencegah kangker,mengontrol diabetes, menyembukan demam berdarah, meningkatkan kesehatan kulit dan rambut"
    elif pred[0] == "insulin":
        khasiat = "Manfaat mencekah kangker, Membantu Mencegah dan Mengobati Diabetes, Mengatasi Gangguan Hati, Pankreas, dan Ginjal, Memiliki Efek Diuretik, Mencegah Penyakit Kardiovaskular, "
    elif pred[0] == "bayammerah":
        khasiat = "Manfaat  Menyehatkan pencernaan, Mengatasi anemia, Menguatkan sistem imun, Mencegah osteoporosis"
    else:
        khasiat = "manfaat belum diketahui."
    print('Pred = ',pred[0])
    return pred

print(' * Loading K-nn Model...')

getModel()

@app.route('/')
def hello_world():
    return 'hello world'

# @app.route('/upload', methods=['POST']) 
# def upload_base64_file(): 
#     """ 
#         Upload image with base64 format and get car make model and year 
#         response 
#     """

#     data = request.get_json()
#     # print(data)

#     if data is None:
#         print("No valid request body, json missing!")
#         return jsonify({'error': 'No valid request body, json missing!'})
#     else:

#         img_data = data['image']

#         # this method convert and save the base64 string to image
#         convert_and_save(img_data)

#     return jsonify({
#                     'status': 'success',
#                     }) , 400

# def convert_and_save(b64_string):
#     with open("imageToSave2.png", "wb") as fh:
#         fh.write(base64.decodebytes(b64_string.encode()))

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        if request.files.get('image'):
            image = request.files['image']
            
            if image.filename == "":
                return jsonify({
                    'status': 'error', 
                    'result': 'filename tidak boleh kosong'
                    }) , 400
            
            if allowed_image(image.filename):
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
                # image = request.files['image'].read()
                # image = Image.open(io.BytesIO(image))
                image = preprocess_image(filename)
                

                return jsonify({
                        "status" : 'success', 
                        "predict" : pred[0],
                        "khasiat" : khasiat
                    }) , 200
            else :
                return jsonify({
                    'status': 'error', 
                    'result': 'ups extensi file salah, hanya file jpeg,jpg, png'
                    }) , 400

if __name__ == '__main__':
    app.run()             

