from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
from io import BytesIO
import base64
from PIL import Image
from datetime import datetime
from models import db, DetectionHistory

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///history.db'
db.init_app(app)

def create_tables():
    with app.app_context():
        db.create_all()


@app.route('/')
def index():
    return render_template('index.html')

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

model = tf.keras.models.load_model('model.h5')

class_names = {
    0: 'Busuk_Buah_Matang', 
    1: 'Busuk_Rhizopus', 
    2: 'Daun_Gosong', 
    3: 'Kepang_Kelabu', 
    4: 'Tip_Burn'
}

disease_details = {
    'Busuk_Buah_Matang': {
        'symptoms': 'Gejala busuk buah matang yaitu buah yang matang akan membusuk dengan warna coklat muda. yang buahnya memiliki spora berwarna merah muda dan terdapat beberapa lubang pada buahnya.',
        'control': 'Pengendalian busuk buah matang yaitu dapat membuang buahnya terutama yang telah terserang. Kemudian menyemprotkan dengan menggunakan fungisida yang berbahan aktif tembaga. Contohnya yaitu Kocide 80 As, Funguran 82 WP atau Cupravit ) OB 21.',
        'medicine': [
            {'name': 'Kocide 80 As', 'image': 'images/kocide.jpeg'},
            {'name': 'Funguran 82 WP', 'image': 'images/funguran_82_wp.jpeg'},
        ]
    },
    'Busuk_Rhizopus': {
        'symptoms': 'Gejala busuk rhizopus pada buah ini akan mulai tampak terluhat membusuk, buahnya kan berair, warna dari buah menjadi coklat muda, apabila Anda menekan maka akan mengeluarkan cairan dari buah serta memiliki spora yang hitam warnanya.',
        'control': 'Pengendalian busuk rhizopus dapat membuang sebagian yang terserang atau dapat dipisah, kemudian menyemprotkan fungisida yang mengandung bahan aktif,',
        'medicine': [
            {'name': 'Baktisida Biopatek', 'image': 'images/biopatek.jpeg'}
        ]
    },
    'Daun_Gosong': {
        'symptoms': 'Gejala daun gosong pada bagian daun akan terlihat bercak berwarna agak ungu dan daun juga menjadi tubuh sesuai dengan biasanya.',
        'control': 'Pengendalian daun gosong dapat memangkas daunnya yang terkena penyakit, kemudian menyemprotkan fungisida Dithane M-45 atau Antracol 70 WP.',
        'medicine': [
            {'name': 'Dithane M-45', 'image': 'images/gambar1.jpg'},
            {'name': 'Antracol 70 WP', 'image': 'images/gambar2.jpg'}
        ]
    },
    'Kepang_Kelabu': {
        'symptoms': 'Gejala kepang kelabu akan timbul beberapa bagian buah menjadi busuk, berwarna menjadi coklat kemudian menjadi kering lalu buah mulai berjatuhan.',
        'control': 'Pengendalian kepang kelabu dapat membuang beberapa buah yang sudah terserang penyakit ini. Kemudian dapat menyemprotkan fungisida Benlate /Gosid 50 SD.',
        'medicine': [
            {'name': 'Baktisida Biopatek', 'image': 'images/biopatek.jpeg'},
        ]
    },
    'Tip_Burn': {
        'symptoms': 'Gejala tip burn bisa dilihat pada ujung daun tanaman strawberry yang terlihat terbakar. Daun yang menunjukkan gejala menjadi tidak segar dan pertumbuhannya terhambat. Gejala ini dapat terlihat di daun baru, lalu menyebar dari pusar tumbuh. Tip burn bisa menjadi indikasi kekurangan kalsium yang mengakibatkan sel daun',
        'control': 'Pengendalian tip burn pada tanaman stroberi melibatkan penyiraman yang konsisten, penyediaan kalsium yang cukup, keseimbangan nutrisi, pengelolaan pH tanah yang ideal (5.5-6.5), menjaga kondisi lingkungan tetap sejuk, penggunaan mulsa untuk menjaga kelembapan, pemangkasan daun yang terkena, dan pemantauan rutin. Jika masalah berlanjut, lakukan uji tanah atau konsultasi dengan ahli agronomi.',
        'medicine': [
            {'name': 'Dithane M-45', 'image': 'images/gambar1.jpg'}           
        ]
    }
}

def predict_image(model, img_path):
    img_height, img_width = 128, 128
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)  # Get the highest confidence score

    if confidence < 0.6:  # Set threshold as 60%
        predicted_label = 'Tidak Diketahui'
    else:
        predicted_label = class_names.get(predicted_class, 'Tidak Diketahui')
    
    return predicted_label, confidence

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


### Deteksi #####
@app.route('/deteksi', methods=['GET', 'POST'])
def deteksi():
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                predicted_label, confidence = predict_image(model, file_path)
                details = disease_details.get(predicted_label, {
                    'symptoms': 'Tidak Diketahui',
                    'control': 'Tidak Diketahui',
                    'medicine': 'Tidak Diketahui'
                })
                # Save to database
                new_history = DetectionHistory(result=predicted_label, timestamp=datetime.now(), img_path=filename)
                db.session.add(new_history)
                db.session.commit()
                # Redirect to the results page
                return redirect(url_for('hasil', prediction=predicted_label, img_path=filename, 
                                        symptoms=details['symptoms'], control=details['control'], 
                                        medicine=details['medicine'], confidence=confidence))
        elif 'image-data' in request.form and request.form['image-data'] != '':
            image_data = request.form['image-data']
            image_data = image_data.replace('data:image/png;base64,', '')
            image_data = BytesIO(base64.b64decode(image_data))
            image = Image.open(image_data)
            filename = 'captured_image.png'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(file_path)
            predicted_label, confidence = predict_image(model, file_path)
            details = disease_details.get(predicted_label, {
                'symptoms': 'Tidak Diketahui',
                'control': 'Tidak Diketahui',
                'medicine': 'Tidak Diketahui'
            })
            # Save to database
            new_history = DetectionHistory(result=predicted_label, timestamp=datetime.now(), img_path=filename)
            db.session.add(new_history)
            db.session.commit()
            # Redirect to the results page
            return redirect(url_for('hasil', prediction=predicted_label, img_path=filename, 
                                    symptoms=details['symptoms'], control=details['control'], 
                                    medicine=details['medicine'], confidence=confidence))
        else:
            return redirect(request.url)
    return render_template('deteksi.html', prediction=None, img_path=None)

### Hasil Predisi ###
@app.route('/hasil')
def hasil():
    prediction = request.args.get('prediction')
    img_path = request.args.get('img_path')
    symptoms = request.args.get('symptoms')
    control = request.args.get('control')
    medicine = request.args.get('medicine')
    confidence = request.args.get('confidence')
    
    result = {
        'prediction': prediction,
        'img_path': img_path,
        'symptoms': symptoms,
        'control': control,
        'medicine': medicine,
        'confidence': confidence
    }
    
    return render_template('hasil.html', result=result)

@app.route('/rekomendasi_obat')
def rekomendasi_obat():
    disease = request.args.get('disease')
    if disease in disease_details:
        medicines = disease_details[disease]['medicine']
    else:
        medicines = [{'name': 'Tidak Diketahui', 'image': 'default.png'}]
    return render_template('rekomendasi_obat.html', disease=disease, medicines=medicines)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

### History ###
@app.route('/history')
def history():
    histories = DetectionHistory.query.order_by(DetectionHistory.timestamp.desc()).all()
    return render_template('history.html', histories=histories)


### Artikel ####
@app.route('/artikel', methods=['GET', 'POST'])
def artikel():
    diseases = [
        {'name': 'Busuk Rhizopus', 'image': 'images/busukrhizopus.jpg', 'detail_route': 'busuk_rhizopus'},
        {'name': 'Busuk Buah Matang', 'image': 'images/busuk-buah-matang.jpg', 'detail_route': 'busuk_buah_matang'},
        {'name': 'Kapang Kelabu', 'image': 'images/kepang-kelabu.jpg', 'detail_route': 'kapang_kelabu'},
        {'name': 'Tip Burn', 'image': 'images/Tip-burnnn.jpg', 'detail_route': 'tip_burn'},
        {'name': 'Daun Gosong', 'image': 'images/Daun_gosong117.jpg', 'detail_route': 'daun_gosong'}
    ]
    return render_template('artikel.html', diseases=diseases)

@app.route('/busuk_rhizopus')
def busuk_rhizopus():
    return render_template('busuk_rhizopus.html')

@app.route('/busuk_buah_matang')
def busuk_buah_matang():
    return render_template('busuk_buah_matang.html')

@app.route('/kapang_kelabu')
def kapang_kelabu():
    return render_template('kapang_kelabu.html')

@app.route('/tip_burn')
def tip_burn():
    return render_template('tip_burn.html')

@app.route('/daun_gosong')
def daun_gosong():
    return render_template('daun_gosong.html')

if __name__ == '__main__':
    create_tables()  # Ensure the tables are created before the app runs
    app.run(debug=True)