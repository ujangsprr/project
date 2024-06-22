import os
import numpy as np
import cv2
import json
import tflite_runtime.interpreter as tflite

# Memuat model TFLite
print("Memuat model TFLite...")
interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
print("Model TFLite berhasil dimuat.")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def classify(image):
    """
    Mengklasifikasikan gambar menggunakan model TFLite yang telah dimuat.

    Parameter:
    image (numpy.ndarray): Gambar yang akan diklasifikasikan.

    Returns:
    tuple: Kelas prediksi dan tingkat kepercayaan.
    """
    print("Mengklasifikasikan gambar...")
    # Pra-pemrosesan gambar
    input_shape = input_details[0]['shape']
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    input_data = np.expand_dims(normalized_image_array, axis=0)
    
    # Mengatur input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Melakukan inferensi
    interpreter.invoke()
    
    # Mendapatkan hasil prediksi
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    img_class = {0: "Bandeng", 1: "Nila", 2: "Udang", 3: "Nothing"}
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx] * 100
    predicted_class = img_class[class_idx]
    
    print(f"Gambar diklasifikasikan sebagai {predicted_class} dengan kepercayaan {confidence:.2f}%")
    return predicted_class, confidence

def draw_text(frame, text, pos):
    """
    Menggambar teks pada frame di posisi yang ditentukan.

    Parameter:
    frame (numpy.ndarray): Gambar frame.
    text (str): Teks yang akan digambar.
    pos (tuple): Posisi (x, y) di mana teks akan digambar.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, pos, font, 1, (255, 0, 0), 2, cv2.LINE_AA)

# Memulai penangkapan video
print("Memulai penangkapan video...")
cap = cv2.VideoCapture(0)

# Membuat atau memeriksa folder output
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Folder output dibuat: {output_folder}")

# Inisialisasi daftar untuk menyimpan data output
output_data = []

while True:
    ret, frame = cap.read() 
    if ret:
        height, width, _ = frame.shape
        print("Membaca frame...")
        # Membagi frame menjadi 4 bagian yang sama
        sub_frames = [
            frame[0:height//2, 0:width//2],
            frame[0:height//2, width//2:width],
            frame[height//2:height, 0:width//2],
            frame[height//2:height, width//2:width]
        ]
        
        detected_objects = {"Bandeng": 0, "Nila": 0, "Udang": 0}
        counts = {"Bandeng": 0, "Nila": 0, "Udang": 0}
        
        for sub_frame in sub_frames:
            # Mengubah ukuran sub-frame untuk klasifikasi
            frame_to_clf = cv2.resize(sub_frame, (224, 224))
            
            # Mengklasifikasikan sub-frame
            class_name, confidence = classify(frame_to_clf)
            
            # Memperbarui kamus objek yang terdeteksi
            if class_name != 'Nothing':  # Hanya diperbarui jika kelas bukan 'Nothing'
                detected_objects[class_name] += confidence
                counts[class_name] += 1
        
        # Menghitung rata-rata kepercayaan untuk setiap kelas objek yang terdeteksi
        for key in detected_objects:
            if counts[key] > 0:
                detected_objects[key] /= counts[key]
        
        # Menampilkan kelas objek yang terdeteksi dan kepercayaan mereka secara vertikal
        y_offset = 40
        for class_name, confidence in detected_objects.items():
            text = f"{class_name}: {confidence:.2f}%"
            draw_text(frame, text, (20, y_offset))
            y_offset += 40
        
        # Menampilkan frame
        cv2.imshow("Prediksi", frame)
        
        # Menyimpan gambar output untuk setiap kelas yang terdeteksi (kecuali "Nothing" dan kepercayaan 0%)
        for class_name, confidence in detected_objects.items():
            if class_name != "Nothing" and confidence > 0:
                output_file = os.path.join(output_folder, f"{class_name}_{confidence:.2f}.jpg")
                cv2.imwrite(output_file, frame)
                print(f"Gambar {class_name} dengan kepercayaan {confidence:.2f}% disimpan ke {output_file}")

        # Menambahkan data label kelas ke daftar
        output_data.append(detected_objects)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menyimpan data output ke file JSON
output_json = os.path.join(output_folder, 'labels.json')
with open(output_json, 'w') as f:
    json.dump(output_data, f, indent=4)
    print(f"Data output disimpan ke {output_json}")

# Melepaskan penangkapan video
cap.release()
cv2.destroyAllWindows()
print("Penangkapan video dilepas dan semua jendela ditutup.")
