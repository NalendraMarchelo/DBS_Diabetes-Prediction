# Laporan Proyek Machine Learning - Nalendra Marchelo
## Domain Proyek
Diabetes mellitus merupakan salah satu penyakit kronis yang banyak ditemui di seluruh dunia dan menjadi salah satu penyebab utama kematian. Prediksi dini terhadap risiko diabetes penting dilakukan agar tindakan pencegahan dapat segera diambil untuk mengurangi komplikasi dan beban kesehatan. Oleh karena itu, pembangunan model prediksi diabetes menggunakan data medis pasien sangat dibutuhkan guna membantu tenaga medis dalam melakukan screening dan diagnosis awal.

Menurut International Diabetes Federation (IDF), pada tahun 2021 terdapat sekitar 537 juta orang dewasa yang hidup dengan diabetes di seluruh dunia, dan angka ini diperkirakan akan meningkat di masa depan. Oleh sebab itu, pengembangan metode prediksi yang akurat sangat penting untuk mendukung upaya kesehatan masyarakat. Studi terkait penggunaan machine learning dalam prediksi diabetes telah banyak dilakukan, salah satunya oleh Kavakiotis et al. (2017) yang mengulas berbagai teknik machine learning pada dataset Pima Indians Diabetes Database (PIDD) [1].

## Business Understanding
### Problem Statements
- Bagaimana membangun model prediksi risiko diabetes berdasarkan data medis pasien?
- Bagaimana meningkatkan akurasi prediksi diabetes agar dapat membantu tenaga medis dalam diagnosis dini?
- Apa saja variabel medis yang paling berpengaruh dalam memprediksi diabetes?

### Goals
- Mengembangkan model klasifikasi yang dapat memprediksi apakah seorang pasien berisiko menderita diabetes atau tidak dengan akurasi yang memadai.
- Menganalisis fitur-fitur penting yang mempengaruhi prediksi diabetes.
- Membandingkan performa beberapa algoritma machine learning untuk memilih model terbaik.

##### Solution statements
- - Menerapkan dua algoritma machine learning, yaitu Logistic Regression dan Random Forest, untuk membangun model prediksi.
- - Melakukan preprocessing data seperti penggantian nilai nol yang tidak valid dan scaling fitur untuk meningkatkan kualitas data input.
- - Mengevaluasi performa model menggunakan metrik akurasi, precision, recall, dan F1 score untuk memilih model terbaik.
- - Melakukan hyperparameter tuning pada model terpilih untuk meningkatkan akurasi prediksi.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Pima Indians Diabetes Dataset (PIDD) yang tersedia secara publik di UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes). Dataset ini berisi data medis dari wanita Pima Indian berusia di atas 21 tahun, dan berisi beberapa variabel yang berhubungan dengan diabetes.

### Variabel-variabel pada dataset Pima Indians Diabetes adalah sebagai berikut:
- Pregnancies: Jumlah kehamilan
- Glucose: Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral
- BloodPressure: Tekanan darah diastolik (mm Hg)
- SkinThickness: Ketebalan lipatan kulit trisep (mm)
- Insulin: Insulin serum 2 jam (mu U/ml)
- BMI: Indeks massa tubuh (berat badan dalam kg dibagi kuadrat tinggi dalam meter)
- DiabetesPedigreeFunction: Fungsi riwayat diabetes (penyakit keturunan)
- Age: Usia (tahun)
- Outcome: Kelas target (0 = tidak diabetes, 1 = diabetes)
Sebagian nilai pada beberapa fitur seperti Glucose, BloodPressure, SkinThickness, Insulin, dan BMI memiliki nilai nol yang tidak mungkin secara medis sehingga perlu penanganan khusus pada tahap data preparation.

### Informasi umum dataset eksplisit
- Jumlah baris: 768
- Jumlah kolom: 9
- Tipe data: 7 kolom bertipe int64, 2 kolom bertipe float64
- Tidak terdapat data duplikat (df.duplicated().sum() = 0)

### Distribusi Target:
Kolom Outcome merupakan variabel target, dengan distribusi sebagai berikut:
- 0 (tidak menderita diabetes): 500 data
- 1 (menderita diabetes): 268 data

beberapa fitur memiliki nilai minimum 0, seperti Glucose, BloodPressure, SkinThickness, Insulin, dan BMI. Nilai nol tersebut tidak mungkin secara medis dan dianggap sebagai nilai hilang (missing values).

### Analisis Outlier:
Analisis outlier dilakukan pada beberapa fitur yang secara statistik menunjukkan penyimpangan yang besar, seperti:
- Insulin: memiliki nilai maksimum 846 sementara median-nya hanya 30.5

Hasil deteksi outlier menggunakan metode Interquartile Range (IQR) pada kolom Insulin menemukan sebanyak 34 nilai outlier.

## Data Preparation
Tahap data preparation dilakukan untuk memastikan kualitas dan konsistensi data sebelum digunakan dalam pelatihan model machine learning. Langkah-langkah yang dilakukan adalah sebagai berikut:
- Mengatasi nilai tidak valid: Beberapa kolom seperti Glucose, BloodPressure, SkinThickness, Insulin, dan BMI memiliki nilai 0 yang secara medis tidak mungkin. Nilai-nilai ini dianggap sebagai data tidak valid dan diganti dengan NaN agar dapat diproses sebagai data hilang.
- Penanganan missing values: Nilai NaN pada kolom-kolom tersebut diisi menggunakan median masing-masing kolom. Pemilihan median dilakukan untuk meminimalkan pengaruh outlier.
- Pemisahan fitur dan target: Data dipisahkan menjadi fitur (X) dan target (y). Fitur terdiri dari semua kolom kecuali Outcome, sedangkan target adalah kolom Outcome.
- Pembagian data: Dataset dibagi menjadi data latih dan data uji dengan rasio 80:20. Stratifikasi dilakukan berdasarkan variabel target untuk menjaga proporsi kelas yang seimbang.
- Standardisasi fitur: Seluruh fitur numerik dinormalisasi menggunakan StandardScaler agar memiliki distribusi standar dengan mean 0 dan standar deviasi 1. Langkah ini penting untuk meningkatkan kinerja algoritma yang sensitif terhadap skala fitur, seperti Logistic Regression.

## Modeling
Pada tahap ini, dua algoritma klasifikasi digunakan untuk membangun model prediksi diabetes, yaitu Logistic Regression dan Random Forest Classifier. Kedua model dilatih menggunakan data fitur yang telah melalui proses preprocessing dan standardisasi.

#### 1. Logistic Regression
Logistic Regression adalah algoritma klasifikasi linear yang digunakan untuk memprediksi probabilitas suatu kelas, dalam hal ini apakah pasien berisiko diabetes atau tidak. Model ini diinisialisasi dengan parameter:
- random_state=42: Menetapkan seed random agar hasil eksperimen dapat direproduksi secara konsisten.

Parameter lain menggunakan nilai default dari library scikit-learn, seperti:
- penalty='l2': Regularisasi L2 untuk menghindari overfitting.
- solver='lbfgs': Optimizer yang efisien untuk dataset kecil hingga sedang.
- max_iter=100: Batas maksimum iterasi untuk konvergensi model.

Model ini cocok untuk baseline karena sederhana, cepat dilatih, dan interpretatif.

#### 2. Random Forest Classifier
Random Forest adalah algoritma ensemble berbasis pohon keputusan yang membentuk beberapa decision tree dan menggabungkan hasilnya melalui voting untuk meningkatkan akurasi dan mengurangi overfitting. Model ini diinisialisasi dengan:
- random_state=42: Menetapkan seed agar hasil dapat direproduksi.

Parameter lainnya masih default, seperti:
- n_estimators=100: Jumlah pohon dalam forest.
- max_depth=None: Tidak ada batasan kedalaman pohon, pohon tumbuh hingga semua daun homogen.
- min_samples_split=2: Minimal jumlah sampel yang dibutuhkan untuk membagi node.
- bootstrap=True: Pengambilan sampel dilakukan dengan pengembalian.

Random Forest dipilih sebagai kandidat utama karena kemampuannya menangani data dengan hubungan nonlinier, serta lebih tahan terhadap outlier dan fitur yang tidak relevan.

## Evaluation
Metrik evaluasi yang digunakan meliputi:
- Accuracy: Proporsi prediksi benar terhadap seluruh data.
- Precision: Rasio prediksi positif yang benar dari seluruh prediksi positif.
- Recall: Rasio prediksi positif yang benar dari seluruh data positif sebenarnya.
- F1 Score: Harmonik rata-rata dari precision dan recall, berguna jika distribusi kelas tidak seimbang.

Berdasarkan hasil evaluasi, Random Forest menunjukkan performa terbaik dengan akurasi sekitar 77,9%, precision 71,7%, recall 61,1%, dan F1 score 66%. Sementara untuk Logistic Regression akurasinya sekitar 70,8%, precision 60,0%, recall 50%, dan F1 score 54,5%.

#### Pemilihan Model Terbaik
Berdasarkan hasil di atas, Random Forest dipilih sebagai model terbaik karena memberikan kombinasi akurasi, precision, dan recall yang lebih tinggi, serta nilai F1 Score yang lebih seimbang.

#### Kaitan dengan Business Understanding
Model ini dikembangkan dengan tujuan mendukung pemeriksaan awal risiko diabetes. Dalam konteks business dan healthcare:
- Recall menjadi metrik penting karena menunjukkan seberapa baik model mendeteksi pasien yang benar-benar mengidap diabetes. Kesalahan dalam bentuk false negative (pasien positif tapi terdeteksi negatif) dapat berdampak serius karena pasien tidak segera mendapat penanganan.
- Precision juga penting, karena terlalu banyak false positive dapat menyebabkan biaya tambahan untuk pemeriksaan lebih lanjut yang sebenarnya tidak diperlukan.
- Dengan recall Random Forest mencapai 61.1%, model ini cukup efektif dalam mengidentifikasi pasien berisiko dan mampu menyeimbangkan antara deteksi dini dan efisiensi sumber daya.

#### Kesesuaian dengan Problem Statement dan Goals
- Problem statement: Bagaimana memprediksi risiko diabetes berdasarkan data medis pasien — terjawab dengan model klasifikasi yang dapat membedakan antara pasien dengan dan tanpa diabetes.
- Goals: Membangun model prediktif dengan akurasi dan recall yang baik — tercapai, terutama dengan Random Forest yang memberikan performa optimal.

Solusi yang dirancang (sistem prediksi risiko diabetes) berdampak signifikan dalam konteks pencegahan dan pengambilan keputusan awal di bidang kesehatan masyarakat.