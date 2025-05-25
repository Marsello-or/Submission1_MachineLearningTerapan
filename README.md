# Predictive Analytics : Bank Customer Churn Prediction - Marsello Ormanda

## Domain Proyek: Keuangan
Industri perbankan menghadapi tantangan dalam mempertahankan pelanggan. Dengan semakin banyaknya pilihan layanan keuangan, risiko pelanggan beralih ke bank lain (churn) menjadi tinggi. Memprediksi pelanggan yang cenderung akan churn berdasarkan data historis dapat membantu bank untuk mengambil tindakan proaktif dalam strategi retensi pelanggan, sehingga dapat meminimalkan kehilangan pendapatan dan biaya akuisisi pelanggan baru.

## Business Understanding

### Problem Statements
- Bagaimana cara memprediksi pelanggan bank yang cenderung akan churn (berhenti menggunakan layanan) berdasarkan data historis nasabah?
- Faktor-faktor apa saja (misalnya skor kredit, usia, saldo, jumlah produk, status keanggotaan aktif, estimasi gaji) yang paling memengaruhi keputusan pelanggan untuk churn?

### Goals
- Mengembangkan model prediksi churn pelanggan yang akurat untuk mengidentifikasi nasabah berisiko tinggi.
- Memberikan pemahaman tentang hubungan antara karakteristik nasabah dan perilakunya dengan kemungkinan churn.
- Membantu bank dalam merancang strategi retensi pelanggan yang proaktif dan efektif.

### Solution Statements
- Membangun model machine learning (misalnya Random Forest Classifier, Logistic Regression, atau SVM) untuk memprediksi probabilitas atau kelas churn (0: tidak churn, 1: churn).
- Melakukan analisis fitur dari dataset (yang mencakup Credit Score, Country, Gender, Age, Balance, dll.) untuk mengetahui fitur-fitur mana yang paling signifikan dalam menentukan churn.
- Menggunakan evaluasi model seperti Accuracy, Precision, Recall, F1-Score, dan ROC AUC untuk mengukur performa model dan memilih pendekatan terbaik.

## Data Understanding
Dataset Bank Customer Churn Prediction berasal dari Kaggle, berisi informasi historis nasabah bank yang digunakan untuk memprediksi apakah mereka akan churn. Dataset ini mencakup berbagai atribut seperti skor kredit, negara, jenis kelamin, usia, masa keanggotaan (tenure), saldo rekening, jumlah produk bank yang dimiliki, status kartu kredit, status keanggotaan aktif, dan estimasi gaji. Target variabelnya adalah 'churn' (1 jika nasabah churn, 0 jika tidak).
Sumber didapatkan dari link berikut: [Customer Churn Data](https://www.kaggle.com/datasets/bhuviranga/customer-churn-data)

### Exploratory Data Analysis
Pada tahap ini dilakukan analisis untuk data yang ada di dalam dataset. Dataset memiliki 10000 data dan 12 kolom.

| Kolom | Jumlah Non-Null | Tipe Data |
|---|---|---|
| customer_id | 10000 | int64 |
| credit_score | 10000 | int64 |
| country | 10000 | object |
| gender | 10000 | object |
| age | 10000 | int64 |
| tenure | 10000 | int64 |
| balance | 10000 | float64 |
| products_number | 10000 | int64 |
| credit_card | 10000 | int64 |
| active_member | 10000 | int64 |
| estimated_salary | 10000 | float64 |
| churn | 10000 | int64 |

Output di atas menunjukkan bahwa dataset memiliki 10000 data dan 12 kolom.
- Terdapat 2 tipe data float64
- Terdapat 8 tipe data int64
- Terdapat 2 tipe data object (country, gender)

Berikut adalah statistik deskriptif dari dataset:

| Statistik | credit_score | age | tenure | balance | products_number | credit_card | active_member | estimated_salary | churn |
|---|---|---|---|---|---|---|---|---|---|
| Count | 10000.00 | 10000.00 | 10000.00 | 10000.00 | 10000.00 | 10000.00 | 10000.00 | 10000.00 | 10000.00 |
| Mean | 650.52 | 38.92 | 5.01 | 76485.89 | 1.53 | 0.71 | 0.52 | 100000.24 | 0.20 |
| Std | 96.65 | 10.49 | 2.90 | 62397.40 | 0.58 | 0.46 | 0.50 | 57510.49 | 0.40 |
| Min | 350.00 | 18.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 11.58 | 0.00 |
| 25% | 584.00 | 32.00 | 2.00 | 0.00 | 1.00 | 0.00 | 0.00 | 51002.11 | 0.00 |
| 50% | 652.00 | 37.00 | 5.00 | 97198.54 | 1.00 | 1.00 | 1.00 | 100193.91 | 0.00 |
| 75% | 718.00 | 44.00 | 8.00 | 127644.24 | 2.00 | 1.00 | 1.00 | 149388.25 | 0.00 |
| Max | 850.00 | 92.00 | 10.00 | 250898.09 | 4.00 | 1.00 | 1.00 | 199992.48 | 1.00 |

Digunakan `describe()` untuk memberikan informasi statistik.
- Count adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum.
- 25% adalah kuartil pertama.
- 50% adalah kuartil kedua.
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.

#### Checking Missing and Duplicated Value
Dataset tidak memiliki data duplikat dan missing value. Oleh karena itu, proses dapat dilanjutkan kepada analisis dan visualisasi data.

#### Boxplot Visualization
Untuk mempermudah visualisasi data, fitur dibagi menjadi `categorical_features` dan `numerical_features`.

**Deskripsi Boxplot:**
- `credit_score`: Rata-rata skor kredit sekitar 650. Sebagian besar data antara 584 hingga 718. Terdapat outlier di sisi bawah (skor sangat rendah).
- `age`: Rata-rata usia sekitar 38 tahun. Sebagian besar data antara 32 hingga 44 tahun. Terdapat outlier di sisi atas (usia sangat tua).
- `tenure`: Rata-rata masa keanggotaan sekitar 5 tahun. Sebagian besar data antara 3 hingga 7 tahun. Distribusi cukup simetris.
- `balance`: Rata-rata saldo rekening sekitar 76.485. Sebagian besar data antara 0 hingga 127.644. Ada banyak nasabah dengan saldo 0 (outlier ekstrim di sisi bawah atau nilai nol yang dominan).
- `products_number`: Rata-rata jumlah produk sekitar 1.5. Mayoritas nasabah memiliki 1 atau 2 produk. Terdapat outlier di sisi atas (nasabah dengan 3 atau 4 produk).
- `credit_card`: Variabel biner (0 atau 1). Mayoritas memiliki kartu kredit.
- `active_member`: Variabel biner (0 atau 1). Mayoritas adalah anggota aktif.
- `estimated_salary`: Rata-rata estimasi gaji sekitar 100.000. Sebagian besar data tersebar merata, tidak ada outlier yang jelas di luar rentang, distribusi cukup simetris.
- `churn`: Variabel biner (0 atau 1). Target. Mayoritas nasabah tidak churn.

### EDA - Univariate Analysis

**Analisis Hasil Distribusi Fitur Numerik:**
- `credit_score`, `age`, `tenure`, `estimated_salary`: Distribusi terlihat cukup simetris, mendekati normal atau seragam.
- `balance`: Terlihat sangat miring ke kanan (positively skewed) dengan konsentrasi besar di nilai nol, menunjukkan banyak nasabah tidak memiliki saldo atau saldo sangat rendah.
- `products_number`: Merupakan variabel diskrit, dengan distribusi yang menunjukkan mayoritas nasabah memiliki 1 atau 2 produk.

**Analisis Hasil Distribusi Fitur Kategorikal:**
- **Country Distribution (Distribusi Negara):** Mayoritas nasabah berasal dari 'France' (Prancis), diikuti oleh 'Germany' (Jerman) dan 'Spain' (Spanyol) dengan jumlah yang lebih sedikit namun seimbang antara keduanya.
- **Gender Distribution (Distribusi Jenis Kelamin):** Jumlah nasabah 'Male' (Laki-laki) sedikit lebih banyak dibandingkan 'Female' (Perempuan) dalam dataset.

#### EDA - Multivariate Analysis
**Analisis Hasil Box Plots (Numerical Features by Churn):**
- **`credit_score` by Churn:** Rata-rata skor kredit tampak sedikit lebih rendah untuk nasabah yang churn (1) dibandingkan dengan yang tidak churn (0).
- **`age` by Churn:** Nasabah yang churn cenderung memiliki usia rata-rata yang lebih tinggi dan rentang usia yang lebih sempit dibandingkan nasabah yang tidak churn.
- **`tenure` by Churn:** Distribusi tenure terlihat cukup mirip antara nasabah churn dan tidak churn.
- **`balance` by Churn:** Nasabah yang churn memiliki saldo rata-rata yang lebih tinggi dan variasi yang lebih besar dibandingkan nasabah yang tidak churn, yang sebagian besar memiliki saldo nol.
- **`products_number` by Churn:** Nasabah yang churn cenderung memiliki jumlah produk yang lebih tinggi, terutama 3 atau 4 produk, dibandingkan nasabah yang tidak churn yang mayoritas memiliki 1 atau 2 produk.
- **`estimated_salary` by Churn:** Distribusi estimasi gaji terlihat sangat mirip antara nasabah churn dan tidak churn.

#### Correlation Matrix
Visualisasi ini digunakan untuk mencari tahu fitur apa saja yang memiliki korelasi paling besar.
**Interpretasi Umum Heatmap:**
* **Warna Merah Cerah:** Menunjukkan korelasi positif yang kuat (mendekati +1).
* **Warna Biru Cerah:** Menunjukkan korelasi negatif yang kuat (mendekati -1).
* **Warna Pucat/Putih (mendekati 0):** Menunjukkan korelasi yang sangat lemah atau tidak ada korelasi linier.

**Analisis Korelasi Utama Terhadap 'Churn':**
* **`churn` vs. `age` (Korelasi 0.29):** Terdapat korelasi positif moderat antara `churn` dan `age`. Ini mengindikasikan bahwa semakin tua usia nasabah, semakin tinggi kemungkinan mereka untuk churn.
* **`churn` vs. `products_number` (Korelasi -0.06):** Korelasi negatif yang sangat lemah.
* **`churn` vs. `active_member` (Korelasi -0.16):** Korelasi negatif yang lemah. Nasabah yang lebih aktif cenderung tidak churn.
* **`churn` vs. `balance` (Korelasi 0.12):** Korelasi positif yang lemah. Nasabah dengan saldo lebih tinggi cenderung churn.
* **Fitur lainnya vs. `churn` (Korelasi Mendekati Nol):** `credit_score` (-0.03), `tenure` (-0.01), `credit_card` (-0.00), dan `estimated_salary` (0.01) memiliki korelasi yang sangat lemah dengan `churn`.

**Analisis Korelasi Antar Fitur Lainnya:**
* **`products_number` vs. `balance` (Korelasi -0.31):** Korelasi negatif moderat. Nasabah dengan lebih banyak produk cenderung memiliki saldo yang lebih rendah.
* **Tidak ada multikolinearitas ekstrem:** Tidak ada pasangan fitur yang menunjukkan multikolinearitas ekstrem (korelasi > 0.9) yang memerlukan penghapusan segera.

**Kesimpulan dari Heatmap Data Original:**
* `age`, `active_member`, dan `balance` adalah fitur numerik dengan korelasi paling relevan terhadap `churn`.
* Fitur lain memiliki korelasi linier yang sangat rendah dengan `churn`.
* Tidak ada masalah multikolinearitas parah antar fitur prediktor numerik.

### Data Preparation
Teknik yang digunakan:
- Filter Data: Melakukan filter terhadap data yang tidak perlu.
- Penghapusan Outlier: Menghapus data dengan nilai yang ekstrem.
- Label Encoding: Mengubah kategori menjadi tipe data numerik.
- Feature Scaling: Melakukan penskalaan data numerik.
- Train-test split data: Data dibagi menjadi 80% Train dan 20% Test.

#### Filter Data
Dilakukan penghapusan kolom `customer_id` karena tidak mengandung informasi yang berarti untuk menentukan churn dari customer.

#### Penghapusan Outlier
Outlier adalah nilai ekstrem dalam dataset yang dapat mengganggu analisis statistik dan kinerja model pembelajaran mesin. Dalam proyek ini, kami menggunakan metode **Interquartile Range (IQR)** untuk mendeteksi dan menghapus outlier dari fitur numerik.

##### Langkah-Langkah Metode IQR:
1. **Pilih Fitur Numerik**: Kolom yang berisi data numerik dipilih karena lebih rentan terhadap keberadaan outlier.
2. **Hitung Kuartil dan IQR**:
   - **Q1 (Kuartil 1)**: Nilai yang memisahkan 25% data terendah dari data lainnya.
   - **Q3 (Kuartil 3)**: Nilai yang memisahkan 75% data terendah dari 25% data tertinggi.
   - **IQR (Interquartile Range)**: Selisih antara Q3 dan Q1 ($IQR = Q3 - Q1$).
3. **Tentukan Ambang Batas Outlier**:
   - Batas bawah: $Q1 - 1.5 \times IQR$ dianggap Outlier Bawah.
   - Batas atas: $Q3 + 1.5 \times IQR$ dianggap Outlier Atas.
4. **Filter Outlier**: Baris data dengan nilai di luar batas bawah dan atas dianggap sebagai outlier dan dihapus dari dataset.

##### Alasan Menggunakan IQR?
- Metode ini tahan terhadap nilai ekstrem, berbeda dengan metode yang menggunakan rata-rata atau standar deviasi.
- Memastikan data dalam rentang kuartil tidak terpengaruh, sehingga informasi yang penting tetap terjaga.

Jumlah baris sebelum penghapusan outlier: 10000.
Jumlah baris setelah penghapusan outlier: 9568.
Jumlah baris yang dihapus (outlier): 432.

Pada tahap ini, dilakukan filtering outliers yang bertujuan untuk menghapus outliers atau data-data yang berada di luar Interquartile Range (IQR) karena akan memberikan hasil yang signifikan kepada model jika tidak dihapus. Outlier dapat menyebabkan model menjadi bias atau kurang akurat dalam prediksinya, terutama pada model-model yang sensitif terhadap nilai ekstrem.

#### Label Encoding
Untuk feature categorical seperti `country` dan `gender` akan diubah menjadi angka menggunakan `LabelEncoder()`.

#### Train-test Split Data
Dataset dibagi untuk fitur menjadi variabel `X` dan label (`churn`) menjadi variabel `y`. Untuk train dibagi menjadi 80% dan test 20%. Ditambahkan `stratify=y` pada `train_test_split` untuk memastikan proporsi kelas target (churn/tidak churn) tetap terjaga di dataset training dan testing, mengingat kemungkinan adanya ketidakseimbangan kelas.

### Data Modelling
Digunakan 5 model klasifikasi yaitu Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Support Vector Classifier (SVC), dan K-Nearest Neighbors Classifier untuk mencari tahu model klasifikasi terbaik.
Ditambahkan `StandardScaler` untuk penskalaan fitur numerik. Ini penting karena algoritma seperti Logistic Regression, SVC, dan KNN sensitif terhadap skala fitur.

##### Logistic Regression
Logistic Regression adalah algoritma klasifikasi linier yang digunakan untuk memprediksi probabilitas hasil biner. Model ini menggunakan fungsi logistik (sigmoid) untuk memetakan prediksi ke probabilitas antara 0 dan 1.
* **Kelebihan**: Sederhana, mudah diinterpretasikan, dan cepat untuk dilatih. Memberikan probabilitas kelas, bukan hanya prediksi kelas. Baik sebagai *baseline* kinerja.
* **Kekurangan**: Hanya mampu menangkap hubungan linier antar variabel. Asumsi independensi fitur. Dapat berkinerja buruk pada dataset dengan hubungan non-linier kompleks.

##### Decision Tree Classifier
Decision Tree Classifier adalah algoritma yang membagi data ke dalam kelompok berdasarkan fitur dengan aturan "if-then" hingga mencapai keputusan klasifikasi.
* **Kelebihan**: Mudah diinterpretasikan secara visual (jika pohon tidak terlalu dalam). Mampu menangani fitur numerik dan kategorikal. Dapat menangkap hubungan non-linier.
* **Kekurangan**: Sangat rentan terhadap overfitting. Kurang stabil; perubahan kecil pada data bisa menghasilkan struktur pohon yang sangat berbeda.

##### Random Forest Classifier
Random Forest Classifier adalah algoritma ensemble berbasis pohon keputusan. Algoritma ini membangun banyak pohon keputusan secara acak pada subset data, kemudian menggabungkan hasil prediksi dari masing-masing pohon (biasanya dengan voting mayoritas) untuk menghasilkan prediksi akhir.
* **Kelebihan**: Memberikan akurasi tinggi dan mampu menangani hubungan non-linier kompleks. Kurang rentan terhadap overfitting dibandingkan Decision Tree tunggal. Dapat menangani data dengan dimensi tinggi dan mengatasi masalah multikolinearitas. Mengurangi varians.
* **Kekurangan**: Lebih kompleks dan kurang interpretable dibandingkan model linier. Waktu komputasi bisa lebih tinggi, terutama dengan banyak pohon.

##### Support Vector Classifier (SVC)
Support Vector Classifier (SVC) adalah algoritma berbasis Support Vector Machine (SVM) yang digunakan untuk tugas klasifikasi. SVC berusaha menemukan hyperplane yang memiliki *margin* terbesar yang dapat memisahkan kelas-kelas data dengan jelas.
* **Kelebihan**: Efektif dalam ruang berdimensi tinggi. Mampu menangani hubungan non-linier melalui penggunaan kernel. Cukup kuat terhadap outlier karena berfokus pada *support vectors*.
* **Kekurangan**: Waktu komputasi bisa sangat tinggi untuk dataset besar. Interpretasi model lebih sulit dibandingkan model linier atau pohon. Pemilihan kernel dan parameter C, gamma sangat memengaruhi performa.

##### K-Nearest Neighbors (KNN) Classifier
K-Nearest Neighbors (KNN) Classifier adalah algoritma non-parametrik berbasis instance yang memprediksi kelas untuk data baru berdasarkan kelas mayoritas dari $k$ tetangga terdekatnya di ruang fitur.
* **Kelebihan**: Sederhana untuk dipahami dan diimplementasikan. Non-parametrik, cocok untuk data dengan distribusi kompleks tanpa asumsi tentang data.
* **Kekurangan**: Waktu prediksi bisa sangat lambat untuk dataset besar karena perlu menghitung jarak ke semua titik data pelatihan. Sensitif terhadap skala fitur dan dimensi tinggi (curse of dimensionality). Pemilihan nilai `k` sangat penting dan dapat memengaruhi kinerja secara signifikan.

### Evaluation
#### K-Fold Cross-Validation (Stratified)
* Pada evaluasi ini digunakan **5 fold Stratified Cross-Validation**, artinya dataset dibagi menjadi 5 subset.
* Setiap subset digunakan bergantian sebagai data uji, sementara subset lainnya digunakan sebagai data latih.
* **Stratified** berarti setiap fold memiliki proporsi kelas target (churn atau tidak churn) yang sama dengan proporsi di seluruh dataset. Ini penting untuk dataset yang tidak seimbang (imbalanced), seperti dataset churn, untuk memastikan setiap fold representatif.

##### Alasan Menggunakan Stratified K-Fold Cross Validation:
1.  **Evaluasi Konsisten**: Membagi data ke dalam beberapa lipatan memberikan evaluasi model yang lebih stabil dan robust, karena setiap data digunakan sebagai data latih dan uji. Ini memberikan estimasi kinerja yang lebih andal daripada hanya satu kali pembagian data.
2.  **Mengurangi Bias pada Kelas Imbalanced**: Dengan menjaga proporsi kelas target di setiap fold, Stratified K-Fold membantu mengurangi bias dan memastikan bahwa model dievaluasi secara adil pada kedua kelas (churn dan tidak churn), yang sangat penting pada dataset churn yang umumnya memiliki lebih sedikit kasus churn.
3.  **Generalisasi Model**: Stratified K-Fold Cross-Validation memberikan gambaran yang lebih baik tentang bagaimana model akan bekerja pada data baru yang belum pernah dilihat sebelumnya, karena model diuji pada beragam variasi data dengan distribusi kelas yang konsisten.

#### Metrik Evaluasi yang Digunakan:
Untuk masalah klasifikasi churn, beberapa metrik evaluasi penting adalah:
* **Accuracy:** Proporsi total prediksi yang benar.
* **Precision:** Proporsi prediksi positif yang sebenarnya positif (penting untuk meminimalkan *false positives* - memprediksi churn padahal tidak).
* **Recall (Sensitivity):** Proporsi positif aktual yang teridentifikasi dengan benar (penting untuk meminimalkan *false negatives* - tidak memprediksi churn padahal sebenarnya churn).
* **F1-Score:** Rata-rata harmonik dari Precision dan Recall, memberikan keseimbangan antara keduanya.
* **ROC AUC (Receiver Operating Characteristic - Area Under the Curve):** Mengukur kemampuan model untuk membedakan antara kelas positif dan negatif di berbagai ambang batas klasifikasi. Nilai mendekati 1 menunjukkan kinerja yang sangat baik.

#### Dampak Model terhadap Business Understanding

##### Apakah Model Menjawab Problem Statement?
* **Ya**, model klasifikasi yang dikembangkan secara langsung bertujuan untuk memprediksi probabilitas atau kelas churn (0: tidak churn, 1: churn) berdasarkan data historis nasabah.
* **Ya**, dengan melatih model seperti Random Forest Classifier atau Decision Tree Classifier, dapat menganalisis fitur importance (pentingnya fitur) yang menunjukkan seberapa besar kontribusi setiap fitur dalam memprediksi churn.

##### Apakah Model Berhasil Mencapai Goals?
* **Ya**, proses ini membangun dan mengevaluasi beberapa model klasifikasi dengan metrik seperti Accuracy, Precision, Recall, F1-Score, dan ROC AUC. Tujuannya adalah untuk menemukan model yang paling akronim dan efektif dalam mengidentifikasi nasabah yang berisiko tinggi churn.
* **Ya**, melalui EDA dan analisis fitur importance dari model yang dipilih (terutama model berbasis pohon), dapat memahami hubungan antara karakteristik nasabah dan perilakunya yang mengarah pada churn.
* **Ya**, model prediksi churn akan memungkinkan bank untuk mengidentifikasi nasabah yang berpotensi churn sebelum mereka benar-benar pergi, mengembangkan strategi retensi yang dipersonalisasi, dan mengalokasikan sumber daya retensi secara lebih efisien.

##### Apakah Solusi yang Direncanakan Berdampak?
* **Ya**, solusi ini berpotensi memiliki dampak positif yang signifikan pada bisnis bank melalui peningkatan retensi pelanggan, pengurangan biaya akuisisi pelanggan baru, peningkatan pendapatan, dan pemahaman bisnis yang lebih baik.

### Analisis Model Terbaik
Berdasarkan perbandingan rata-rata ROC AUC pada test set, model **Random Forest Classifier** menunjukkan kinerja terbaik.

| Model | Rata-rata Test ROC AUC | Rata-rata Test Accuracy | Rata-rata Test Precision | Rata-rata Test Recall | Rata-rata Test F1-Score |
|---|---|---|---|---|---|
| Random Forest Classifier | 0.8415 | 0.8529 | 0.7573 | 0.4269 | 0.5463 |

Terlihat bahwa *Accuracy* dan *Precision* dari Random Forest Classifier sangat tinggi di antara model lainnya, menandakan model ini sangat baik dalam memprediksi "tidak churn" dan saat memprediksi "churn", prediksinya cenderung benar. Namun, *Recall* dan *F1-Score* yang relatif lebih rendah dibandingkan *Precision* menunjukkan bahwa model ini masih melewatkan beberapa kasus churn yang sebenarnya. Adanya perbedaan signifikan antara metrik train dan test untuk Random Forest Classifier (misalnya Accuracy 1.0000 vs 0.8529) menunjukkan adanya indikasi *overfitting* pada data latih, namun *generalization performance* pada data test masih sangat baik.

#### Classification Report (Random Forest Classifier pada Test Set Penuh)

```
              precision    recall  f1-score   support

           0       0.88      0.95      0.91      1529
           1       0.75      0.43      0.55       385

    accuracy                           0.86      1914
   macro avg       0.82      0.69      0.73      1914
weighted avg       0.85      0.86      0.85      1914
```

* **Kelas 0 (Tidak Churn):**
    * **Precision (0.88):** Dari semua nasabah yang diprediksi tidak churn, 88% di antaranya benar-benar tidak churn.
    * **Recall (0.95):** Dari semua nasabah yang sebenarnya tidak churn, 95% di antaranya berhasil diidentifikasi dengan benar oleh model.
    * **F1-Score (0.91):** Menunjukkan keseimbangan yang sangat baik antara precision dan recall untuk kelas "tidak churn".

* **Kelas 1 (Churn):**
    * **Precision (0.75):** Dari semua nasabah yang diprediksi churn, 75% di antaranya benar-benar churn.
    * **Recall (0.43):** Dari semua nasabah yang sebenarnya churn, hanya 43% yang berhasil diidentifikasi oleh model.
    * **F1-Score (0.55):** Merupakan rata-rata harmonik dari Precision dan Recall untuk kelas churn.

* **Accuracy (0.86):** Akurasi keseluruhan model adalah 86%.

#### Confusion Matrix (Random Forest Classifier pada Test Set Penuh)

```
       Predicted Label
       Not Churn  Churn
True   Not Churn  1450      79
Label  Churn       220      165
```

* **True Negative (TN = 1450):** 1450 nasabah yang sebenarnya tidak churn berhasil diprediksi dengan benar sebagai tidak churn.
* **False Positive (FP = 79):** 79 nasabah yang sebenarnya tidak churn diprediksi salah sebagai churn.
* **False Negative (FN = 220):** 220 nasabah yang sebenarnya churn diprediksi salah sebagai tidak churn.
* **True Positive (TP = 165):** 165 nasabah yang sebenarnya churn berhasil diprediksi dengan benar sebagai churn.

Secara keseluruhan, Random Forest Classifier menunjukkan kinerja yang sangat kuat untuk memprediksi churn nasabah. Meskipun Recall untuk kelas churn masih bisa ditingkatkan, Precision yang tinggi dan ROC AUC yang sangat baik menjadikan model ini alat yang sangat berharga bagi bank untuk mengidentifikasi nasabah berisiko tinggi dan merancang strategi retensi yang proaktif.

## Kesimpulan
Berdasarkan hasil di atas, dapat dikatakan bahwa model mampu memprediksi churn nasabah selaras dengan hasil akurasi menggunakan algoritma Random Forest Classifier sebesar 85.29% dan ROC AUC sebesar 0.8415. Meskipun terdapat indikasi overfitting pada data latih, kinerja generalisasi model ini tetap sangat baik.

## Referensi
1. Dbs_machinelearningterapan_marselloormanda_submission1 (6).py
```
