# E-Sports-Biometrics-Performance-DATASET-Machine-Learning

## 1. Deskripsi Proyek

Proyek ini membahas penerapan Machine Learning untuk menganalisis data biometrik dan aktivitas gaming pada pemain e-sports. Fokus pengembangan adalah mempelajari pemilihan algoritma regresi berdasarkan karakteristik data, ukuran dataset, dan linearitas hubungan antarvariabel.

Proyek ini juga membahas evaluasi model menggunakan Mean Absolute Error (MAE), Mean Squared Error (MSE), dan Coefficient of Determination (R²) untuk membandingkan hasil prediksi beberapa algoritma regresi.

## 2. Dataset

Dataset utama yang digunakan adalah `esports_gaming_biometrics_250k.csv`, dengan `data_dictionary.csv` sebagai referensi untuk memahami kolom dan karakteristik data.

Beberapa variabel yang dipelajari meliputi:

- `Player_ID`: identitas pemain.
- `Age`: usia pemain.
- `Game_Genre`: genre permainan.
- `PlayTime_Hours`: durasi bermain.
- `Sleep_Hours`: durasi tidur.
- `APM`: aktivitas per menit dalam permainan.
- `Heart_Rate_BPM`: denyut jantung per menit.
- `HRV_Score`: skor variabilitas detak jantung.

Definisi dan tipe setiap variabel perlu diperiksa kembali melalui data dictionary sebelum digunakan dalam pemodelan.

## 3. Landasan Teori

### 3.1 Machine Learning

Machine Learning merupakan pendekatan komputasi yang memungkinkan sistem mempelajari pola dari data untuk membuat prediksi atau mengambil keputusan tanpa harus menentukan seluruh aturan secara manual.
Dalam proyek ini, Machine Learning digunakan untuk mempelajari hubungan antara karakteristik pemain, aktivitas gaming, dan variabel biometrik yang tersedia dalam dataset.

### 3.2 Regresi dalam Machine Learning

Regresi merupakan metode supervised learning yang digunakan untuk memprediksi nilai numerik berdasarkan satu atau lebih variabel input.
Secara umum, model regresi dapat dituliskan sebagai:

**Y = f(X) + ε**

Keterangan:

- Y: variabel target yang ingin diprediksi.
- X: satu atau lebih variabel input.
- f(X): hubungan yang dipelajari model.
- ε: komponen kesalahan atau variasi yang tidak dijelaskan model.

Dalam konteks e-sports, regresi dapat digunakan untuk memprediksi variabel numerik seperti `HRV_Score`, `Heart_Rate_BPM`, atau `PlayTime_Hours`, selama target tersebut sesuai dengan tujuan penelitian.

### 3.3 Faktor Pemilihan Algoritma Regresi

Pemilihan algoritma tidak hanya ditentukan oleh tingkat akurasi, tetapi juga dipengaruhi oleh tiga faktor utama.

**A. Jenis dan karakteristik data**

Jenis data mencakup data numerik, kategorikal, distribusi nilai, pencilan, korelasi antarfitur, dan jumlah variabel. Data kategorikal perlu diubah menjadi representasi numerik jika algoritma yang digunakan tidak dapat memprosesnya secara langsung.

**B. Ukuran dataset**

Jumlah sampel dan fitur memengaruhi waktu pelatihan, kebutuhan memori, dan kompleksitas model. Model linear sering menjadi baseline yang efisien, sedangkan model ensemble dapat dipertimbangkan ketika pola data lebih kompleks. Namun, ukuran dataset saja tidak cukup untuk menentukan algoritma terbaik.

**C. Linearitas hubungan antarvariabel**

Linearitas menunjukkan apakah hubungan antara fitur dan target dapat didekati dengan fungsi linear. Jika hubungan cenderung linear, Linear Regression dapat menjadi pilihan awal. Jika pola lebih kompleks atau nonlinier, algoritma seperti Random Forest Regression dan Gradient Boosting Regression dapat diuji.

Selain ketiga faktor tersebut, pemilihan model juga perlu mempertimbangkan interpretabilitas, kecepatan prediksi, dan kebutuhan aplikasi.

### 3.4 Algoritma Regresi

| Algoritma | Karakteristik dan kegunaan |
|---|---|
| Linear Regression | Model dasar untuk hubungan linear dan interpretasi koefisien. |
| Ridge Regression | Regresi linear dengan regularisasi L2 untuk mengurangi besarnya koefisien. |
| Lasso Regression | Regresi dengan regularisasi L1 yang dapat membuat sebagian koefisien menjadi nol. |
| Elastic Net | Menggabungkan regularisasi L1 dan L2. |
| Least Angle Regression (LARS) | Metode regresi linear yang dapat digunakan dalam pemodelan dengan banyak fitur. |
| Polynomial Regression | Memodelkan hubungan melengkung menggunakan fitur polinomial. |
| Decision Tree Regression | Menangkap pola nonlinier menggunakan aturan berbentuk pohon. |
| Random Forest Regression | Menggabungkan banyak pohon keputusan untuk menghasilkan prediksi yang lebih stabil. |
| Gradient Boosting Regression | Membangun model secara bertahap untuk memperbaiki kesalahan prediksi. |
| XGBoost Regression | Implementasi gradient boosting yang dirancang untuk pemodelan yang efisien dan fleksibel. |
| Support Vector Regression (SVR) | Menggunakan pendekatan support vector dan kernel untuk memodelkan hubungan linear maupun nonlinier. |
| K-Nearest Neighbors Regression | Menghasilkan prediksi berdasarkan target dari sampel yang berdekatan. |
| Neural Network Regression | Mempelajari hubungan kompleks menggunakan jaringan neuron. |

Tidak semua algoritma harus digunakan sekaligus. Kandidat model dipilih berdasarkan karakteristik dataset, sumber daya komputasi, dan tujuan prediksi.

### 3.5 Evaluasi Model Regresi

Evaluasi dilakukan untuk mengukur perbedaan antara nilai aktual dan hasil prediksi.

**A. Mean Absolute Error (MAE)**

MAE menghitung rata-rata nilai absolut selisih prediksi dan nilai aktual.
MAE yang lebih rendah menunjukkan kesalahan absolut yang lebih kecil. Nilainya bergantung pada skala target dan tidak harus kurang dari satu.

**B. Mean Squared Error (MSE)**

MSE menghitung rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual.
MSE yang lebih rendah menunjukkan kesalahan kuadrat yang lebih kecil. Metrik ini memberikan penalti lebih besar terhadap kesalahan yang besar dan memiliki satuan kuadrat dari target.

**C. Coefficient of Determination (R²)**

R² mengukur kemampuan model dalam menjelaskan variasi target berdasarkan data evaluasi.

Interpretasi umum:

- R² = 1: prediksi sempurna pada data evaluasi.
- R² mendekati 1: model menjelaskan variasi target dengan baik.
- R² = 0: performa setara dengan baseline rata-rata target pada data evaluasi.
- R² kurang dari 0: performa lebih buruk daripada baseline tersebut.

Pada perhitungan R² standar, nilai tidak harus berada di antara nol dan satu karena R² dapat bernilai negatif. Nilai R² lebih dari satu tidak diharapkan pada evaluasi standar yang valid.

Ketiga metrik perlu dianalisis bersama karena R² yang tinggi tidak selalu berarti kesalahan prediksi kecil atau model mampu melakukan generalisasi dengan baik.

## 4. Metodologi Penelitian

Tahapan penelitian yang direncanakan adalah:

1. **Data Understanding:** memahami struktur dataset dan definisi variabel melalui data dictionary.
2. **Data Cleaning:** memeriksa nilai hilang, duplikasi, tipe data, dan pencilan.
3. **Exploratory Data Analysis (EDA):** menganalisis distribusi, korelasi, dan pola hubungan antarvariabel.
4. **Feature Selection:** menentukan fitur yang relevan dan menghindari kebocoran informasi target.
5. **Data Splitting:** memisahkan data training dan testing.
6. **Preprocessing:** melakukan encoding dan scaling jika dibutuhkan oleh algoritma.
7. **Model Training:** melatih algoritma regresi yang dipilih.
8. **Model Evaluation:** membandingkan MAE, MSE, R², waktu pelatihan, dan waktu prediksi.
9. **Model Comparison:** menentukan model yang paling sesuai berdasarkan hasil evaluasi dan kebutuhan penelitian.

Jika proses pemodelan menggunakan data yang memiliki urutan waktu, pembagian data harus mempertimbangkan urutan tersebut agar informasi masa depan tidak bocor ke data pelatihan.

## 5. Rencana Eksperimen

Eksperimen dirancang untuk menjawab pertanyaan berikut:

1. Bagaimana karakteristik dataset memengaruhi pemilihan algoritma regresi?
2. Bagaimana perbedaan performa model linear dan nonlinier?
3. Algoritma mana yang menghasilkan MAE dan MSE terendah serta R² terbaik?
4. Bagaimana ukuran dataset memengaruhi waktu pelatihan dan prediksi?
5. Algoritma mana yang paling sesuai dengan kebutuhan interpretasi dan prediksi pada data e-sports?

Target prediksi harus ditentukan sebelum eksperimen dilakukan. Pemilihan target mempertimbangkan tujuan penelitian, definisi variabel, dan ketersediaan informasi yang relevan.

## 6. Landasan Penelitian Terdahulu

### 6.1 Zebari (2022)

**Judul:** *Systematic Review of Regression Algorithms for Predictive Analytics.*
Penelitian ini dijadikan referensi awal untuk mempelajari berbagai algoritma regresi dan penggunaannya dalam predictive analytics.
Relevansi dengan proyek: membantu menyusun kandidat algoritma yang akan dibandingkan pada dataset e-sports.
DOI: https://doi.org/10.48161/qtj.v1n4a17

### 6.2 Uddin dan Lu (2024)

**Judul:** *Dataset meta-level and statistical features affect machine learning performance.*
Penelitian ini membahas hubungan antara karakteristik statistik dataset dan performa algoritma Machine Learning.
Relevansi dengan proyek: mendukung pentingnya menganalisis ukuran sampel, jumlah fitur, dan karakteristik statistik sebelum menentukan algoritma.
DOI: https://doi.org/10.1038/s41598-024-51825-x

### 6.3 Bendersky dan Fridkin (2026)

**Judul:** *The Evolving Landscape of Modern Regression: A Systematic Review and Mapping Study of Penalized, Algorithmic, and Interpretable Models.*
Penelitian ini digunakan sebagai rujukan mengenai perkembangan pendekatan regresi modern, termasuk regularisasi, metode algoritmik, dan interpretabilitas.
Relevansi dengan proyek: memberikan kerangka untuk membandingkan model linear dan metode yang lebih kompleks berdasarkan kebutuhan pemodelan.
DOI: https://doi.org/10.1002/widm.70078

Catatan: identitas bibliografi, isi, dan relevansi setiap artikel harus diverifikasi melalui sumber penerbit sebelum digunakan sebagai dasar klaim spesifik dalam laporan akademik.

## 7. Dokumentasi Tambahan

Dokumentasi resmi scikit-learn untuk evaluasi model:
https://scikit-learn.org/stable/modules/model_evaluation.html
Dokumentasi tersebut menjelaskan berbagai metrik evaluasi, termasuk MAE, MSE, dan R².

## 8. Kesimpulan

Pemilihan algoritma regresi perlu mempertimbangkan jenis dan karakteristik data, ukuran dataset, serta linearitas hubungan antarvariabel. Linear Regression dapat menjadi baseline untuk hubungan linear, sedangkan Ridge, Lasso, dan Elastic Net dapat dipertimbangkan untuk regularisasi. Random Forest, Gradient Boosting, dan XGBoost dapat diuji ketika data memiliki pola nonlinier yang lebih kompleks.

Model terbaik tidak dapat ditentukan hanya berdasarkan teori. Penentuan akhir harus dilakukan melalui eksperimen yang konsisten dengan mengevaluasi MAE, MSE, R², kebutuhan komputasi, dan kemampuan generalisasi pada data yang tidak digunakan selama pelatihan.

## 9. Status Proyek

Dokumentasi ini merupakan landasan teori dan rencana eksperimen. Klaim tentang algoritma terbaik akan ditambahkan setelah pengujian pada dataset dilakukan dan hasil evaluasi tersedia.
