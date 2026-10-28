# Laporan Proyek: Layanan Named-Entity Recognition (NER) Artikel Berita
**Ujian Tengah Semester (UTS) Data Science**

Layanan ekstraksi entitas bernama (_Named-Entity Recognition_) berbasis web untuk memproses artikel berita daring. Layanan ini menerima judul (*title*) dan isi berita (*body*), menggabungkan keduanya untuk mempertahankan konteks semantik dan anafora, mengekstrak entitas beserta batas karakternya (_character offsets_), menyajikan estimasi skor keyakinan (_confidence scores_), menormalisasi nama entitas ke bentuk kanonikal (_canonical normalization_), serta menyajikan hasilnya melalui REST API modern dan antarmuka web interaktif yang responsif.

---

## Latar Belakang dan Tujuan

### Latar Belakang
Dalam industri pemantauan media (*media intelligence* / *news monitoring*), tim analis dituntut untuk menyaring ratusan hingga ribuan artikel berita daring setiap hari guna mengidentifikasi penyebutan entitas penting seperti lembaga perbankan, tokoh korporat/politik, wilayah geografis, maupun regulasi pemerintah. Pemantauan naskah berita secara manual memerlukan waktu dan sumber daya yang besar serta rentan terhadap inkonsistensi manusia (*human error*).

Oleh karena itu, diperlukan sebuah purwarupa fungsional (*working prototype*) layanan cerdas **Named-Entity Recognition (NER)** yang dapat mengekstrak dan mengklasifikasikan entitas bernama secara otomatis, akurat, dan deterministik dari masukan judul (*title*) dan isi berita (*body*). Layanan ini dirancang khusus untuk memproses karakteristik naskah berita perbankan dan finansial regional (seperti kliping media online perbankan Malaysia dan ASEAN), baik pada artikel berita standar maupun naskah dengan kasus batas yang menantang (*challenging edge cases*).

### Tujuan Proyek
Mengacu pada sasaran pembelajaran dalam dokumen studi kasus (*Problem Set: Article Named-Entity Recognition Service*), tujuan yang dicapai dalam pengembangan proyek ini adalah:
1. **Formulasi Masalah sebagai Layanan Web (*Service Formulation*)**: Membangun arsitektur layanan mandiri dengan kontrak input-output JSON yang baku, deterministik, serta dilengkapi validasi ketat (*HTTP 400 Bad Request* untuk input invalid).
2. **Penggabungan Teks yang Aman (*Safe Text Preprocessing*)**: Menggabungkan judul (*title*) dan tubuh artikel (*body*) secara konsisten untuk menjaga keutuhan rantai koreferensi, konteks anafora, dan resolusi singkatan dalam satu sistem koordinat offset karakter tunggal.
3. **Penerapan & Adaptasi Model NLP (*Model Adaptation*)**: Mengimplementasikan model statistik pra-latih (*baseline*) dan mengembangkannya menjadi model terkalibrasi (*adapted*) menggunakan aturan leksikal domain perbankan (*EntityRuler*) serta segmenter multibahasa (Melayu dan Mandarin Hanzi).
4. **Pembedaan Komponen Entitas yang Presisi**: Membedakan secara tegas antara teks entitas harfiah (*text*), kategori entitas (*label*), dan lokasi karakter eksak (*start* dan *end offsets*).
5. **Penyediaan Antarmuka & REST API**: Merancang antarmuka web interaktif yang intuitif dengan visualisasi penyorotan teks (*highlighting*), tabel hasil ekstraksi, dan indikator status pemrosesan (*loading state*), didukung endpoint REST API berbasis FastAPI.
6. **Evaluasi Kuantitatif Terukur**: Mengukur kualitas ekstraksi menggunakan metrik *Precision*, *Recall*, dan *F1-Score* berbasis pencocokan rentang persis (*exact-span matching*) terhadap dataset acuan berlabel (*ground-truth benchmark*).
7. **Komunikasi Keterbatasan, Analisis Galat, & Pertimbangan Etis**: Mengidentifikasi pola galat sistem (*error analysis*), mendokumentasikan batasan operasional, serta menjawab pertimbangan etika kecerdasan buatan, privasi (PII), hak cipta (*fair use*), dan mitigasi bias representasi bahasa.

---

## 1. Arsitektur dan Komponen Sistem

Sistem dirancang secara modular dengan memisahkan antarmuka pengguna (_presentation layer_), pengatur lalu lintas permintaan (_API routing & validation_), serta lapisan pemrosesan NLP cerdas (_inference & adaptation layer_).

### Diagram Arsitektur Layanan

```mermaid
flowchart TD
    subgraph ClientLayer ["1. Lapisan Klien & Konsumen"]
        UI["Antarmuka Web (HTML/CSS/JS)<br/>src/static/"]
        CLI["Pengujian Otomatis / pytest / cURL"]
    end

    subgraph APILayer ["2. Lapisan Layanan API (FastAPI - src/app.py)"]
        EndpointSingle["Endpoint: POST /extract-entities<br/>Payload: title, body, model"]
        EndpointBatch["Endpoint: POST /extract-entities/batch<br/>Payload: articles"]
        Validator{"Validasi Pydantic<br/>Tipe string, non-empty<br/>Maks: Title 1000 char, Body 50000 char"}
        ErrResponse["HTTP 400 Bad Request<br/>Pesan error & validasi"]
        Combiner["Penggabung Naskah<br/>combined_text = title + body<br/>Offset karakter tunggal"]
    end

    subgraph NLPLayer ["3. Lapisan Pemrosesan NLP (src/ner_model.py)"]
        Router{"Router Bahasa & Model"}
        Ruler["EntityRuler (Kamus Domain & Pola Finansial)"]
        StatNER1["Statistical Neural NER<br/>spaCy en_core_web_sm"]
        Segmenter["pkuseg Segmenter + zh_core_web_sm<br/>Multibahasa Mandarin/Hanzi"]
        StatNER2["Baseline Statistical NER<br/>spaCy en_core_web_sm murni"]
        PostProc["Post-Processing Pipeline<br/>Pemetaan Label, Confidence Scorer<br/>Normalisasi Kanonikal & Deduplikasi Span"]
    end

    subgraph OutputLayer ["4. Format Respons JSON (HTTP 200)"]
        Response["Respons Terstruktur<br/>entities: text, label, start, end, confidence, normalized<br/>meta: entity_count, model, language"]
    end

    UI -->|Kirim Permintaan| EndpointSingle
    CLI -->|Kirim Permintaan| EndpointSingle
    CLI -->|Kirim Batch| EndpointBatch
    EndpointSingle --> Validator
    EndpointBatch --> Validator
    Validator -->|Gagal Validasi| ErrResponse
    Validator -->|Valid| Combiner
    Combiner --> Router

    Router -->|Bahasa Latin dan Model Adapted| Ruler
    Ruler --> StatNER1
    StatNER1 --> PostProc

    Router -->|Aksara Mandarin zh| Segmenter
    Segmenter --> PostProc

    Router -->|Model Baseline| StatNER2
    StatNER2 --> PostProc

    PostProc --> Response
    Response -->|Render Highlight dan Tabel| UI
    Response -->|Verifikasi Assert JSON| CLI
```

### Rincian Komponen Teknis

1. **Backend REST API (`src/app.py`)**: Layanan berbasis framework berkecepatan tinggi FastAPI. Menyediakan endpoint tunggal `POST /extract-entities` dan endpoint pemrosesan massal `POST /extract-entities/batch`. Dilengkapi middleware validasi Pydantic yang menjamin penolakan input kosong, tipe data tidak valid, input bertipe non-string, serta teks yang melebihi batas panjang dengan status **`HTTP 400 Bad Request`**.
2. **Pipeline NLP Domain-Adapted (`src/ner_model.py`)**:
   - Memadukan pendekatan *Hybrid NLP*: aturan leksikal berbasis domain (*domain dictionary patterns*) melalui `EntityRuler` yang dieksekusi sebelum model statistik saraf (*statistical neural NER*) spaCy `en_core_web_sm`.
   - Modul multibahasa terintegrasi untuk teks beraksara non-Latin Hanzi/Mandarin menggunakan tokenizer segmen berbasis C++ (`spacy-pkuseg`) dan model `zh_core_web_sm`.
   - Modul normalisasi kanonikal entitas (*canonical entity normalizer*) dan penilai skor keyakinan (*confidence scoring*).
3. **Antarmuka Web Interaktif (`src/static/`)**:
   - Antarmuka web modern bergaya netral gelap (_dark theme_), dibangun murni menggunakan Vanilla HTML5, CSS3 kustom, dan Vanilla JavaScript tanpa dependensi eksternal yang memberatkan.
   - Dilengkapi penghitung karakter *real-time*, tombol *preset* contoh artikel uji, status visual saat pemrosesan (*loading spinner* & tombol nonaktif), penampil teks dengan penyorotan warna entitas interaktif (*visual text highlighter*), tabel hasil ekstraksi, dan penampil format mentah JSON.
4. **Modul Pengujian & Evaluasi (`tests/test_api.py`, `src/evaluate.py`, `src/data_profiler.py`)**:
   - 14 unit test otomatis berbasis `pytest` yang memverifikasi seluruh skenario uji fungsional, tipe data, *edge cases*, determinisme, dan penanganan kesalahan.
   - Skrip evaluasi kuantitatif yang menghitung metrik formal (*Precision*, *Recall*, *F1-score*, *Confusion Matrix*, dan *Per-Label Support*) dengan perbandingan model `baseline` vs `adapted`.
   - Skrip pemrofil data (*data profiler*) untuk mengaudit dataset masukan CSV.

### Tampilan Antarmuka Panel Pengguna

Dokumentasi visual antarmuka web dan seluruh halaman/fitur ekstraksi entitas:

#### 1. Dasbor Utama (*Initial State / Standby*)
Tampilan awal saat aplikasi dimuat, menyediakan formulir masukan judul dan tubuh berita, penghitung karakter langsung (*live character counter*), selektor varian model (Adapted vs Baseline), serta tombol pintas berita contoh (*sample presets*).

![Dasbor Utama Input Artikel](screenshots/01_initial_dashboard.png)

#### 2. Visualisasi Penyorotan Teks (*Text Highlights View*)
Menampilkan hasil ekstraksi entitas langsung pada naskah berita dengan penyorotan warna unik per kategori label, lencana entitas (*badge tag*), ringkasan frekuensi kategori (*category chips*), dan label bahasa yang terdeteksi secara otomatis.

![Visualisasi Penyorotan Teks Entitas](screenshots/02_text_highlights.png)

#### 3. Tampilan Tabel Terstruktur (*Table View*)
Menyajikan daftar seluruh entitas yang terdeteksi dalam format tabel rapi, memuat teks entitas asli (*original entity*), kategori label semantik, dan koordinat offset karakter eksak (*start & end character offsets*).

![Tampilan Tabel Entitas](screenshots/03_table_view.png)

#### 4. Keluaran Data JSON API (*JSON Output View*)
Menampilkan representasi muatan JSON terstruktur yang dihasilkan langsung oleh layanan backend, lengkap dengan estimasi skor keyakinan (*confidence score*), normalisasi nama kanonikal (*canonical entity*), dan metadata inferensi model.

![Keluaran JSON Respons API](screenshots/04_json_output.png)

#### 5. Pemrosesan Multibahasa Berita Mandarin (*Chinese Multilingual Extraction*)
Kemampuan pipeline khusus aksara Hanzi dalam memproses dan mengekstrak entitas perbankan serta lokasi regional dari naskah berita Mandarin tanpa spasi kata.

![Ekstraksi Berita Mandarin](screenshots/05_chinese_extraction.png)

#### 6. Dokumentasi Interaktif API (*FastAPI Swagger UI*)
Antarmuka dokumentasi OpenAPI interaktif yang dapat diakses langsung melalui rute `/docs` untuk melakukan pengujian interaktif terhadap endpoint tunggal (`/extract-entities`) maupun pemrosesan massal (`/extract-entities/batch`).

![Dokumentasi Interaktif Swagger UI](screenshots/06_api_docs_swagger.png)

---

## 2. Pemrofilan Dataset Awal (Provided Data Profiling)

Sesuai instruksi pada berkas studi kasus (*Problem Set Page 2*), mahasiswa diwajibkan untuk memuat seluruh berkas CSV dari direktori `use_cases/pnm_study_case/sample/`, memelihara tanda petik teks dengan benar (_quoted text_), mengaudit jumlah baris, mendeteksi duplikasi, mendata kolom kosong (_missing values_), memetakan sebaran bahasa, serta menginspeksi kolom `entities`.

Pemrofilan dataset dijalankan secara otomatis melalui skrip `src/data_profiler.py`:

```bash
python src/data_profiler.py
```

### Hasil Audit dan Pemrofilan Dataset

| Indikator Audit | Nilai Temuan | Keterangan Teknis |
| :--- | :---: | :--- |
| **Total Berkas CSV Dimuat** | **9 berkas** | Format CSV dipisahkan titik koma (`;`) dengan pemisah baris LF/CRLF standar. |
| **Total Baris Dimuat (_Rows Loaded_)** | **1.451 baris** | Seluruh baris terbaca tanpa baris rusak (_malformed line skipping_ = 0). |
| **Artikel Unik (`original_id`)** | **846 artikel** | Dihitung berdasarkan nilai pengenal unik kolom `original_id`. |
| **Baris Duplikat (_Duplicate Rows_)** | **605 baris** | Terjadi akibat pemantauan multikata kunci (misal: berita yang sama memuat CIMB dan Maybank masuk ke dua file kliping berbeda). |
| **Judul Kosong (_Missing Titles_)** | **0 baris (0,0%)** | 100% judul berita terisi lengkap dan valid. |
| **Isi Berita Kosong (_Missing Bodies_)** | **0 baris (0,0%)** | 100% teks tubuh berita terisi lengkap. |

### Rincian Per-Berkas CSV Masukan

| Nama Berkas CSV | Jumlah Baris | Jumlah Kolom | Subjek Pemantauan Kliping |
| :--- | :---: | :---: | :--- |
| `..._-nwub4cROX.csv` | 29 | 35 | Maybank Foundation CSR |
| `..._4sxH8Dz0hr.csv` | 2 | 35 | Public Bank Berhad |
| `..._58yzyiBQ9U.csv` | 105 | 35 | RHB Bank |
| `..._9xtqAsPitG.csv` | 96 | 35 | Hong Leong Bank |
| `..._bRQ9afwSIP.csv` | 347 | 35 | CIMB Group Malaysia |
| `..._hMxhx9QLTq.csv` | 534 | 35 | Maybank Group |
| `..._i3BJEnT77K (1).csv` | 160 | 35 | Maybank CSR (Salinan Kliping 1) |
| `..._i3BJEnT77K.csv` | 160 | 35 | Maybank CSR (Salinan Kliping 2) |
| `..._WdjZl0bRfZ.csv` | 18 | 35 | Malaysia Debt Ventures (MDV) |
| **Total** | **1.451** | **35** | **9 Berkas Kliping Media** |

### Distribusi Bahasa Dokumen

Berdasarkan analisis kolom `language` pada 1.451 baris data:
- **Bahasa Inggris (`en`)**: 1.215 artikel (83,74%)
- **Bahasa Melayu (`ms`)**: 181 artikel (12,47%)
- **Bahasa Mandarin / Hanzi (`zh`)**: 38 artikel (2,62%)
- **Bahasa Indonesia (`id`)**: 15 artikel (1,03%)
- **Bahasa Vietnam (`vi`)**: 1 artikel (0,07%)
- **Bahasa Spanyol (`es`)**: 1 artikel (0,07%)

### Hasil Inspeksi Kritis Kolom `entities`
Sesuai arahan tugas: *"The entities column is not automatically a gold-standard NER annotation. Students must inspect its format and provenance."*
- **Temuan**: Setelah dilakukan iterasi menyeluruh pada seluruh 1.451 baris data di ke-9 file CSV, kolom `entities` berstatus **100% kosong (NULL/NaN, 0 baris bernilai non-null)**.
- **Konsekuensi & Keputusan Ilmiah**: Karena kolom bawaan tersebut sama sekali tidak memuat anotasi entitas, koordinat batas karakter (*span offset*), maupun label kategori, maka kolom tersebut **tidak dapat digunakan sebagai ground truth**.
- **Solusi**: Peneliti menyusun dataset anotasi acuan independen (*gold-standard ground truth benchmark*) secara cermat dengan mengurasi artikel sampel dan memetakan rentang karakter entitas secara terverifikasi manusia (*human-reviewed*) di dalam `data/evaluation_examples.json`.

---

## 3. Metodologi Anotasi, Pembagian Data, dan Pemilihan Model

### Skema Anotasi & Standar Ground Truth
Dataset acuan disusun dalam format terstruktur `data/evaluation_examples.json` berisi 12 dokumen acuan (mencakup teks panjang, teks pendek, variasi bahasa, dan kasus kontrol negatif) dengan total **61 entitas ground truth**. Anotasi merekam:
- Identitas pengenal artikel asal (`id` dan `source_reference`)
- Judul (`title`) dan Tubuh artikel (`body`)
- Naskah gabungan (`combined_text`)
- Daftar entitas mencakup teks persis (`text`), label kategori (`label`), indeks karakter awal inklusif (`start`), dan indeks karakter akhir eksklusif (`end`).

### Strategi Pembagian Data (*Data Split*)
Sesuai pedoman tugas (*Problem Set Page 4 & 5*):
1. **Unit Pembagian**: Pembagian dilakukan berbasis **identitas artikel utuh (_article identity_)**, bukan berbasis kalimat atau baris parsial. Hal ini menjamin tidak ada fragmen teks atau entitas dari artikel yang sama muncul sekaligus di set latih dan set uji (*zero data leakage*).
2. **Proporsi Pembagian**:
   - **70% Training / Calibration Set**: Digunakan untuk mengidentifikasi pola kosakata entitas finansial (kamus nama bank, akronim bursa, nama menteri, mata uang, pola alamat wilayah).
   - **15% Validation Set**: Digunakan untuk menetapkan ambang batas skor keyakinan (*confidence threshold*), memverifikasi pemetaan label (*LABEL_MAPPING*), serta memilih pustaka segmenter.
   - **15% Held-Out Test Set**: Data uji murni yang diisolasi (*untouched*) dan baru dievaluasi pada tahap akhir penulisan laporan.
3. **Random Seed Deterministik**: Seluruh proses pembagian dan eksperimen dikunci menggunakan seed deterministik `seed=42`.
4. **Pernyataan Integritas Data Uji**: *Held-out test examples* dipastikan tidak pernah mempengaruhi pembentukan aturan *EntityRuler*, penulisan kamus normalisasi, maupun penyesuaian parameter pipa (*no test set contamination*).

### Distribusi Entitas & Penanganan Ketidakseimbangan Kelas (*Class Imbalance*)
Total 61 entitas ground truth terdistribusi ke dalam 6 kategori:
- **`ORG` (Organization)**: 29 entitas (47,54%) — *Mayoritas karena domain perbankan & korporasi.*
- **`GPE` (Geopolitical Entity)**: 14 entitas (22,95%) — *Negara, kota, dan negara bagian.*
- **`PERSON` (Individual)**: 8 entitas (13,11%) — *Nama tokoh, menteri, eksekutif.*
- **`DATE` (Date/Time)**: 7 entitas (11,48%) — *Tanggal pelaksanaan, bulan, tahun.*
- **`LOC` (Non-GPE Location)**: 2 entitas (3,28%) — *Fasilitas olahraga, stadion.*
- **`EVENT` (Named Event)**: 1 entitas (1,64%) — *Hari raya keagamaan.*

Ketidakseimbangan kelas (*class imbalance*) ditangani dengan menetapkan metrik terpisah per-kategori label (*per-label precision, recall, F1*) di samping metrik agregat makro, sehingga kategori minoritas seperti `EVENT` dan `LOC` tetap terpantau performanya tanpa tertutup dominasi `ORG`.

### Rasionalisasi Pemilihan Model NLP
1. **Model Baseline**: Menggunakan model pra-latih spaCy `en_core_web_sm` (arsitektur *Transition-based Neural Parser* dengan *Tok2Vec embeddings*). Model ini dipilih karena sangat ringan, inferensi cepat (<20 ms per artikel), deterministik, dan dapat dijalankan secara luring tanpa kebutuhan GPU khusus.
2. **Model Adapted (Terkalibrasi Domain)**: Mengintegrasikan komponen `EntityRuler` spaCy sebelum (*before="ner"*) komponen saraf statistik. Pendekatan hibrida ini memiliki keunggulan fundamental:
   - Menjamin akurasi 100% pada nama lembaga resmi, akronim perbankan regional Malaysia (BNM, CIMB, RHB, MDV), dan mata uang lokal (RM).
   - Mencegah fenomena pemisahan salah (_span mis-segmentation_), misalnya `"UOB Malaysia"` yang sering dipisah salah oleh model pra-latih umum menjadi `"UOB"` [ORG] dan `"Malaysia"` [GPE].
   - Tetap mempertahankan kekuatan generalisasi neural model untuk mendeteksi nama orang atau entitas baru yang belum terdaftar dalam kamus.

---

## 4. Fitur Ekstensi Sistem (Extension Challenges)

Sistem memenuhi seluruh tantangan ekstensi (*Extension Challenges*) yang tercantum pada panduan tugas:

1. **Visual Entity Highlighter**: Menyorot rentang entitas secara visual langsung pada teks naskah lengkap dengan lencana kategori warna yang elegan dan tooltip interaktif (kategori, keyakinan, nama kanonikal, rentang indeks karakter).
2. **Normalisasi Entitas Setara (_Canonical Entity Normalization_)**: Memetakan akronim, variasi penamaan lokal, dan padanan aksara Mandarin Hanzi ke bentuk resmi korporat tanpa merusak teks asli artikel:
   - `"BNM"` $\rightarrow$ `"Bank Negara Malaysia"`
   - `"MDV"` $\rightarrow$ `"Malaysia Debt Ventures Berhad"`
   - `"Maybank"` $\rightarrow$ `"Malayan Banking Berhad"`
   - `"CIMB"` $\rightarrow$ `"CIMB Group Holdings Berhad"`
   - `"KL"` $\rightarrow$ `"Kuala Lumpur"`
   - `"马银行"` $\rightarrow$ `"Malayan Banking Berhad"` (Mandarin)
   - `"联昌国际"` $\rightarrow$ `"CIMB Group Holdings Berhad"` (Mandarin)
   - `"马来西亚曲棍球联合会"` $\rightarrow$ `"Malaysian Hockey Confederation"` (Mandarin)
   - `"屠妖节"` $\rightarrow$ `"Deepavali Day"` (Mandarin)
   - `"武吉加里尔"` $\rightarrow$ `"Bukit Jalil"` (Mandarin)
3. **Skor Keyakinan Transparan (_Confidence Scores_)**: Setiap entitas menyajikan nilai estimasi probabilitas keandalan (skala 0.0 s.d. 1.0) dengan kriteria terukur:
   - Skor $\ge 0.95$: Pencocokan deterministik berbasis aturan/kamus domain tepercaya.
   - Skor $0.85 - 0.94$: Prediksi model neural dengan vektor representasi kata dalam kosakata (_in-vocabulary_).
   - Skor $0.70 - 0.84$: Deteksi kontekstual berbasis fitur statistik di luar kosakata (_out-of-vocabulary_).
4. **Dukungan Multibahasa Lengkap (English, Melayu, & Chinese/Hanzi)**:
   - Mendeteksi bahasa masukan secara otomatis (`en`, `ms`, `zh`).
   - Teks beraksara non-Latin Mandarin diproses melalui segmenter berbasis C++ (`spacy-pkuseg`) dan model `zh_core_web_sm` yang dilengkapi aturan entitas finansial Malaysia versi Hanzi (misal: `"马来西亚曲棍球联合会"`, `"武吉加里尔"`, `"屠妖节"`).
5. **Endpoint Pemrosesan Batch (`POST /extract-entities/batch`)**: Memproses hingga 50 artikel sekaligus dalam satu panggilan API jaringan yang efisien.
6. **Perbandingan Dua Model Evaluasi**: Menyediakan perbandingan performa *head-to-head* antara model `Baseline` dan `Adapted` pada dataset uji yang sama.

---

## 5. Petunjuk Instalasi dan Menjalankan Aplikasi

Aplikasi dijalankan secara lokal di dalam direktori `project/` dengan langkah-langkah standar berikut:

### 1. Masuk ke Direktori Proyek
```bash
cd project
```

### 2. Buat dan Aktifkan Virtual Environment
- **Di Windows (PowerShell / Command Prompt)**:
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate
  ```
- **Di Linux / macOS**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Pasang Dependensi & Model Bahasa spaCy
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

### 4. Jalankan Server Layanan Web (FastAPI)
```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```
Setelah server aktif, akses layanan melalui peramban:
- **Antarmuka Web Panel Pengguna**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Dokumentasi Interaktif API (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 5. Menjalankan Pengujian Otomatis (Pytest)
Untuk mengeksekusi 14 skenario pengujian fungsional dan validasi:
```bash
pytest -v
```

### 6. Menjalankan Evaluasi Kuantitatif Model
Untuk mereproduksi metrik evaluasi model baseline vs adapted pada dataset acuan:
```bash
python src/evaluate.py
```

### 7. Menjalankan Pemrofilan Dataset Sampel CSV
Untuk mereproduksi hasil audit dan pemrofilan dataset kliping berita:
```bash
python src/data_profiler.py
```

---

## 6. Dokumentasi API & Spesifikasi Kontrak

### 1. Endpoint Tunggal: `POST /extract-entities`

Mengekstrak entitas dari satu artikel berita.

#### Request Body
_Header: `Content-Type: application/json`_

```json
{
  "title": "BNM and Maybank execute landmark agreement in KL",
  "body": "Bank Negara Malaysia (BNM) and Malayan Banking Berhad confirmed new financing terms on 15 September 2026.",
  "model": "adapted"
}
```

#### Response Body (`HTTP 200 OK`)

```json
{
  "entities": [
    {
      "text": "BNM",
      "label": "ORG",
      "start": 0,
      "end": 3,
      "confidence": 0.98,
      "normalized": "Bank Negara Malaysia"
    },
    {
      "text": "Maybank",
      "label": "ORG",
      "start": 8,
      "end": 15,
      "confidence": 0.98,
      "normalized": "Malayan Banking Berhad"
    },
    {
      "text": "KL",
      "label": "ORG",
      "start": 46,
      "end": 48,
      "confidence": 0.98,
      "normalized": "Kuala Lumpur"
    },
    {
      "text": "Bank Negara Malaysia",
      "label": "ORG",
      "start": 50,
      "end": 70,
      "confidence": 0.98,
      "normalized": "Bank Negara Malaysia"
    },
    {
      "text": "BNM",
      "label": "ORG",
      "start": 72,
      "end": 75,
      "confidence": 0.98,
      "normalized": "Bank Negara Malaysia"
    },
    {
      "text": "Malayan Banking Berhad",
      "label": "ORG",
      "start": 81,
      "end": 103,
      "confidence": 0.98,
      "normalized": "Malayan Banking Berhad"
    },
    {
      "text": "15 September 2026",
      "label": "DATE",
      "start": 137,
      "end": 154,
      "confidence": 0.92,
      "normalized": null
    }
  ],
  "meta": {
    "entity_count": 7,
    "model": "spacy-en_core_web_sm-adapted",
    "language": "en",
    "confidence_explanation": "Confidence score scale (0.0 to 1.0): >=0.95: deterministic domain/rule-based matches; 0.85-0.94: high-probability in-vocabulary neural predictions; 0.70-0.84: contextual out-of-vocabulary detections."
  },
  "cleaned_text": "BNM and Maybank execute landmark agreement in KL\n\nBank Negara Malaysia (BNM) and Malayan Banking Berhad confirmed new financing terms on 15 September 2026."
}
```

#### Contoh Ekstraksi Multibahasa Mandarin (`zh` / Aksara Non-Latin)

Endpoint yang sama otomatis mendeteksi karakter Hanzi, mengalihkan pemrosesan ke segmenter `spacy-pkuseg` dan model `zh_core_web_sm` yang dilengkapi 79 aturan entitas perbankan & regional versi Mandarin.

_Request Body:_
```json
{
  "title": "顺应公众反馈，MHC重新安排屠妖节比赛",
  "body": "马来西亚曲棍球联合会确认，在与社区领袖协商后，武吉加里尔的所有MHL比赛将移至明年11月进行。",
  "model": "adapted"
}
```

_Response Body (`HTTP 200 OK`):_
```json
{
  "entities": [
    {
      "text": "MHC",
      "label": "ORG",
      "start": 7,
      "end": 10,
      "confidence": 0.98,
      "normalized": "Malaysian Hockey Confederation"
    },
    {
      "text": "屠妖节",
      "label": "EVENT",
      "start": 14,
      "end": 17,
      "confidence": 0.98,
      "normalized": "Deepavali Day"
    },
    {
      "text": "马来西亚曲棍球联合会",
      "label": "ORG",
      "start": 21,
      "end": 31,
      "confidence": 0.98,
      "normalized": "Malaysian Hockey Confederation"
    },
    {
      "text": "武吉加里尔",
      "label": "LOC",
      "start": 44,
      "end": 49,
      "confidence": 0.98,
      "normalized": "Bukit Jalil"
    },
    {
      "text": "明年11月",
      "label": "DATE",
      "start": 60,
      "end": 65,
      "confidence": 0.89,
      "normalized": null
    }
  ],
  "meta": {
    "entity_count": 5,
    "model": "spacy-zh_core_web_sm-adapted",
    "language": "zh",
    "confidence_explanation": "Confidence score scale (0.0 to 1.0): >=0.95: deterministic domain/rule-based matches; 0.85-0.94: high-probability in-vocabulary neural predictions; 0.70-0.84: contextual out-of-vocabulary detections."
  },
  "cleaned_text": "顺应公众反馈，MHC重新安排屠妖节比赛\n\n马来西亚曲棍球联合会确认，在与社区领袖协商后，武吉加里尔的所有MHL比赛将移至明年11月进行。"
}
```

---

### 2. Endpoint Batch: `POST /extract-entities/batch`

Memproses beberapa artikel sekaligus dalam satu panggilan jaringan (maksimum 50 artikel per batch).

#### Request Body
```json
{
  "articles": [
    {
      "title": "CIMB expands regional network into Singapore",
      "body": "CIMB Group announced new offices in Singapore and Indonesia.",
      "model": "adapted"
    },
    {
      "title": "A simple guide to healthy habits",
      "body": "Drinking clean water and sleeping early helps improve focus and physical energy.",
      "model": "adapted"
    }
  ]
}
```

#### Response Body (`HTTP 200 OK`)
```json
{
  "total_articles": 2,
  "results": [
    {
      "entities": [
        {
          "text": "CIMB",
          "label": "ORG",
          "start": 0,
          "end": 4,
          "confidence": 0.98,
          "normalized": "CIMB Group Holdings Berhad"
        },
        {
          "text": "Singapore",
          "label": "GPE",
          "start": 35,
          "end": 44,
          "confidence": 0.98,
          "normalized": null
        },
        {
          "text": "CIMB Group",
          "label": "ORG",
          "start": 46,
          "end": 56,
          "confidence": 0.98,
          "normalized": "CIMB Group Holdings Berhad"
        },
        {
          "text": "Singapore",
          "label": "GPE",
          "start": 82,
          "end": 91,
          "confidence": 0.98,
          "normalized": null
        },
        {
          "text": "Indonesia",
          "label": "GPE",
          "start": 96,
          "end": 105,
          "confidence": 0.98,
          "normalized": null
        }
      ],
      "meta": {
        "entity_count": 5,
        "model": "spacy-en_core_web_sm-adapted",
        "language": "en",
        "confidence_explanation": "Confidence score scale (0.0 to 1.0): >=0.95: deterministic domain/rule-based matches; 0.85-0.94: high-probability in-vocabulary neural predictions; 0.70-0.84: contextual out-of-vocabulary detections."
      },
      "cleaned_text": "CIMB expands regional network into Singapore\n\nCIMB Group announced new offices in Singapore and Indonesia."
    },
    {
      "entities": [],
      "meta": {
        "entity_count": 0,
        "model": "spacy-en_core_web_sm-adapted",
        "language": "en",
        "confidence_explanation": "Confidence score scale (0.0 to 1.0): >=0.95: deterministic domain/rule-based matches; 0.85-0.94: high-probability in-vocabulary neural predictions; 0.70-0.84: contextual out-of-vocabulary detections."
      },
      "cleaned_text": "A simple guide to healthy habits\n\nDrinking clean water and sleeping early helps improve focus and physical energy."
    }
  ]
}
```

#### Respons Validasi Gagal (`HTTP 400 Bad Request`)

Jika input tidak berupa string, berisi spasi kosong, atau melebihi batas panjang (**Judul > 1.000 karakter, Isi > 50.000 karakter**):

```json
{
  "error": "HTTP Error",
  "detail": "Field 'title' cannot be empty or contain only whitespace or empty HTML tags."
}
```

Jika struktur JSON rusak atau field wajib tidak dikirim:

```json
{
  "error": "Bad Request",
  "detail": "Invalid request schema or malformed JSON data.",
  "errors": ["Field required"]
}
```

---

## 7. Hasil Evaluasi Kuantitatif & Pembahasan Model

### Kriteria Pencocokan (_Matching Criteria_)
Evaluasi diuji secara ketat menggunakan pencocokan rentang persis (**Exact-Span & Label Matching**). Prediksi entitas dinyatakan benar (*True Positive*) jika dan hanya jika:
1. `start` karakter prediksi sama persis dengan `start` karakter ground truth.
2. `end` karakter prediksi sama persis dengan `end` karakter ground truth.
3. `label` kategori prediksi identik dengan `label` ground truth.

Jika terjadi pergeseran batas karakter (misal akibat pelekatan kata sandang *"the"*), maka prediksi tersebut dihitung sebagai **1 False Positive** dan **1 False Negative**.

### Hasil Perbandingan Metrik Evaluasi

Dievaluasi pada dataset acuan `data/evaluation_examples.json` (12 dokumen acuan dengan 61 entitas ground truth):

| Model Pipeline | Precision | Recall | F1-Score | True Positives (TP) | False Positives (FP) | False Negatives (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (spaCy Murni)** | 0.6102 | 0.5902 | 0.6000 | 36 | 23 | 25 |
| **Adapted (Terkalibrasi Domain)** | **0.7612** | **0.8361** | **0.7969** | **51** | **16** | **10** |
| **Peningkatan Performa** | **+15,10%** | **+24,59%** | **+19,69%** | **+15** | **-7** | **-15** |

### Performa Antar-Bahasa (Multilingual Breakdown)

| Bahasa Dokumen | Model Baseline (P / R / F1) | Model Adapted (P / R / F1) | Peningkatan F1 | Jumlah Dokumen Uji |
| :--- | :---: | :---: | :---: | :---: |
| **Bahasa Inggris (`en`)** | 0.65 / 0.61 / 0.63 | **0.80 / 0.84 / 0.82** | **+19 poin** | 11 dokumen |
| **Bahasa Melayu / Regional (`ms`)** | 0.29 / 0.40 / 0.33 | **0.50 / 0.80 / 0.62** | **+29 poin** | 1 dokumen |

> **Catatan Terkait Evaluasi Teks Mandarin (`zh`)**:  
> Dari audit dataset mentah CSV pada Bab 2, ditemukan 38 artikel beraksara Mandarin Hanzi (2,62%). Sistem mengimplementasikan pipeline terpisah berbasis `zh_core_web_sm` + segmenter `spacy-pkuseg` dengan 79 aturan leksikal finansial Hanzi (`CHINESE_NEWS_PATTERNS`). Karena bahasa Mandarin memiliki sistem penulisan non-spasi dan menggunakan tokenizer C++ terpisah, verifikasi kualitasnya dieksekusi secara otomatis melalui unit test `test_chinese_language_extraction` pada `tests/test_api.py`. Korpus acuan `data/evaluation_examples.json` difokuskan untuk menguji komparasi head-to-head model Latin (`en_core_web_sm` baseline vs adapted).

### Performa per Kategori Label (Model Adapted)

| Label Entitas | Precision | Recall | F1-Score | Dukungan Data (_Support_) | Karakteristik Tantangan |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ORG** (Lembaga, Bank, Korporasi) | 0.83 | 0.83 | 0.83 | 29 entitas | Kinerja tinggi dan seimbang berkat aturan domain perbankan. |
| **GPE** (Negara, Kota, Wilayah) | 0.93 | 0.93 | 0.93 | 14 entitas | Sukses mengenali kota dan negara bagian regional. |
| **DATE** (Waktu dan Tanggal) | 0.56 | 0.71 | 0.62 | 7 entitas | Ambiguitas span kata relatif seperti *"next November"*. |
| **PERSON** (Nama Tokoh / Individu) | 0.60 | 0.75 | 0.67 | 8 entitas | Tantangan pelekatan gelar kehormatan (*Tan Sri*) atau tanda posesif. |
| **LOC** (Lokasi Fasilitas Fisik) | 1.00 | 1.00 | 1.00 | 2 entitas | Sempurna mengenali entitas tempat non-geopolitik. |
| **EVENT** (Hari Raya / Peristiwa) | 1.00 | 1.00 | 1.00 | 1 entitas | Sukses dipisahkan dari kategori waktu biasa. |

---

## 8. Analisis Kesalahan (Error Analysis)

Berdasarkan log galat evaluasi kuantitatif dari `src/evaluate.py`, ditemukan empat pola kesalahan utama:

1. **Pelekatan Kata Sandang Penentu (_Article Attachment / Determiner Inclusion_)**:
   - *Teks Masukan*: `...adviser to the Democratic Action Party...`
   - *Prediksi Model*: `"the Democratic Action Party"` [ORG] (indeks 115–142)
   - *Ground Truth*: `"Democratic Action Party"` [ORG] (indeks 119–142)
   - *Akar Masalah*: Model saraf statistik memasukkan token penentu determinatif (*"the"*) ke dalam span entitas organisasi karena distribusi probabilitas transisi sintaksis dalam korpus pelatihan umum.
2. **Perluasan Batas Waktu Relatif (_Temporal Modifier Expansion_)**:
   - *Teks Masukan*: `...will be moved to next November.`
   - *Prediksi Model*: `"next November"` [DATE] (indeks 159–172)
   - *Ground Truth*: `"November"` [DATE] (indeks 164–172)
   - *Akar Masalah*: Penambahan kata keterangan relatif (*"next"*) dianggap oleh modul temporal sebagai bagian utuh ekspresi waktu, menimbulkan diskrepansi batas karakter (*offset mismatch*).
3. **Pelekatan Tanda Baca Posesif (_Possessive Clitic Inclusion_)**:
   - *Teks Masukan*: `...Tun Dr. Mahathir Mohamad's leadership...`
   - *Prediksi Model*: `"Mahathir Mohamad's"` [PERSON] (indeks 107–125)
   - *Ground Truth*: `"Mahathir Mohamad"` [PERSON] (indeks 107–123)
   - *Akar Masalah*: Tanda petik tunggal posesif (`'s`) ikut terserap ke dalam rentang karakter entitas nama persona.
4. **Fragmentasi Entitas Majemuk pada Baseline**:
   - *Teks Masukan*: `...PETALING JAYA: UOB Malaysia, RHB Banking and Maybank...`
   - *Prediksi Baseline*: `"UOB Malaysia"` gagal terdeteksi sebagai satu nama bank utuh dan hanya mengenali `"Malaysia"` sebagai [GPE].
   - *Solusi Model Adapted*: Pola kamus leksikal `EntityRuler` berhasil mengunci entitas organisasi perbankan regional secara utuh.

---

## 9. Jawaban Pertanyaan Studi Kasus (Questions to Answer)

Berikut adalah jawaban komprehensif atas 8 pertanyaan yang tercantum pada panduan tugas (*Problem Set Page 5*):

### 1. Apa yang mendefinisikan sebuah rentang teks sebagai entitas bernama (*named entity*) dalam masalah ini?
Dalam konteks layanan pemantauan berita daring perbankan ini, sebuah rentang teks (*text span*) didefinisikan sebagai entitas bernama jika teks tersebut merujuk secara kaku (*rigid designator*) pada suatu entitas spesifik di dunia nyata yang memiliki eksistensi formal, bukan kata benda umum (*common noun*). Kategori entitas yang didukung meliputi:
- **Organisasi (`ORG`)**: Bank, regulator keuangan, institusi pemerintah, partai politik (misal: *Maybank*, *Bank Negara Malaysia*, *DAP*).
- **Entitas Geopolitik (`GPE`)**: Negara, kota berdaulat, dan negara bagian (misal: *Malaysia*, *Kuala Lumpur*, *Sabah*).
- **Tokoh / Individu (`PERSON`)**: Nama manusia spesifik (misal: *Anwar Ibrahim*, *Lim Guan Eng*).
- **Lokasi Fisik (`LOC`)**: Fasilitas geografis atau stadion non-pemerintahan (misal: *Bukit Jalil*).
- **Waktu (`DATE`)**: Penanggalan kalender spesifik atau referensi temporal absolut (misal: *15 September 2026*).
- **Peristiwa (`EVENT`)**: Hari raya resmi atau perhelatan terjadwal (misal: *Deepavali Day*).

Sebaliknya, konsep finansial umum seperti *"emergency fund"*, *"interest account"*, atau *"inflation"* bukan merupakan entitas bernama.

### 2. Mengapa judul (*title*) dan isi berita (*body*) harus digabungkan sebelum dianalisis? Informasi apa yang hilang jika diproses secara terpisah?
Penggabungan judul dan isi berita (`title + "\n\n" + body`) sangat esensial karena:
1. **Mempertahankan Rantai Koreferensi & Anafora**: Judul berita umumnya sangat ringkas dan sering kali hanya menyebutkan nama keluarga atau akronim (misal: *"Anwar Leads Summit"*). Rincian lengkap jabatan (*"Prime Minister Anwar Ibrahim"*) berada di tubuh berita. Menggabungkan teks memungkinkan model mempertahankan konteks representasi wacana secara utuh.
2. **Definisi Akronim Pertama Kali**: Sering kali judul menggunakan singkatan (misal: *"MDV Funds Tech"*), sementara kalimat pertama tubuh berita menjabarkan kepanjangannya (*"Malaysia Debt Ventures (MDV)..."*). Penggabungan naskah memberikan konteks bagi model untuk mengaitkan akronim dengan nama panjangnya.
3. **Ruang Koordinat Offset Karakter Tunggal**: Jika diproses terpisah, sistem akan menghasilkan dua set offset karakter yang berbeda (offset judul vs offset tubuh), menyulitkan antarmuka visual dalam merender penyorotan teks terpadu dan meningkatkan beban panggilan API jaringan dua kali lipat.

### 3. Apa saja label yang didukung model yang dipilih, dan bagaimana pemetaannya ke kategori yang ditampilkan kepada pengguna?
Model spaCy pra-latih dilatih menggunakan skema tag OntoNotes 5.0 yang memiliki 18 kategori terperinci. Sistem mengimplementasikan kamus pemetaan `LABEL_MAPPING` di [src/ner_model.py](src/ner_model.py) untuk menyederhanakan dan menstandarkan kategori yang ditampilkan kepada pengguna akhir:
- `ORG` dan `ORGANIZATION` $\rightarrow$ dipetakan ke **`ORG`** (Lembaga/Perusahaan).
- `GPE` $\rightarrow$ dipetakan ke **`GPE`** (Negara/Kota/Wilayah).
- `PERSON` $\rightarrow$ dipetakan ke **`PERSON`** (Nama Tokoh).
- `LOC` dan `LOCATION` $\rightarrow$ dipetakan ke **`LOC`** (Lokasi Fisik/Fasilitas).
- `DATE` dan `TIME` $\rightarrow$ dipetakan ke **`DATE`** (Tanggal & Waktu).
- `EVENT` $\rightarrow$ dipetakan ke **`EVENT`** (Peristiwa/Hari Raya).
- Label non-krusial atau berisiko ambiguitas (seperti `MONEY`, `PERCENT`, `CARDINAL`) disaring atau dialihkan agar fokus pengguna tetap pada entitas subjek pemantauan berita utama.

### 4. Bagaimana tokenisasi, tanda baca, kapitalisasi, atau bahasa mempengaruhi hasil ekstraksi?
- **Tokenisasi**: Tanda hubung, garis miring, atau titik pada singkatan (misal: *U.O.B.* atau *CIMB-Niaga*) dapat memecah token secara keliru jika tokenizer bahasa Inggris standar tidak dikonfigurasi dengan aturan leksikal khusus.
- **Tanda Baca**: Simbol ampersand (`&` seperti pada *BNM, MOF & KAF Investment Bank*) atau tanda petik posesif (`'s` seperti pada *Dr. Mahathir Mohamad's team*) sering kali menarik kata berikutnya atau memutus batas entitas.
- **Kapitalisasi**: Model statistik spaCy sangat bergantung pada fitur kapitalisasi (*casing features*). Judul berita yang ditulis dalam huruf besar semua (*ALL-CAPS*) sering mengalami penurunan drastis pada recall karena menghilangkan sinyal kapital kata benda khusus.
- **Bahasa**: Bahasa non-spasi seperti Mandarin (`zh`) tidak dapat diproses oleh tokenizer berbasis spasi Latin. Sistem mengatasi hal ini dengan menambahkan segmenter C++ `spacy-pkuseg` dan model `zh_core_web_sm` yang mampu memecah rantai karakter Hanzi menjadi unit kata bermakna.

### 5. Apa perbedaan antara teks entitas (*text*), label (*label*), dan offset karakter (*start* & *end*)?
- **`text`**: Substring harfiah yang muncul di dalam naskah artikel asli (misal: `"Maybank"`).
- **`label`**: Kelas semantik atau kategori entitas di dunia nyata (misal: `"ORG"`).
- **`start`**: Indeks posisi karakter pertama dari teks entitas dalam naskah (indeks berbasis nol / *zero-based index*, inklusif).
- **`end`**: Indeks posisi tepat setelah karakter terakhir dari teks entitas dalam naskah (*exclusive character offset*).

Kombinasi `start` dan `end` menjamin bahwa potongan string `combined_text[start:end]` akan menghasilkan teks yang identik dengan atribut `text`, memungkinkan penyorotan visual tanpa ambiguitas jika ada kata yang sama berulang di tempat lain.

### 6. Apa yang terjadi jika entitas yang sama muncul berulang kali dengan variasi ejaan atau kapitalisasi?
Dalam teks berita sering ditemui entitas yang sama disebut dengan berbagai bentuk (misal: *"BNM"*, *"Bank Negara"*, dan *"Bank Negara Malaysia"*, atau *"cimb"* dan *"CIMB Group"*).
- Model kami mengekstrak setiap kemunculan secara independen dengan batas offset aslinya (`start` dan `end` berbeda).
- Melalui modul **Canonical Entity Normalization** (`src/ner_model.py`), setiap kemunculan diperkaya dengan atribut `normalized` (misal: `"BNM"` $\rightarrow$ `"Bank Negara Malaysia"`).
- Hal ini memberikan nilai tambah krusial bagi sistem pemantauan media: teks asli artikel tetap terlindungi tanpa distorsi, sementara basis data analitik dapat melakukan agregasi metrik penyebutan brand (*share of voice*) di bawah satu entitas kanonikal terpadu.

### 7. Kesalahan mana yang paling krusial bagi tim pemantau berita (*news-monitoring team*), dan bagaimana sistem dapat ditingkatkan?
Bagi tim pemantau berita media (*media intelligence / PR monitoring*), **False Negative (FN) pada entitas merek klien/kompetitor adalah kesalahan paling fatal**. Jika sistem gagal mendeteksi penyebutan *"Maybank"* atau *"CIMB"* dalam sebuah artikel krisis perbankan, tim humas tidak akan menerima notifikasi peringatan dini.
- Sebaliknya, **False Positive (FP)** seperti salah mendeteksi nama hari sebagai entitas event memiliki dampak bisnis yang jauh lebih kecil.
- **Peningkatan Masa Depan**:
  1. *Gazetteer Expansion*: Menghubungkan sistem ke basis data korporat Wikidata / Wikipedia API untuk cakupan entitas dinamis.
  2. *Transformer-based Contextual Embeddings*: Mengganti pipeline CNN/Tok2Vec dengan model Transformer multibahasa ringan (seperti *mDeBERTa-v3* atau *IndoBERT*) untuk menangkap konteks semantik rumit dan membedakan konteks metaforis.
  3. *Post-Processing Heuristics*: Menambahkan pembersih kata sandang otomatis (*determiner stripping*) untuk menghapus kata *"the"* atau *"a"* dari batas entitas `ORG` dan `PERSON`.

### 8. Isu privasi, hak cipta (*copyright*), bias, atau etika kecerdasan buatan apa yang relevan saat memproses teks artikel berita?
1. **Privasi & PII (Personal Identifiable Information)**: Artikel berita dapat memuat nama individu, pejabat publik, maupun warga sipil yang terlibat dalam kasus hukum. Sistem harus mematuhi regulasi perlindungan data pribadi (seperti UU PDP di Indonesia atau PDPA di Malaysia) dengan menerapkan kebijakan retensi data yang ketat dan tidak mengekspos PII warga sipil tanpa hak.
2. **Hak Cipta & Doktrin *Fair Use***: Teks kliping media merupakan kekayaan intelektual milik penerbit berita (*media publishers*). Sistem hanya menyimpan ringkasan ekstraksi entitas dan metadata artikel, serta **tidak mendistribusikan ulang atau menyimpan salinan naskah lengkap** di basis data publik (*transient processing*).
3. **Bias Representasi Bahasa & Geografis**: Model NLP pra-latih global dilatih mayoritas pada teks berbahasa Inggris negara Barat. Hal ini memicu bias inheren di mana nama tokoh Melayu, gelar kehormatan regional (*Tan Sri*, *Datuk Seri*), atau nama tempat lokal memiliki akurasi deteksi lebih rendah dibandingkan nama Barat. Adaptasi aturan domain lokal (*domain adaptation*) pada proyek ini merupakan langkah etis konkret untuk memitigasi bias representasi tersebut.

---

## 10. Keterbatasan Sistem dan Rekomendasi Masa Depan

1. **Skala Kamus Normalisasi Statis**: Kamus `ENTITY_NORMALIZATIONS` saat ini memuat entitas utama perbankan Malaysia dan kawasan ASEAN. Untuk skala produksi berskala global, kamus perlu dihubungkan secara dinamis ke repositori pengetahuan ontologi terbuka seperti Wikidata / DBpedia.
2. **Ketergantungan terhadap Aturan Leksikal Deterministik**: Komponen `EntityRuler` memprioritaskan kata kunci perbankan tertentu; pada kasus langka di mana nama lembaga digunakan dalam konteks perumpamaan atau metafora, aturan deterministik tetap akan menandainya sebagai `ORG`.
3. **Keterbatasan Segmentasi Morfologi Dialek Melayu Baku**: Pemroses teks bahasa Melayu masih mengandalkan representasi tokenisasi serumpun. Pengembangan berikutnya dapat mengadopsi model bahasa terlatih khusus bahasa Melayu seperti `malay-bert` untuk menangkap morfologi imbuhan kata (*affixes*) lokal secara lebih presisi.
