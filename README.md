# Prakiraan harga saham bank besar BEI (h+1 sampai h+5)

Aplikasi Streamlit untuk memprakirakan harga penutupan **BBCA, BBRI, dan BMRI** (BBNI opsional)
1 sampai 5 hari bursa ke depan. Data diambil langsung dari Yahoo Finance lewat `yfinance`, harga
live diperbarui otomatis, dan setiap model diuji dengan backtest rolling-origin terhadap tolok ukur
random walk sebelum direkomendasikan.

> Bukan nasihat investasi. Harga saham harian sangat dekat dengan random walk; tujuan aplikasi ini
> adalah prakiraan yang jujur beserta ketidakpastiannya, bukan sinyal trading.

## Isi aplikasi

| Bagian | Isi |
|---|---|
| Harga live | Kartu harga per saham, perubahan vs close sebelumnya, sparkline intraday, status sesi BEI. Diperbarui tiap 30 s/d 300 detik. Data Yahoo untuk BEI tertunda sekitar 10 menit. |
| Ringkasan | Prakiraan h+1..h+5 dari model terbaik tiap saham beserta skill-nya terhadap random walk. |
| Detail | Grafik kipas (harga aktual + median + interval), tabel prakiraan dengan batas bawah/atas, pilihan model, pembanding median model lain. |
| Evaluasi | MAE relatif per horizon, uji Diebold-Mariano, akurasi arah, cakupan interval out-of-sample. |
| Sentimen (opsional) | Tone berita GDELT per hari bursa, judul berita terbaru, dan ablation dengan vs tanpa sentimen. |
| Reproduksibilitas | Versi paket, seluruh setelan, unduhan CSV prakiraan dan metrik, metadata run (JSON). |

## Model

| Kunci | Model | Keterangan |
|---|---|---|
| `naive` | Random walk | Prakiraan = harga origin. Selalu ikut sebagai tolok ukur. |
| `gbm` | Gradient boosting | `HistGradientBoostingRegressor` (keluarga LightGBM), strategi direct: satu model per horizon untuk return log kumulatif, loss absolut (median bersyarat), refit tiap 20 origin. |
| `chronos2` | Chronos-2 (Amazon) | Foundation model zero-shot, mendukung past covariates. |
| `chronos_bolt` | Chronos-Bolt small | Varian cepat dari paket yang sama. |
| `timestral` | Timestral-209M (Mistral) | Bobot berlisensi Modified MIT (ada klausul pendapatan untuk perusahaan besar); mendukung past covariates. |
| `timesfm` | TimesFM 2.5 (Google) | Bobot 2.5 berlisensi Apache-2.0. Bobot TimesFM 3.0 berlisensi non-komersial, jadi tidak dipakai. |
| `ensemble` | Ensembel | Median prakiraan model non-sentimen (otomatis bila ada minimal dua). |

Model foundation bersifat opsional. Model yang paketnya belum terpasang ditandai di sidebar beserta
perintah instalasinya. Model yang gagal dimuat (misalnya unduhan terputus) ditampilkan sebagai error
dan dikeluarkan dari perbandingan, tanpa diganti diam-diam.

## Instalasi dan menjalankan

Butuh Python 3.11 atau lebih baru.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt       # lengkap: torch CPU, model foundation, GDELT
# atau, tanpa torch (random walk + gradient boosting saja):
pip install -r requirements-lite.txt
streamlit run streamlit_app.py
```

- Pemakaian pertama tiap model foundation mengunduh bobotnya dari Hugging Face (ratusan MB per model).
- GPU NVIDIA: hapus baris `--extra-index-url` di `requirements.txt`, pasang torch sesuai versi CUDA,
  lalu ubah `device = "cuda"` di `settings.toml`.
- Setelah instalasi berhasil, kunci versi untuk reproduksibilitas: `pip freeze > requirements.lock.txt`.
- Untuk menjalankan tes: `pip install -r requirements-dev.txt`, lalu `pytest`.

## Pengaturan

Semua parameter yang memengaruhi hasil ada di `settings.toml`: daftar saham, periode data, jam sesi
BEI, jumlah origin backtest, tingkat interval, hiperparameter gradient boosting, panjang konteks model
foundation, dan setelan sentimen. Tidak ada nilai default tersembunyi di kode; kunci yang hilang
membuat aplikasi berhenti saat start. Tema tampilan ada di `.streamlit/config.toml`.

Tanda air "Muhammad Ahsan · ITS" tampil samar di pojok kanan bawah halaman dan di dalam setiap
grafik, sehingga tangkapan layar grafik tetap membawanya. Teksnya diatur lewat `watermark` di
bagian `[ui]`; isi `""` untuk mematikannya.

Di sidebar:

- **Data dan model** (diterapkan dengan tombol *Terapkan*): saham, riwayat harga, model, jumlah
  origin backtest, sentimen berita.
- **Tampilan**: tingkat interval (hanya tingkat yang didukung jumlah origin), mode harga live sebagai
  close sementara, interval pembaruan harga live.

### Dua mode origin

- **Mati (default):** prakiraan dihitung dari close final terakhir. Saat bursa buka, h+1 adalah close
  hari ini, dan aplikasi menunjukkan apakah harga live sedang berada di dalam interval h+1.
- **Hidup:** harga live diperlakukan sebagai close sementara hari ini; h+1 menjadi hari bursa
  berikutnya dan prakiraan dihitung ulang setiap harga live berubah. Interval tetap berasal dari
  residual close final.

Bar harian hari ini baru dianggap final setelah 16:30 WIB (pra-penutupan selesai 16:00, ditambah
delay Yahoo dan margin). Begitu close final muncul, aplikasi menghitung ulang prakiraan secara otomatis.

## Metodologi

**Target.** Harga close yang disesuaikan dividen dan stock split (`auto_adjust=True`), sehingga return
log harian bebas lompatan akibat aksi korporasi dan close terakhir sama dengan harga pasar.

**Backtest rolling-origin.** Origin adalah N hari bursa terakhir yang kelima horizonnya sudah
teramati, sehingga semua model dan semua horizon dinilai pada origin yang sama. Tidak ada model yang
melihat bar setelah origin-nya. Ini diuji: data setelah origin diacak dan prakiraan pada origin itu
harus identik. Untuk gradient boosting, baris latih hanya dipakai bila targetnya sudah teramati saat
refit. Pada Chronos-2, `cross_learning=False`, supaya jendela backtest tidak saling membocorkan harga.

**Pembanding dan uji.** MAE, RMSE, MAPE per horizon; MAE relatif terhadap random walk; uji
Diebold-Mariano dengan koreksi Harvey-Leybourne-Newbold pada galat absolut (autokovarians sampai lag
h-1, perilaku fallback mengikuti `forecast::dm.test` di R); koreksi Holm atas semua pasangan
model × horizon per saham; akurasi arah dengan interval Wilson. Hari dengan harga tidak berubah tidak
dihitung untuk akurasi arah.

**Model terbaik.** MAE rata-rata h+1..h+5 terendah. Pemilihan memakai backtest yang sama dengan
pelaporannya, jadi skill model terbaik sedikit optimistis.

**Interval.** Split-conformal dari residual log backtest per horizon (peringkat
ceil((n+1)(1-alpha/2)) di tiap sisi). Jaminan cakupannya mengasumsikan residual exchangeable, yang
pada saham hanya berlaku kira-kira karena volatilitas mengelompok. Karena itu cakupan out-of-sample
diukur dengan kalibrasi pada paruh awal origin dan evaluasi pada paruh akhir. Interval 95% butuh
minimal 39 origin, 90% minimal 19, dan 80% minimal 9.

## Sentimen berita sebagai input

Aktifkan dengan toggle **Sentimen berita sebagai input** (perlu `gdeltdoc`).

- **Sumber:** GDELT DOC 2.0 API (gratis, tanpa API key). Mode timeline-nya mencakup berita sejak
  1 Januari 2017 dari 65 bahasa yang diterjemahkan mesin, dan bisa dicari dengan kata kunci bahasa
  Inggris. Kata kunci per saham ada di `[sentiment.queries]`.
- **Fitur:** tone rata-rata tertimbang jumlah artikel (`sent_tone`), rata-rata eksponensialnya
  (`sent_tone_ewm`), dan log jumlah artikel (`sent_log_count`). Gradient boosting memakai ketiganya.
  Chronos-2 dan Timestral memakai tone dan jumlah artikel sebagai past covariates (nilai kosong
  diisi 0 = tone netral).
- **Aturan waktu (tanpa kebocoran):** bin harian GDELT berbasis UTC (07:00 WIB s/d 07:00 WIB), jadi
  bin hari t memuat berita setelah close hari t. Prakiraan dari close hari t hanya memakai bin sampai
  hari t-1, dan berita akhir pekan masuk ke hari Senin. Hal ini diuji otomatis.
- **Ablation:** tiap model dijalankan dengan dan tanpa sentimen pada origin yang sama, lalu
  dibandingkan dengan uji DM (koreksi Holm). Hasilnya ada di bagian *Ablation sentimen berita*.
- **Catatan:** tone GDELT berbasis leksikon atas teks terjemahan mesin, sehingga kasar untuk berita
  berbahasa Indonesia, dan cakupan media lokal tidak lengkap. Pengambilan pertama butuh beberapa menit
  karena permintaan diberi jeda 5 detik. Tahun yang sudah lewat lalu di-cache. Bukti empiris di
  Indonesia beragam, jadi manfaatnya perlu dibuktikan lewat ablation, bukan diasumsikan.

## Validasi

`pytest` menjalankan 55 tes pada data **sintetis** (bukan hasil), antara lain:

- **Kausalitas:** fitur, backtest gradient boosting, model foundation (dengan pipeline tiruan), dan
  sentimen tidak berubah ketika data setelah origin diacak. Fitur yang sengaja dibuat bocor terdeteksi.
- **Ukuran uji DM:** pada simulasi galat tumpang-tindih MA(4) (h=5, n=60), mengabaikan autokorelasi
  memberi tingkat penolakan 26,9% pada alpha nominal 5%, sedangkan implementasi ini memberi 8,5%.
  Artinya p-value di horizon panjang dengan origin sedikit perlu dibaca hati-hati. Untuk h=1
  tingkatnya 4,3%.
- **Cakupan conformal:** pada residual exchangeable, cakupan mendekati nilai teoretis 80,3% (n=60).
- **Sanity gradient boosting:** pada return AR(1) dengan phi=0,6, MAE relatif h+1 = 0,84 dan
  signifikan. Pada random walk murni modelnya lebih buruk daripada naive di h+4/h+5 (MAE relatif
  sekitar 1,15), contoh mengapa tolok ukur random walk wajib ada.

`tests/test_app_smoke.py` menjalankan seluruh skrip Streamlit secara headless (AppTest) dengan data
tiruan. Tes ini otomatis dilewati bila Streamlit belum terpasang, jadi jalankan `pytest` setelah
instalasi untuk mengujinya di komputer Anda.

## Keterbatasan

- `yfinance` adalah klien tidak resmi; Yahoo bisa membatasi permintaan atau mengubah API. Error
  pengambilan data ditampilkan per saham.
- Model foundation mungkin pernah melihat sebagian riwayat saham ini saat pra-pelatihan, sehingga
  backtest pada periode lama bisa optimistis.
- Tanggal h+k mengikuti kalender `exchange_calendars` (XIDX). Perbarui paketnya agar libur bursa
  terbaru ikut. Jam sesi BEI ada di `settings.toml`; ubah bila BEI mengubah jam perdagangan.
- Kegagalan mengalahkan random walk adalah hasil yang sah, bukan bug.

## Deploy ke Streamlit Community Cloud

1. Simpan proyek ini di repositori GitHub, misalnya `ahsan4th/idx-bank-forecast`.
2. Di [share.streamlit.io](https://share.streamlit.io), klik **Create app** → **Yup, I have an app**, lalu
   isi repositori, branch `main`, dan main file `streamlit_app.py`.
3. Sebelum **Deploy**, buka **Advanced settings** dan pilih **Python 3.12**. Default Community Cloud
   mengikuti versi Python terbaru, dan versi Python tidak bisa diganti setelah aplikasi dibuat (harus
   dihapus lalu di-deploy ulang). Semua paket model di sini membutuhkan Python ≥ 3.10.

Community Cloud memasang `requirements.txt` dengan uv dan beralih ke pip bila uv gagal. Instalasi
pertama (torch CPU, transformers) bisa memakan beberapa menit, dan bobot model diunduh dari Hugging
Face saat model pertama kali dipakai. RAM aplikasi dibatasi sekitar 2,7 GB; bila aplikasi berhenti
karena batas sumber daya, kurangi model foundation yang dipilih (Chronos-Bolt paling ringan) atau
ganti isi `requirements.txt` dengan `requirements-lite.txt`.

## Struktur

```
streamlit_app.py          UI Streamlit
settings.toml             semua parameter
forecaster/
  config.py               pemuat settings.toml (dataclass beku, tanpa default tersembunyi)
  data.py                 yfinance, laporan kualitas data, bar sementara, kutipan live
  market.py               fase sesi BEI, finalitas bar harian, tanggal hari bursa berikutnya
  features.py             fitur kausal + target return log untuk gradient boosting
  models.py               random walk, gradient boosting, pembungkus model foundation, ensembel
  evaluation.py           metrik, uji DM-HLN, Holm, Wilson, interval conformal, cakupan
  pipeline.py             backtest rolling-origin dan prakiraan akhir
  sentiment.py            GDELT, agregasi harian, penyelarasan ke hari bursa tanpa kebocoran
  charts.py               opsi Apache ECharts (st.echarts_chart)
tests/                    pytest (data sintetis)
```

## Referensi

- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business &
  Economic Statistics*, 13(3), 253–263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared
  errors. *International Journal of Forecasting*, 13(2), 281–291.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of
  Statistics*, 6(2), 65–70.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free
  predictive inference for regression. *JASA*, 113(523), 1094–1111.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA*,
  22(158), 209–212.
- Kode dan dokumentasi model: github.com/amazon-science/chronos-forecasting,
  github.com/google-research/timesfm, github.com/mistralai/timestral.
- Data: github.com/ranaroussi/yfinance, github.com/gerrymanoim/exchange_calendars,
  github.com/alex9smith/gdelt-doc-api, blog.gdeltproject.org (DOC 2.0 API).
- Bukti sentimen di Indonesia: "Evaluasi Ketahanan IndoBERT dalam Analisis Sentimen Berita Ekonomi dan
  Implikasinya terhadap Prediksi Harga Saham", IPB University, 2026
  (repository.ipb.ac.id/handle/123456789/180005).
