# Cara Kerja segtrack

Dokumen ini ditulis untuk dibaca dari atas ke bawah. Tujuannya supaya
kamu bisa membuka repo ini enam bulan dari sekarang dan langsung paham
apa yang dulu kamu tulis, tanpa harus membaca semua kode.

Semua angka di dokumen ini dihitung dari kode yang benar-benar ada di
repo, bukan contoh karangan. Kalau kamu ubah kodenya, angkanya ikut
berubah.

---

## 1. Aplikasi ini sebenarnya apa

Masukkan satu video. Keluarannya video yang sama, tapi setiap objek di
dalamnya:

- diwarnai sesuai bentuk aslinya, bukan cuma kotak,
- diberi nomor yang **tetap sama** selama objek itu masih terlihat,
- diberi ekor yang menunjukkan jalur yang sudah dilewatinya,
- dihitung kalau melewati garis yang kamu tentukan.

Contoh pemakaian nyata: menghitung berapa motor dan berapa mobil yang
lewat satu ruas jalan, atau menghitung berapa orang yang masuk dan keluar
pintu dalam satu jam.

Aplikasi ini **tidak melatih model**. Ia memakai model YOLO yang sudah
jadi. Kalau kamu mau model yang mengenali objek khusus (misalnya jenis
penyakit daun), kamu latih dulu lewat `ultralytics`, lalu arahkan
`--model` ke file `best.pt` hasil latihanmu.

## 2. Tiga istilah yang harus dibedakan dulu

Ini bagian yang paling sering tertukar, dan tanpa ini sisa dokumen tidak
akan masuk.

**Deteksi (detection)** menjawab: *ada objek apa, di kotak mana.*
Hasilnya kotak `(x1, y1, x2, y2)` + nama kelas + skor keyakinan.

**Segmentasi (segmentation)** menjawab: *piksel mana saja yang milik
objek itu.* Hasilnya mask, yaitu matriks sebesar frame yang isinya
true/false per piksel. Lebih mahal dihitung daripada deteksi, tapi
bentuknya mengikuti objek, bukan kotak.

**Tracking** menjawab: *objek di frame ke-2 ini, apakah objek yang sama
dengan yang di frame ke-1?* Ini pertanyaan yang sama sekali berbeda.
Deteksi bekerja per frame dan tidak punya ingatan. Tanpa tracking, satu
mobil yang lewat selama 100 frame adalah 100 deteksi terpisah, dan kamu
tidak bisa menghitungnya sebagai satu mobil.

Tracking memberi setiap objek sebuah **track id**, angka yang bertahan
antar frame. Semua hal menarik di aplikasi ini bergantung pada id itu:
ekor butuh id untuk tahu titik-titik mana yang satu jalur, counter butuh
id untuk tidak menghitung satu mobil dua kali.

Di repo ini tracking dikerjakan oleh **BoT-SORT** atau **ByteTrack**,
keduanya sudah ada di dalam paket `ultralytics`. Proyek aslinya memakai
**DeepSORT**, yang harus diunduh terpisah dari Google Drive.

## 3. Urutan membaca kode

Jangan mulai dari `tracker.py`. Mulai dari yang paling kecil:

| Urutan | File | Baris | Isinya |
| --- | --- | --- | --- |
| 1 | `points.py` | ~30 | kotak jadi satu titik |
| 2 | `trails.py` | ~75 | menyimpan riwayat titik per id |
| 3 | `colors.py` | ~100 | kelas jadi warna, dan warna teks label |
| 4 | `counting.py` | ~190 | geometri garis dan penghitungnya |
| 5 | `overlay.py` | ~95 | menempel mask ke frame |
| 6 | `device.py`, `classes.py` | ~95, ~75 | pilih GPU/CPU, filter kelas |
| 7 | `draw.py` | ~250 | semua gambar-gambar OpenCV |
| 8 | `tracker.py` | ~250 | yang menyatukan semuanya |
| 9 | `cli.py` | ~250 | penerjemah argumen terminal |

File 1-6 tidak mengimpor OpenCV dan tidak mengimpor `ultralytics`. Itu
sengaja: artinya logikanya bisa diuji dalam sepersekian detik tanpa GPU,
tanpa file model, tanpa video. Semua 110 test yang jalan tanpa OpenCV ada
di situ.

## 4. Perjalanan satu frame

Ini inti dokumen. Satu frame masuk, lewat sembilan langkah, keluar
sebagai gambar.

```
video.mp4
    |
    v
[1] ultralytics: baca frame, jalankan model YOLO-seg
    |     keluar: kotak + mask + kelas + skor
    v
[2] ultralytics: BoT-SORT cocokkan dengan frame sebelumnya
    |     keluar: tambahan satu kolom -> track id
    v
[3] tracker._tracks_from()  ->  list[Track]
    |     ambil id, kelas, kotak; hitung warna; hitung anchor
    v
[4] tracker._paint_masks()  ->  frame bermask
    |     ukuran mask disesuaikan, lalu overlay.blend_masks()
    v
[5] tracker._count()        ->  trail diperbarui, crossing dicatat
    |     trails.update() + counting.crossed()
    v
[6] draw.draw_trail()       ->  ekor digambar
    v
[7] draw.draw_box()         ->  kotak + label "7 car"
    v
[8] draw.draw_hud()         ->  panel fps / jumlah / hitungan
    v
[9] draw.draw_watermark()   ->  baris hak cipta di kanan bawah
    |
    v
FrameResult  ->  cli.py  ->  file video / jendela preview
```

Dua hal yang perlu diperhatikan soal urutannya:

**Mask digambar sebelum kotak.** Kalau dibalik, mask yang setengah
transparan akan menutupi garis kotak dan tulisan labelnya.

**Ekor digambar dalam loop sendiri, sebelum semua kotak.** Kalau ekor dan
kotak digambar dalam satu loop, ekor objek yang diproses belakangan akan
melintas di atas kotak objek yang diproses lebih dulu. Dua loop terpisah
membuat semua kotak selalu berada di lapisan paling atas.

## 5. Modul satu per satu

### `points.py` — kotak jadi satu titik

Ekor dan counter tidak bisa bekerja dengan kotak; keduanya butuh satu
titik saja per objek per frame.

```python
anchor_point((100, 200, 180, 320), "bottom")  # -> (140, 320)
anchor_point((100, 200, 180, 320), "center")  # -> (140, 260)
```

`bottom` adalah tengah sisi bawah kotak, yaitu kira-kira tempat objek
menyentuh tanah. Itu yang kamu mau untuk garis hitung di jalan: mobil
dianggap lewat saat rodanya lewat, bukan saat atapnya lewat.

`center` lebih stabil kalau objek sering tertutup sebagian, karena sisi
bawah kotak melompat begitu kaki objek keluar frame.

Di proyek aslinya baris ini ditulis `int((y2 + y2) / 2)`. Secara hitungan
itu sama dengan `y2`, jadi hasilnya benar, tapi bentuk tulisannya
terbaca seperti salah ketik dari `(y1 + y2) / 2`. Siapa pun yang membaca
kode itu harus berhenti dan memastikan dulu. Di sini pilihannya
dinamai, jadi tidak ada yang perlu ditebak.

### `trails.py` — riwayat posisi

Satu `deque` per track id, panjang dibatasi (`maxlen`, default 64).
Titik terbaru ditaruh di depan.

```python
t = TrailStore(maxlen=4)
for x in (10, 20, 30, 40, 50):
    t.update(7, (x, 100))
t.points(7)   # -> [(50,100), (40,100), (30,100), (20,100)]
```

Titik `(10,100)` sudah terbuang karena `maxlen=4`. `deque` dipakai
justru karena itu: membuang dari ujung belakang sambil menambah di depan
adalah operasi O(1), sedangkan `list` harus menggeser semua isinya.

Satu hal penting: `TrailStore` adalah **objek**, bukan variabel global.
Di proyek aslinya riwayat ini disimpan di `data_deque = {}` di level
modul. Akibatnya, kalau kamu memproses dua video dalam satu proses
Python, ekor dari video pertama masih menempel di video kedua. Dengan
objek, setiap pipeline punya miliknya sendiri, dan ada `reset()`.

`prune(active_ids)` membuang ekor objek yang sudah hilang dari frame.
Dua perbaikan sekaligus di sini: ia menerima iterable apa pun termasuk
yang kosong (versi lama meledak kalau diberi `None`, padahal `None`
adalah nilai default parameternya sendiri), dan ia mengubah daftar id
jadi `set` dulu, sehingga pencarian jadi O(1) per key bukan pemindaian
linier di array NumPy setiap frame.

### `colors.py` — kelas jadi warna

Ini bagian dengan cerita paling berguna untuk dipelajari, karena versi
pertamanya salah dan test-nya yang menangkap.

**Versi pertama:** hash nama kelas (SHA-256) jadi hue. Alasannya bagus —
warna jadi stabil per nama, tidak tergantung urutan kelas di model.

**Masalahnya:** hashing itu penempatan acak, dan penempatan acak
bertabrakan. Dari 30 nama kelas COCO, dua di antaranya mendarat di
warna 8-bit yang persis sama. Ini paradoks ulang tahun: dengan 30 item
dan ruang warna yang terbatas, tabrakan bukan kemungkinan kecil, tapi
hal yang wajar terjadi.

**Versi sekarang:** hue diambil dari **indeks** kelas lewat barisan
golden ratio.

```python
hue = (class_id * 0.618033988749895) % 1.0
```

`0.618...` adalah `1/phi`. Sifatnya: setiap kelipatan berikutnya selalu
jatuh di celah terbesar yang masih tersisa. Jadi berapa pun jumlah
kelasnya, hue-nya tersebar rata, bukan menumpuk. Istilah teknisnya
*low-discrepancy sequence*.

Saturasi dan value diputar di bawahnya, supaya kelas yang hue-nya
berdekatan tetap bisa dibedakan:

```python
saturation = (0.95, 0.70, 0.85)[class_id % 3]
value      = (0.95, 0.72)[(class_id // 3) % 2]
```

Hasil terukur untuk 80 kelas COCO: **nol duplikat**, dan jarak
tetangga terdekat **19.4** di ruang BGR. Untuk 10 kelas, jaraknya 40+.

Contoh nyata:

| id | kelas | BGR | warna teks label | kontras |
| --- | --- | --- | --- | --- |
| 0 | person | `(12, 12, 242)` | hitam | 4.83 |
| 2 | car | `(36, 242, 157)` | hitam | 15.16 |
| 5 | bus | `(28, 112, 184)` | hitam | 5.37 |
| 7 | truck | `(73, 242, 80)` | hitam | 14.14 |

Kolom terakhir itu `text_color_for()`. Proyek aslinya selalu menulis
label dengan warna putih, yang hilang di atas kotak berwarna terang. Di
sini fungsi itu memilih hitam atau putih, mana pun yang kontrasnya lebih
tinggi. Ada jaminan matematisnya: memilih yang terbaik dari hitam dan
putih tidak pernah bisa lebih buruk dari **4.58:1** untuk warna latar apa
pun, jadi semua label selalu lewat ambang WCAG 4.5:1 untuk teks normal.

Pelajaran yang mau saya simpan dari bagian ini: *"deterministik" dan
"bisa dibedakan" itu dua syarat berbeda, dan hashing hanya memenuhi yang
pertama.*

### `counting.py` — geometri garis

Ini fitur yang dijanjikan README proyek aslinya tapi tidak ada
kodenya di repo-nya (garis hitungnya berupa `cv2.line` yang
dikomentari). Jadi bagian ini ditulis dari nol, dan dapat test paling
banyak di repo ini.

**Menentukan sisi.** Untuk garis berarah dari `p1` ke `p2` dan titik
`q`, hitung cross product:

```
cross = (p2.x - p1.x) * (q.y - p1.y) - (p2.y - p1.y) * (q.x - p1.x)
```

Tandanya memberi tahu sisi mana. Untuk garis `(0,100) -> (200,100)`:

| titik | hitungan | sisi |
| --- | --- | --- |
| `(100, 50)` | `200*(50-100) - 0*(100-0) = -10000` | `-1` |
| `(100, 150)` | `200*(150-100) - 0 = +10000` | `+1` |
| `(100, 100)` | `200*0 - 0 = 0` | `0`, tepat di garis |

**Menghitung penyeberangan.** Sebuah track dihitung kalau sisinya
berubah dari `-1` ke `+1` (disebut `in`) atau sebaliknya (`out`).

```python
c = LineCounter(((0,100), (200,100)))
c.update(1, "car", (100, 40))    # -> None   (penampakan pertama)
c.update(1, "car", (100, 80))    # -> None   (masih sisi -1)
c.update(1, "car", (100, 130))   # -> "in"   (pindah ke sisi +1)
c.update(1, "car", (100, 160))   # -> None   (sudah di sisi +1)
c.totals()                       # -> {"in": 1, "out": 0}
```

Tiga keputusan desain yang penting di sini:

**Penampakan pertama tidak pernah dihitung.** Objek yang sudah berada di
seberang garis saat pertama terlihat tidak pernah *dilihat* menyeberang.
Menghitungnya berarti mengarang kejadian.

**Titik tepat di garis (`sisi 0`) tidak menghitung apa-apa.** Sisi
terakhir yang tersimpan tetap yang lama. Tanpa aturan ini, orang yang
berdiri di garis akan dihitung berkali-kali setiap frame.

**Mode ketat (default) memeriksa potongan garis yang berhingga.**
`update()` memperlakukan garis sebagai garis tak berhingga, jadi mobil
yang lewat jauh di luar ujung garis juga ikut terhitung. `crossed()`
menambah syarat: segmen gerak objek harus benar-benar memotong segmen
garisnya. Itu yang kamu mau kalau garisnya hanya menutupi satu lajur.
`--loose-line` mengembalikan perilaku tak berhingga.

Pemeriksaan potongan segmen (bukan "jarak ke garis < N piksel") juga
yang membuat mobil cepat tetap terhitung. Mobil yang bergerak lebih
cepat daripada frame rate tidak pernah mendarat *di* garis: di frame 10
dia 50 piksel di atas, di frame 11 sudah 300 piksel di bawah. Ambang
jarak akan melewatkannya. Uji potongan segmen menangkapnya.

### `overlay.py` — menempel mask

NumPy saja, tanpa OpenCV, supaya bisa diuji pada matriks 2x2 yang bisa
dihitung tangan.

```python
blend_masks(frame, masks, colors, alpha=0.45)
```

Rumusnya rata-rata berbobot per piksel yang termask:

```
hasil = frame * (1 - alpha) + warna * alpha
```

Jadi frame bernilai 100 dengan warna 200 dan `alpha=0.5` menghasilkan
150. Ada test khusus untuk itu.

Dua hal yang perlu diingat:

**Mask yang lebih dulu menang di area tumpang tindih.** `ultralytics`
mengembalikan deteksi terurut dari skor tertinggi, jadi instance yang
lebih pasti tetap terlihat, bukan tertimpa oleh yang kebetulan diproses
terakhir.

**Seluruh frame dicampur dalam satu operasi,** bukan satu `addWeighted`
per mask. Biayanya tidak bertambah seiring jumlah objek yang
bertumpukan. Caranya: kumpulkan dulu semua warna ke satu lapisan plus
satu mask boolean gabungan, baru campur sekali.

### `draw.py` — semua yang butuh OpenCV

Hal yang menarik di sini penempatan label. Proyek aslinya selalu
menggambar label **di atas** kotak. Objek yang menyentuh tepi atas frame
kehilangan labelnya, karena digambar di koordinat y negatif.

```python
top = y1 - plate_height
if top < 0:
    top = min(y1, frame.shape[0] - plate_height)   # pindah ke dalam kotak
```

Ukuran huruf juga diskalakan terhadap tinggi frame (`_scaled`), supaya
render 4K tidak dipenuhi tulisan 12 piksel.

Ketebalan ekor meruncing relatif terhadap panjang ekornya sendiri:

```python
weight = 1.0 - (index - 1) / span
```

Versi aslinya memakai `sqrt(64 / (i + i)) * 1.5` dengan 64 ditulis
langsung sebagai angka. Artinya kalau kamu ubah panjang ekor, bentuk
runcingnya tidak ikut berubah, dan segmen pertama digambar lebih tebal
daripada garis kotaknya sendiri.

`draw_watermark()` menulis baris hak cipta di kanan bawah: satu kali
gelap lalu satu kali terang digeser 1 piksel, supaya terbaca di latar
terang maupun gelap tanpa perlu kotak di belakangnya. Fontnya Hershey
milik OpenCV, yang **hanya punya glif ASCII** — itu sebabnya teksnya
`(c)` bukan `©`; karakter non-ASCII akan muncul sebagai kotak kosong.

### `device.py` — pilih GPU atau CPU

Di proyek aslinya tracker dibuat dengan `use_cuda=True` ditulis
langsung. Jadi di laptop tanpa GPU NVIDIA, program mati, padahal file
konfigurasinya sendiri menyediakan opsi `device`.

```python
resolve_device("auto")       # cuda -> mps -> cpu, mana yang ada
resolve_device("cuda")       # kalau tidak ada GPU: peringatan, lalu cpu
resolve_device("1")          # -> "cuda:1"
resolve_device("tpu")        # -> ValueError
```

Perhatikan: permintaan `cuda` di mesin tanpa CUDA **tidak** melempar
error, hanya memperingatkan lalu turun ke CPU. Render satu jam tidak
layak mati di frame nol hanya karena mesinnya berganti. Tapi nama device
yang tidak dikenal memang error, karena itu salah tulis dan diam-diam
memilih CPU hanya akan membingungkan.

Fungsi probe hardware-nya dipisah jadi kelas `DeviceProbe` supaya test
bisa memalsukannya. Itu sebabnya 15 test device bisa jalan tanpa torch
sama sekali.

### `classes.py` — filter kelas

Di proyek aslinya ini tertulis sebagai `# TODO: filter by classes`, dan
tidak ada kunci `classes` di konfigurasinya. Padahal ini knop paling
sering dibutuhkan: biasanya kamu hanya mau kendaraan, bukan 80 kelas
COCO sekaligus.

```python
parse_classes("0,2,3", names)        # -> [0, 2, 3]
parse_classes("car,bus", names)      # -> [2, 5]
parse_classes("Car, BUS ", names)    # -> [2, 5]   (nama tidak case-sensitive)
parse_classes("cat", names)          # -> ValueError + daftar nama yang ada
parse_classes(None, names)           # -> None     (artinya semua kelas)
```

Nama yang salah **gagal di awal**, sebelum model jalan. Alternatifnya
adalah render selesai satu jam lalu hasilnya kosong tanpa alasan, dan itu
jauh lebih mahal daripada error di detik pertama.

### `tracker.py` — yang menyatukan

Inti fungsinya satu pemanggilan:

```python
self.model.track(
    source=..., stream=True, tracker="botsort.yaml",
    conf=..., iou=..., imgsz=..., device=..., classes=...,
    retina_masks=True, verbose=False,
)
```

`stream=True` membuatnya generator: frame diproses satu-satu, bukan
seluruh video dimuat ke memori dulu. Video satu jam tidak akan
menghabiskan RAM.

`retina_masks=True` meminta mask di resolusi asli frame, bukan di
resolusi inferensi. Jadi mask tidak perlu diperbesar dan tepinya tidak
kabur.

Bagian yang paling perlu kamu pahami kalau nanti mengubah file ini:

```python
if len(data) != len(tracks):
    # lewati overlay mask, catat peringatan
```

Seluruh dasar pewarnaan mask per track adalah asumsi bahwa mask ke-*i*
milik box ke-*i*. Kalau suatu versi `ultralytics` melanggar itu, lebih
baik tidak menggambar mask sama sekali daripada menempelkan mask satu
objek ke objek lain. Proyek aslinya punya bug persis di area ini:
list `masks`-nya hanya diisi untuk frame yang ada deteksinya, sehingga
indeksnya bisa bergeser dari indeks deteksi di mode multi-stream.

### `cli.py` — penerjemah argumen

`argparse` dari pustaka standar. Proyek aslinya memakai **hydra** dengan
`default.yaml` 130 baris yang mencampur setelan training, validasi, dan
prediksi, dan harus dijalankan dari dalam folder
`ultralytics/yolo/v8/segment` karena ada config yang dimuat lewat path
relatif.

hydra dibuang dari `ultralytics` beberapa minggu setelah snapshot itu
diambil. Itu sebagian besar alasan kenapa kode lama itu tidak bisa
dipindah ke versi sekarang tanpa ditulis ulang. `argparse` tidak akan
mengalami itu.

Satu detail kecil yang mudah terlewat: `--trail 0` mematikan
*penggambaran* ekor, tapi penyimpanannya tetap hidup di panjang minimum.
Sebabnya counter butuh posisi sebelumnya untuk tahu arah penyeberangan.

```python
trail=max(2, args.trail or 2),
draw_trails=bool(args.trail and args.trail >= 2),
```

## 6. Menjalankan dan bereksperimen

```bash
pip install -e ".[dev]"
```

Beberapa eksperimen yang paling cepat memberi pemahaman:

**Lihat bentuk default-nya.**
```bash
segtrack --source clip.mp4 --max-frames 30 --save probe.mp4
```
`--max-frames` membuat percobaan selesai dalam hitungan detik, bukan
menit. Pakai ini setiap kali kamu mencari posisi garis.

**Matikan satu lapisan sekaligus, lihat efeknya.**
```bash
segtrack --source clip.mp4 --no-masks --max-frames 30 --save probe.mp4
segtrack --source clip.mp4 --trail 0  --max-frames 30 --save probe.mp4
segtrack --source clip.mp4 --no-hud   --max-frames 30 --save probe.mp4
```

**Rasakan beda anchor.**
```bash
segtrack --source clip.mp4 --anchor center --max-frames 60 --save probe.mp4
```
Perhatikan ekornya naik ke tengah objek, dan hitungan garis berubah
waktunya.

**Rasakan beda tracker.**
```bash
segtrack --source clip.mp4 --tracker bytetrack.yaml --save bt.mp4
segtrack --source clip.mp4 --tracker botsort.yaml   --save bs.mp4
```
Perhatikan nomor id saat objek saling menutupi. BoT-SORT memakai
kemiripan penampilan, jadi lebih sering berhasil mempertahankan id
setelah objek tertutup lama. ByteTrack lebih ringan.

**Lihat test-nya jalan.**
```bash
pytest -q
pytest tests/test_counting.py -v
```
Baca nama-nama test di `test_counting.py`. Nama-namanya sengaja
berbentuk kalimat, jadi daftar itu sekaligus daftar aturan yang
berlaku di counter.

## 7. Kalau mau menambah fitur, taruh di mana

| Yang mau kamu tambah | File | Butuh OpenCV? |
| --- | --- | --- |
| Garis hitung kedua, atau area poligon | `counting.py` | tidak |
| Bentuk ekor lain (misalnya memudar) | `draw.py` | ya |
| Ekspor hasil ke CSV atau JSON | `cli.py` | tidak |
| Estimasi kecepatan dari jarak antar frame | `tracker.py` + modul baru | tidak |
| Palet warna dari file, bukan dari rumus | `colors.py` | tidak |
| Blur wajah atau plat nomor | `overlay.py` atau `draw.py` | ya |

Aturan yang saya pegang di repo ini: **kalau sebuah logika bisa ditulis
tanpa OpenCV dan tanpa ultralytics, tulis di modul sendiri yang bersih.**
Itu yang membuat 110 test bisa jalan dalam 0.3 detik tanpa GPU, dan itu
yang membuat bug warna di bagian 5 ketahuan sebelum sempat masuk ke
video.

## 8. Daftar istilah

| Istilah | Artinya di repo ini |
| --- | --- |
| frame | satu gambar dari video |
| bbox / `xyxy` | kotak objek, `(x1, y1, x2, y2)` piksel, asal di kiri atas |
| mask | matriks sebesar frame, true di piksel milik objek |
| track id | nomor objek yang bertahan antar frame |
| anchor | satu titik yang mewakili objek, lihat `points.py` |
| trail | ekor, riwayat anchor beberapa frame terakhir |
| crossing | kejadian satu track melewati garis hitung |
| `conf` | skor keyakinan minimum sebuah deteksi diterima |
| `iou` | ambang tumpang tindih untuk membuang deteksi ganda (NMS) |
| NMS | non-maximum suppression, pembuang kotak ganda untuk objek yang sama |
| BGR | urutan kanal warna milik OpenCV, kebalikan RGB |
| HUD | panel status di kiri atas frame |
| WCAG | standar aksesibilitas; dipakai di sini untuk ambang kontras teks |

## 9. Asal-usul

Desain aplikasi ini bukan milik saya. Idenya — jalankan segmentasi per
frame, serahkan kotaknya ke tracker, beri label id, gambar ekornya —
berasal dari
[Muhammad Moin Faisal](https://github.com/MuhammadMoinFaisal), lewat
proyek
[YOLOv8_Segmentation_DeepSORT_Object_Tracking](https://github.com/MuhammadMoinFaisal/YOLOv8_Segmentation_DeepSORT_Object_Tracking),
berlisensi GPL-3.0. YOLO dan paket `ultralytics` milik
[Ultralytics](https://github.com/ultralytics/ultralytics).

Yang saya kerjakan adalah menulis ulang implementasinya: membaca repo
itu, mendaftar cacatnya, lalu membangun kembali di atas API
`ultralytics` yang sekarang. Daftar lengkap apa yang berubah ada di
tabel di [README](../README.md), dan rantai lisensinya ada di
[NOTICE](../NOTICE).
