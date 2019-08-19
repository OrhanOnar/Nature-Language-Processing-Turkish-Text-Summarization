# Türkçe Haber Metni Özetleme Sistemi

Haber metinleri ön işlemden geçirilmiş haldedir. Burada; etiketleme, öznitelik çıkarımı, sınıflandırmalar, referans yöntem ve bunların farklı parametrelerle deney ortamları geliştirilip çalıştırılarak  edinilen sonuçlar verilmiştir.Proje raporunu incelemeniz mutlaka tavsiye edilir.


## Gerekli Yüklemeler

Programın çalışması için gerekli paketler mevcut değilse 'pip' kullanılarak komut satırından yüklenebilir. 

```bash
pip install numpy
pip install sklearn
pip install rouge
pip install easy-rouge
```

## Kullanımı

Program Python 3 ile çalışır. Çalıştırmak için ilgili komutlar verilmiştir:

- Tüm deney sonuçlarının edinilmesi için:
```bash
python summarization.py
```
- Örnek özetleme sonucu görmek için (veri kümesinden rastgele bir haber metni seçilerek hem geliştirdiğimiz sistem hem de referans yöntem tarafından verilen):
```bash
python summarization.py ornek
```
