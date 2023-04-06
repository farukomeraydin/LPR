# AMAÇ

Baştan sona plaka tespiti yapan bir model geliştirmek. Bu modeli başlıca 2 bölüme ayırdık: nesne tespiti ve optik karakter okumak. 

# GENEL AKIŞ

1- Plaka tespiti

2- Karakter Tespiti

3- Karakter Tanıma

4- Optimizasyon


# DENENEN YÖNTEMLER

- Derin öğrenme kullanılarak tespit edilecek plakanın bounding box köşeleri tahmin edilmeye çalışıldı. Yani probleme bir regresyon problemi olarak yaklaşıldı.

- AutoML araçlarından autokeras kullanılarak en iyi model elde edilmeye çalışıldı.

- Görüntüler yamalara ayrıldı ve plaka içeren yamalar öğretildi. Yani sınıflandırma problemi olarak yaklaşıldı. Yamalara ayrılan görüntüler öncesinde hsv filtresinden
geçirildi.

- Template matching yöntemi kullanılarak plaka tespiti yapılmaya çalışıldı.

- OCR için mnist veri seti eğitildi ve kestirim yapıldı.

- OCR için CNN ağı kullanılarak kestirim yapıldı. Eğitmeden önce erosion, dilation gibi ön işlemlerden geçti. Eğitmeden önce plakadaki karakterleri kırpabilmek için en büyük dikdörtgeni tespit eden algoritma kullanıldı.

# PROBLEMLER

