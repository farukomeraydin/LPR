# Genel Akış

Plaka tespiti opencv'nin haar kaskat yapısı kullanılarak gerçekleştirildi. Burada eğitilen model "cascade_1200pos_2600neg_15stage_72_24" dizinindedir. Eğitim için kullanılan veriler "" linkindedir. opencv kullanılarak yapılan bu eğitim "https://www.youtube.com/watch?v=XrCAvs9AePM&list=PL1m2M8LQlzfKtkKq2lK5xko4X-8EZzFPI&index=8" linkinden bakılarak yapılmıştır. Eğitim sırasında ROI'nin boyutu (24, 72) boyutunda belirlenmiştir. Plaka tespitinden sonra karakterlerin ayrıştırılması LPR sınıfındaki split metoduyla yapıldı. En son OCR aşaması konvolüsyonel derin ağ kullanılarak yapılmıştır. Plaka tespiti yaparken kameranın 105 dereceye ayarlandığı varsayılmıştır. Geriye kalan yerler kırpıldı. Baştan sona algoritmik olarak işlemler şöyledir: 

- Plaka tespiti yapılarak ROI'nin kırpılması
- Kırpılan alana adaptif eşikleme yapılması
- Eşikleme yapılan görüntüde arkaplanın olabildiğince siyaha boyanması
- Gerekliyse görüntünün yatay veya dikey kırpılması
- Plakadaki karakterlerin tek tek bölünerek kırpılması
- Bölünen karakterlerin OCR için eğitilen ağa sokularak predict yapılması

Bu aşamalarda test ortamına göre ayarlanacak parametreler aşağıdaki gibidir:

- Plaka tespiti için minNeighbors parametresi default 3'tür. Gereksiz tespitler yapılıyorsa bu arttırılmalıdır ve tek sayı olmalıdır. minSize ve maxSize parametresi ROI'nin boyutunun hangi aralıklarda olacağını belirtir ve video/görüntü boyutuna göre tekrardan ayarlanmalıdır. Düzgün ayarlanmazsa hiç dikdörtgen çizmez veya çok küçük dikdörtgenler çizer.

- Plaka tespiti için birden fazla dikdörtgen çizilebilir. Bu durumda kaçıncı dikdörtgenin alınacağı "k" isminde indeks parametresi olarak verilir. Tipik olarak 0 veya 1 alınız.

- Eşikleme yapıldıktan sonra arkaplanı siyaha boyamak için boyanacak kooordinatlar bir liste biçiminde verilmiştir. Bu listenin elemanlarına ekleme yapılabilir. Bu listede siyaha boyanacak alanın x ve y noktası bulunur. Eğer o nokta zaten siyahsa başka bir koordinat seçilir ve orası siyaha boyanır. Genelde boyanması gereken koordinatlar eklenmiştir. Bu koordinatlar kullanılabilir. Ayrıca kaç defa siyaha boyanacağı da "fillcount" parametresiyle belirlenir. Çok vakit kaybetmemek için tipik olarak 1 veya 2 kullanılır.

- Boyama işleminden sonra gerekliyse yatay bir kırpma uygulanabilir. Bunun için "upper_crop_size" ve "lower_crop_size" parametreleri tipik olarak 5 veya 6 alınır.

- En son karakterleri kırparken aralarındaki beyaz piksellerin sayısına göre kırpılır. Bu beyaz piksellerin sayısı "maxWhite" parametresiyle belirlenir. Tipik olarak 1 veya 2 kullanın.


# libs
 Bu dizindeki python dosyaları ana kodun olduğu dizine alınmalıdır. Bu kütüphanelerden başlıca kullanılan modül LPR.py dosyasıdır. Plaka tanımaya ilişkin metotlar LPR sınıfında yazılmıştır. Diğer modüller daha çok test etme gibi işlemlerde işe yarıyor. Test işlemini kolaylaştırmak için görüntü tespiti için ayrı, video tespiti için ayrı metotlar yazılmıştır. 

LPR sınıfının 87.satırındaki koşul tespit edilecek dikdörtgenlerin hangilerinin dikkate alıncağıyla ilgilidir. Kameranın baktığı yere göre buradaki if koşulu güncellenmelidir. 
Ayrıca 117.satırdaki count değişkeni predict işleminin kaç defa yapılacağını belirtir. Biz çok vakit harcamamak adına 1 defa yapıyoruz.

 # Çalıştırılacak kodlar

 Görüntüde plaka tanıma yapılmak isteniyorsa lpr_on_image.py dosyası, videoda plaka tanıma yapılmak isteniyorsa lpr_on_video.py dosyası çalıştırılmalı. Eğitim modelleri ve veri seti gerekli dizine alınmalıdır. video için "garage2.mkv" isimli dosya kullandık. Başka videolar da kullanılabilir.


 # OCR ve Veri Üretimi

 Mevcut OCR modeli ortalama %80 doğruluk oranına sahiptir. Eğitim için kullanılan veri seti "" linkinde "mydataset" klasöründedir. Ancak doğru tahmin için sınıf bazında doğruluk oranı önemlidir. Şu an veri setindeki dağılım ve sınıf bazında ortalamalar aşağıdaki gibidir:

 ![class_accuracy](https://github.com/farukomeraydin/LPR/assets/59957778/0c2103dc-0cfb-4d33-9926-32791f2c450a)

 
![data_distribution](https://github.com/farukomeraydin/LPR/assets/59957778/4a544ba4-da2d-40d7-b0ce-1caa6d33fa1a)


Eğer veri seti geliştirilmek isteniyor ise (bu mutlaka gereklidir) main.py dosyası çalıştırılmalı. Bu dosyada yukarıda anlatılan plaka tanımaya ilişkin tüm adımlar uygun parametreler seçilerek gerçekleştirilir. Böylece OCR için daha çok görüntü oluşturulur. Veri üretiminden sonra ocr.py dosyası ile tekrardan eğitim yapılıp yeni model kaydedilebilir. Yeni üretilen veriler "mydataset" klasörüne eklenmelidir. Görüntülerin isimleri kaydedilirken "Etiketİsmi_görüntüSayısı" notasyonu uygulandı. Örneğin 9 karakterinin olduğu görüntü eklenecekse "9_77.jpg" gibi kaydedilmelidir. Bunun anlamı 9 karakterinden 77. görüntü veri setine eklenmiştir yani 77 tane 9 karakterinin görüntüsü var demektir.


# Kullanılan kütüphanelerin versiyonları

- tensorflow                    2.12.0
- scikit-learn                  1.2.2
- keras                         2.12.0
- numpy                         1.23.5
- pandas                        2.0.1
- scipy                         1.10.1
- seaborn                       0.12.2
- opencv-python                 4.7.0.72
- matplotlib                    3.7.1

DİKKAT: Yukarıdaki versiyonlar bire bir uyuşmak zorunda değildir. Sadece kayda geçsin diye yazılmıştır. Şimdiye kadar bir paket uyumsuzluğu sorunuyla karşılaşılmadı. Gerekli paketler normal "pip install" komutuyla indirilebilir.
