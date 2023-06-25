# Genel Akış

Plaka tespiti opencv'nin haar kaskat yapısı kullanılarak gerçekleştirildi. Eğitim için kullanılan veriler "" linkindedir. opencv kullanılarak yapılan bu eğitim "https://www.youtube.com/watch?v=XrCAvs9AePM&list=PL1m2M8LQlzfKtkKq2lK5xko4X-8EZzFPI&index=8" linkinden bakılarak yapılmıştır. Eğitim sırasında ROI'nin boyutu (24, 72) boyutunda belirlenmiştir. Plaka tespitinden sonra karakterlerin ayrıştırılması LPR sınıfındaki split metoduyla yapıldı. En son OCR aşaması konvolüsyonel derin ağ kullanılarak yapılmıştır. Plaka tespiti yaparken kameranın 105 dereceye ayarlandığı varsayılmıştır. Geriye kalan yerler kırpıldı. Baştan sona algoritmik olarak işlemler şöyledir: 

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
 Bu dizindeki python dosyaları main kodun olduğu dizine alınmalıdır. Bu kütüphanelerden başlıca kullanılan modül LPR.py dosyasıdır. Plaka tanımaya ilişkin metotlar LPR sınıfında yazılmıştır. Diğer modüller daha çok test etme gibi işlemlerde işe yarıyor. Test işlemini kolaylaştırmak için görüntü tespiti için ayrı, video tespiti için ayrı metotlar yazılmıştır.
