Merhaba;

Lstm nedir diye ingilizce güzel bir makale buldum türkçeye çevirmeye çalıştım. İstifadeye medar olması temennisi ile.

Makale Aslı: http://colah.github.io/posts/2015-08-Understanding-LSTMs/


Tekrarlayan Sinir Ağları
İnsanlar her saniye düşüncelerine yeniden başlamazlar. Bu makaleyi okuduğun zaman, Önceki kelimeleri anlamana bağlı olarak her kelimeyi anlarsın. Her şeyi atma ve tekrar yeniden düşünmeye başlama. Senin düşüncelerin kalıcı olsun. Geleneksel sinir sağları bunu yapmıyordu ve önemli eksikler görünüyor. Örneğin; bir filmde her noktada ne tür bir olayın olduğunu sınıflandırmak istediğinizi düşünün. Geleneksel bir sinir ağı, filmdeki önceki olaylarla ilgili mantığını nasıl geçersiz kılacağını açıkça belli değil. Tekrarlayan sinir ağları bu sorunu ele almaktadır. Bunlar, içinde döngüler bulunan ve bilginin devam etmesine izin veren ağlardır.
 
Tekrarlayan sinir ağları da döngüler vardır. Yukarıdaki diyagramda, “sinir ağı yığını”, “A” girişi (xt)  var ve çıkış değeri (ht) var. Bir döngü bir adımdan diğer sinir ağına bilgi gitmesine izin verir. Bu döngüler, tekrarlayan sinir ağlarının gizemli gibi görünmesini sağlar. Bununla birlikte, biraz daha düşünürseniz, normal bir sinir ağından farksız oldukları ortaya çıkıyor. Tekrarlayan bir sinir ağı aynı ağın birden fazla kopyası olarak düşünülür ve her biri bir halefi bir mesaj gönderir. Döngüyü açtığımızda ne olacağını düşünün:
 

Bu zincir benzeri doğa, tekrar eden sinir ağlarının diziler ve listelerle yakından alakalı olduğunu ortaya koymaktadır. Böyle data kullanmak için sinir ağının doğal bir mimaridirler. Ve kesinlikle kullanılıyor! Son birkaç yılda, RNN'leri çeşitli sorunlara uygulayarak inanılmaz bir başarı elde edildi: Konuşma tanıma, dil modelleme, çeviri, resim yazısı vb. Liste devam ediyor. Andrej Karpathy'nin mükemmel blog yazısı için RNN'lerle elde edebileceğiniz şaşırtıcı yeteneklerin tartışmasını bırakacağım, Tekrarlayan Yapay Sinir Ağlarının mantıksız etkinliği. Ama gerçekten çok şaşırtıcı. Bu başarıların esası "LSTM" lerin kullanılmasıdır. Birçok görev için standart sürümden çok daha iyi çalışan, tekrarlayan sinir ağı özel bir türü.  Tekrarlayan sinir ağlarına dayalı hemen hemen tüm heyecan verici sonuçlar LSTM ile birlikte başarıldı. Bu makale LSTM'ler incelenecektir. 

Uzun Vadeli Bağımlılık Sorunu

RNN'lerin cezbedici özelliklerinden biri, önceki görüntüleri mevcut görev ile ilişkilendirebilecekleri fikridir; örneğin, önceki video çerçevelerinin kullanılması mevcut çerçevenin anlaşılmasını sağlayabilir. RNNs bunu yapabilirse, son derece faydalı olurdu. Fakat yapabilirler mi? Bu değişir.
Bazen, şu anki görevi yerine getirmek için yalnızca son bilgilere bakmamız gerekir. Örneğin, önümüzdeki kelimeyi temel alarak bir sonraki kelimeyi tahmin etmeye çalışan bir dil modeli düşünün. Eğer "the clouds are in the sky (gökyüzünde bulutlar var.)" kelimelerinin son kelimesini tahmin etmeye çalışırsak; başka bir bağlama ihtiyaç duymayız - bir sonraki kelimenin “sky” gökyüzü olacağı açıktır. Bu gibi durumlarda, ilgili bilgi ile ihtiyaç duyulan yer arasındaki uçurumun az olduğu durumlarda, RNN'ler geçmiş bilgileri kullanmayı öğrenebilir.
 
Ancak daha fazla bağlama ihtiyaç duyduğumuz durumlar da var. Metindeki son kelimeyi tahmin etmeye çalışmayı düşünün (“I grew up in France… I speak fluent French.”) Yeni bilgiler, bir sonraki kelimenin muhtemelen bir dilin adı olduğunu önermektedir; ancak hangi dilin daraltılmasını istiyorsak, daha arka arkaya Fransa bağlamına ihtiyacımız var. İlgili bilgi ile bunun çok büyük olması gereken nokta arasındaki boşluk için tamamen mümkündür. Ne yazık ki, bu boşluk arttıkça, RNN'ler bilgiyi bağlamayı öğrenemiyorlar.
 
Teorik olarak, RNN'ler kesinlikle böyle "uzun vadeli bağımlılıkları" idare edebilmektedir. Bir insan, bu formun oyuncak problemlerini çözmek için kendileri için dikkatli bir şekilde parametreler seçebilir. Ne yazık ki, pratikte RNN'ler onları öğrenemiyor gibi görünüyor. Sorun, Hochreiter (1991) [Alman] ve Bengio ve diğerleri tarafından derinlemesine araştırılmıştır. (1994), neden zor olabileceğinin bazı temel sebeplerini bulmuşlardır. Neyse ki, LSTM'lerin bu sorunu yok! 

LSTM(Long- Short Term Memory)
Uzun Kısa Vadeli Hafıza Ağları - genellikle "LSTM'ler" olarak adlandırılır - uzun vadeli bağımlılıkları öğrenebilen özel bir RNN türüdür. Bunlar Hochreiter & Schmidhuber (1997) tarafından tanıtıldı ve aşağıdaki çalışmalarda pek çok kişi tarafından atıf aldı ve yaygınlaştırıldı. Çok çeşitli sorunlar üzerine muazzam derecede çalışırlar ve şu anda yaygın olarak kullanılmaktadırlar. LSTM'ler, uzun vadeli bağımlılık sorununun önüne geçmek için açıkça tasarlanmıştır. Uzun süre bilgi hatırlamak pratikte varsayılan davranışlarıdır, öğrenmek için uğraştıkları bir şey değildir. Tüm tekrarlayan sinir ağları, tekrar eden sinir ağı modül zinciri biçimindedir. Standart RNN'lerde, bu yinelenen modül, tek bir tanh katmanı gibi çok basit bir yapıya sahip olacaktır.
 
LSTM'lerin de art arda birbiri takip eden yapıları vardır. Ancak bir sonraki parça farklı bir yapıya sahiptir. Tek bir sinir ağı katmanı yerine, çok özel bir şekilde etkileşimde olan dört parça var.
 
Neler olup bittiğinin detayları hakkında endişelenmeyin. Şimdilik, kullanacağımız gösterimi rahatlatmaya çalışalım.
 
Yukarıdaki diyagramda, her satır bir düğümün çıktısından başkalarının girişlerine kadar tüm vektörü taşır. Sarı kutular sinir ağı katmanları öğrenilirken, pembe daireler vektör eklenmesi gibi noktasal işlemleri temsil eder. Birleştirilen satırlar birleştirme hattını, çizgi çatallaştırma ise kopyalanan içeriği ve kopyaları farklı yere gideceklerini belirtir.

LSTM'lerin Ardındaki Ana Fikir
LSTM'lerin anahtarı diyagramın üstünden geçen yatay çizgi çalışma hücre halidir. Hücre hali, bir konveyör kemeri gibidir. Tüm zincirin boyunca sadece küçük doğrusal etkileşimlerle çalışır. Bilgi hiç değişmeden zincir boyunca akması çok kolaydır.
 
LSTM, hücre durumu “kapılar denilen yapılarla”  bilgi ekleme veya kaldırma yeteneğine sahiptir. Kapılar, isteğe bağlı olarak bilgi vermenin bir yoludur. Bunlar sigmoid sinir ağı katmanı ve noktalı çarpma işleminden oluşur.
 
Sigmoid katman, sıfırdan bire kadar sayıları çıktı verir ve her bileşenin ne kadarına izin verileceğini açıklar. 1 değeri “her şeyin geçmesine izin ver” anlamına gelirken,  0 değeri “hiçbir şeyin geçmesine izin verme” anlamına gelir. LSTM, hücre durumunu korumak ve kontrol etmek için bu kapılardan üçüne sahiptir.
Adım Adım LSTM Gözden Geçirme
LSTM'mizin ilk adımı, hangi bilginin hücre durumundan atılacağına karar vermektir. Bu karar, "kapıyı unut katmanı" olarak adlandırılan bir sigmoid tabaka tarafından yapılır. xt  ve ht-1  bakar, Ct-1 ‘ye  hücre  durumunu her sayı için  0 ile 1 arasında bir çıkış değeri verir. Çıkış değeri 1 ise, "bunu tamamen tut" temsil ederken, 0  ise "tamamen bundan kurtulun" anlamına gelir. Bir sonraki kelimeyi önceki tüm kelimelere dayanılarak tahmin etmeye çalışan bir dil modeli örneğimize geri dönelim. Böyle bir sorunda; o hücre durumu mevcut konunun cinsiyetini içerebilir, böylece doğru zamirler kullanılabilir. Yeni bir konu gördüğümüzde, eski konunun cinsiyetini unutmak istiyoruz.
 
Bir sonraki adım, hücre durumuna hangi yeni bilgiyi depolayacağımıza karar vermektir. Bu iki bölümden oluşuyor. İlk olarak, "giriş katmanı katmanı" olarak adlandırılan bir sigmoid katman hangi değerleri güncelleyeceğimize karar verir. Sonra ki, bir tanh katmanı yeni aday değerler vektörü oluşturur, Ct, yeni duruma eklenebilir. Bir sonraki adımda, durumu güncellemek için bu ikisini birleştireceğiz. Dil modeli örneğinde; yeni konunun cinsiyetini hücre durumuna eklemek isteriz, unuttuğumuz eski cinsinin yerine geçer.
 
Artık eski hücre durumunu, Ct−1 yeni hücre durumun içine Ct güncelleyebiliriz. Daha önceki adımlar zaten ne yapılacağına karar verdik, sadece bunu yapmamız gerekir. 
\(f_t\) ile eski durumu çarptık, daha önceki kararımızı unutmaya yaradı. Sonra  \(i_t*\tilde{C}_t\) ekledik. Her bir durum değeri ne kadar güncellememize karar vermemizi ölçeklendirerek ve bir aday değer oluşmasını sağlar.
Dil model durumunda, bu önceki nesnenin cinsiyet bilgisini nerede atayacağımız sağlar ve yeni bir bilgi ekler, daha önceki aşamada karar verdiğimiz gibi.




 
Son olarak; çıkışa ne yollayacağımıza karar vermemiz gerekir. Bu çıkış; bizim hücre durumuna bağlıdır. Fakat filtre versiyonu olabilir. İlk olarak, hangi hücre durumunun hangi bölümlerine çıktı çıkarttığımıza karar veren bir sigmoid katmanı çalıştırırız. Daha sonra, hücrenin durumunu \ (\ tanh \) (değerler \ (- 1 \) ile \ (1 \) arasında olacak şekilde) ve sigmoid kapının çıktısıyla çarpın, böylece yalnızca karar verdiğimiz parçalar çıktı.
Dil modeli örneği için, yalnızca bir konuyu gördüğünden, bir fiille alakalı bilgiyi çıktı isteyebilir; Örneğin, konu tekil mi yoksa çoğul mu çıktı, böylece bir sonraki fiilin bir sonraki harfle birleşmesi gerektiğini biliyoruz.
 

Uzun Kısa Vadeli Bellek Üzerindeki Varyantlar
Şu ana dek açıkladığım şey oldukça normal bir LSTM’dir. Ancak tüm LSTM'ler yukarıdaki ile aynı değildir. Aslında, LSTM'leri içeren hemen hemen her makale de biraz farklı bir sürüm kullanıyor gibi görünüyor. Aralarında fazla farklılık yok fakat bazılarından bahsetmeye değer.
Gers & Schmidhuber (2000) tarafından geliştirilen  popüler LSTM’lerden biri olan "gözetleme deliği” ekliyor. Bu, geçiş katmanlarının hücre durumuna bakmasına izin verdiğimiz anlamına geliyor.
 
Yukarıdaki diyagram, tüm kapılara “gözetleme deliği ekliyor, ancak birçok kağıt bazı “gözetleme deliği” verecek değil, diğerleri.
Bir başka varyasyon, birleşmiş “unutma” kapıları ve giriş kapılarını kullanmaktır. Unutulacakları ve yeni bilgiler eklememiz gerektiğini ayrı ayrı belirlemek yerine, bu kararları birlikte alırız. Onun yerine bir şey girdiğimiz zamanları unutuyoruz. Eski bir şeyi unuttuğumuz zaman yalnızca yeni değerler giriyoruz.
 
LSTM'de biraz daha etkileyici bir varyasyon Gated Recurrent Unit veya GRU; (Cho et el) tarafında tanıtıldı. Unut ve giriş kapılarını tek bir "güncelleme kapısı" ‘na birleştiriyor. Aynı zamanda hücre durumunu ve gizli durumu birleştirir ve başka bazı değişiklikler yapar. Elde edilen model, standart LSTM modellerinden daha basittir ve giderek popüler hale gelmektedir.
 
Bunlar, en önemli LSTM varyantlarından yalnızca birkaçıdır. Daha bir çok varyasyonu vardır. Örneğin Yao’nun yaptığı Depth Gated RNN'leri  (2015). Ayrıca, Koutnik ve diğerleri tarafından Clockwork RNN'leri gibi uzun vadeli bağımlılıklarla mücadele etmek için tamamen farklı bir yaklaşım da var. (2014). Bu varyasyonlardan hangisi en iyisi? Farklılıklar önemli mi? Greff ve diğerleri. (2015), popüler varyantların güzel bir karşılaştırmasını yaparak bunların hepsinin aynı olduğunu fark ettiler. Jozefowicz ve diğ. (2015) ondan fazla RNN mimarisi test etti, bazıları belirli görevlerde LSTM'lerden daha iyi çalıştı.

Sonuç
İlk Olarak, insanların RNN'lerle elde ettikleri çarpıcı sonuçlardan bahsettim. Esasen hepsine LSTM kullanarak bu başarıya ulaştı. Birçok görev için gerçekten çok daha iyi çalışıyorlar. Bir denklemler kümesi olarak yazılmış olan LSTM'ler oldukça korkutucu görünüyorlar. Umarım, bu makalede adım adım inceleyerek onları biraz daha cana yakın yapmıştır.
LSTM'ler, RNN'lerle başarabileceğimiz şeylere büyük bir adım oldu. Merak etmek doğaldır; başka bir büyük adım var mı? Araştırmacılar arasında ortak bir görüş şudur ki: "Evet, bir sonraki adım var ve dikkat!" Fikir, bir RNN seçiminin her adımında bazı büyük bilgi topluluğundan bakılmasına izin vermektir. Örneğin, bir görüntüyü açıklayan bir başlık oluşturmak için bir RNN kullanıyorsanız, görüntünün çıktısını aldığı her kelimeye bakmak için resmin bir parçasını seçebilir. Aslında, Xu ve ark(2015) tam olarak bunu - eğer dikkatini keşfetmek istiyorsanız eğlenceli bir başlangıç noktası olabilir! Dikkatle dikkat çeken bir takım gerçekten heyecan verici sonuçlar çıktı ve çok daha fazla olduğu köşede.
Dikkat, RNN araştırmasında tek heyecan veren konu değildir. Örneğin, Kalchbrenner ve diğerleri tarafından Grid LSTM'ler  (2015) son derece umut verici görünüyor. Üretken modellerde RNN'leri kullanarak çalışma - Gregor ve ark. (2015), Chung ve diğ. (2015) veya Bayer & Osendorfer (2015) - de çok ilginç görünüyor. Son birkaç yıl tekrarlayan sinir ağları için heyecan verici bir zaman olmuştur ve gelen olanlar yalnızca daha çok söz veriyor.
Teşekkür
LSTM'leri daha iyi anlamak, görsel öğeler hakkında yorum yapmak ve bu yazı hakkında geri bildirim sağlamak için bana yardımcı olduğu için bir takım insanlara minnettarım. Google'daki meslektaşlarımıza özellikle Oriol Vinyals, Greg Corrado, Jon Shlens, Luke Vilnis ve Ilya Sutskever'a yararlı geribildirimlerini için teşekkür ediyorum. Dario Amodei ve Jacob Steinhardt da dahil olmak üzere bana yardımcı olmak için zaman ayırdığınız için diğer birçok arkadaşınıza ve meslektaşınıza da müteşekkirim. Kyunghyun Cho'ya diyagramlarım hakkında son derece düşünceli yazışmalar için özellikle minnettarım. Bu yazı öncesi, sinir ağları üzerinde öğrettiğim iki seminer dizisi boyunca LSTM'leri açıklamak için pratik yaptım. Bana sabrını ve geribildirimlerini verdikleri herkese teşekkürler.
