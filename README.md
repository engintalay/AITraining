# Finetune Environment Documentation

Bu doküman, `../finetune` konumunda bulunan Python sanal ortamının (virtual environment) nasıl aktive edileceği, içeriği, sıfırdan nasıl oluşturulacağı ve **bu projenin nasıl kullanılacağı** hakkında bilgiler içerir.

## Kurulum (Eğer environment yoksa)

Eğer `../finetune` klasörü mevcut değilse, aşağıdaki komutlarla oluşturun:


```bash
python3 -m venv ../finetune
source ../finetune/bin/activate
```

**Donanımınıza uygun komutu seçin:**

1.  **Önce PyTorch'u kurun:**

    *   **NVIDIA (CUDA 12.1):**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

    *   **AMD (ROCm 6.1):**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
        ```
    
    *   **CPU:**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```

2.  **Diğer kütüphaneleri yükleyin:**

    ```bash
    pip install -r requirements.txt
    ```

*   **Vulkan:**
    *(Not: Standart PyTorch pip paketlerinde Vulkan desteği bulunmamaktadır. Kaynak kodundan derleme gerektirebilir.)*


(Detaylı kütüphane listesi için dokümanın sonundaki [Ortamı Sıfırdan Kurma](#ortamı-sıfırdan-kurma) bölümüne bakabilirsiniz.)

## Ortamı Aktive Etme

Mevcut `finetune` ortamını aktive etmek için terminalde şu komutu çalıştırın (AITraining klasöründe olduğunuz varsayılmıştır):

```bash
source ../finetune/bin/activate
```

Ortamdan çıkmak için:

```bash
deactivate
```

## Proje Kullanımı

Bu proje, **TinyLlama-1.1B** modelini `data.json` içerisindeki verilerle eğitmek (finetune) ve test etmek için geliştirilmiştir.

### 1. Modeli Eğitme (Finetuning)

Modeli `data.json` verisi ile eğitmek için şu komutu çalıştırın:

```bash
python train.py
```

Varsayılan olarak `data.json` dosyasını kullanır. Farklı bir veri dosyası kullanmak isterseniz:

```bash
python train.py benim_datam.json
```


Bu işlem tamamlandığında, eğitilmiş model (adapter) dosyaları `./out` klasörüne kaydedilecektir.
Eğitim süresince `TrainingArguments` kullanıldığı için checkpointler de burada saklanır.

### 2. Modeli Test Etme (Inference)

Eğitilmiş modeli test etmek için `test.py` dosyasını kullanın. Bu dosya `./out` klasöründeki adapter'ı ve base modeli yükler, bir soru sorar ve cevabı ekrana basar.

```bash
python test.py
```

*Not: Eğer `out` klasörü yoksa veya boşsa, önce eğitimi çalıştırmanız gerekir.*

### 3. Base Model ile Test Etme

Eğitimden önceki (ham) modelin nasıl cevap verdiğini görmek için `test_base.py` dosyasını kullanabilirsiniz. Bu script, herhangi bir LoRA adapter kullanmadan saf TinyLlama modelini çalıştırır.

```bash
python test_base.py
```

### 4. Veri Seti

`data.json` dosyası, eğitim için kullanılan soru-cevap çiftlerini içerir. Formatı şöyledir:

```json
[
    {
        "instruction": "Soru...",
        "response": "Cevap..."
    },
    ...
]
```

---

## Yüklü Kütüphaneler

Aşağıda ortamda yüklü olan temel kütüphaneler ve versiyonları listelenmiştir:

| Kütüphane | Versiyon |
|-----------|----------|
| python | 3.x |
| torch | 2.5.1+cu121 |
| transformers | 4.57.3 |
| datasets | 4.4.2 |
| peft | 0.18.0 |
| trl | 0.26.2 |
| bitsandbytes | 0.49.0 |
| accelerate | 1.12.0 |
| huggingface-hub | 0.36.0 |

(Tam liste aşağıda `requirements.txt` bölümünde mevcuttur.)

## Ortamı Sıfırdan Kurma

Bu ortamı sıfırdan oluşturmak isterseniz aşağıdaki adımları takip edebilirsiniz.

1. **Yeni bir sanal ortam oluşturun:**

```bash
python3 -m venv finetune
source finetune/bin/activate
```

2. **Gerekli kütüphaneleri yükleyin:**

PyTorch (CUDA 12.1 destekli) ve diğer temel yapay zeka kütüphaneleri için:

```bash
# Önce pip'i güncelleyin
pip install --upgrade pip

# PyTorch Kurulumu (CUDA 12.1)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Diğer Kütüphaneler
pip install transformers==4.57.3 datasets==4.4.2 peft==0.18.0 trl==0.26.2 bitsandbytes==0.49.0 accelerate==1.12.0 huggingface-hub==0.36.0 python-dateutil==2.9.0.post0 pytz==2025.2 six==1.17.0
```

Alternatif olarak, aşağıdaki içeriği `requirements.txt` dosyasına kaydedip tek komutla yükleyebilirsiniz:

**requirements.txt içeriği:**

(Güncel tam liste için `requirements.txt` dosyasına bakınız.)

Yükleme komutu:

1.  **Önce PyTorch'u kurun:**

    *   **NVIDIA (CUDA 12.1):**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

    *   **AMD (ROCm 6.1):**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
        ```
    
    *   **CPU:**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```

2.  **Diğer kütüphaneleri yükleyin:**

    ```bash
    pip install -r requirements.txt
    ```




