# Finetune Environment Documentation

Bu doküman, `../finetune` konumunda bulunan Python sanal ortamının (virtual environment) nasıl aktive edileceği, içeriği, sıfırdan nasıl oluşturulacağı ve **bu projenin nasıl kullanılacağı** hakkında bilgiler içerir.

## Kurulum (Eğer environment yoksa)

Eğer `../finetune` klasörü mevcut değilse, aşağıdaki komutlarla oluşturun:


```bash
python3.12 -m venv ../finetune
source ../finetune/bin/activate
```

**Donanımınıza uygun komutu seçin:**

1.  **Önce PyTorch'u kurun:**

    *   **NVIDIA (CUDA 12.1):**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

    *   **AMD (ROCm 6.2.4):**
        ```bash
        pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
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

## Önemli Not: AMD 780M / RDNA3 Kullanıcıları İçin
Eğer **"HIP error: invalid device function"** hatası alırsanız, komutların başına şu ortam değişkenini ekleyerek çalıştırın:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 ...
```

Örneğin:
```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python train.py
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

## Eğitim Kalitesini Ayarlama

Eğitim kalitesini ve performansını ayarlamak için `train.py` dosyasındaki aşağıdaki parametreleri değiştirebilirsiniz:

### LoRA Konfigürasyonu (lora_config)

```python
lora_config = LoraConfig(
    r=16,                    # LoRA rank (8-64 arası, yüksek = daha fazla parametre)
    lora_alpha=32,           # LoRA alpha (genelde r*2, öğrenme hızını etkiler)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Hangi katmanlar eğitilecek
    lora_dropout=0.05,       # Dropout oranı (0.0-0.1, overfitting'i önler)
    bias="none",             # Bias eğitimi ("none", "all", "lora_only")
    task_type="CAUSAL_LM"
)
```

**Öneriler:**
- **Daha iyi kalite:** `r=32`, `lora_alpha=64` (daha yavaş, daha fazla bellek)
- **Daha hızlı eğitim:** `r=8`, `lora_alpha=16` (daha az parametre)
- **Overfitting varsa:** `lora_dropout=0.1` artırın

### Eğitim Parametreleri (SFTConfig)

```python
training_args = SFTConfig(
    per_device_train_batch_size=1,      # Batch size (1-4, GPU belleğine göre)
    gradient_accumulation_steps=16,     # Gradient biriktirme (efektif batch = batch_size * bu değer)
    num_train_epochs=30,                # Epoch sayısı (10-50 arası)
    learning_rate=1e-4,                 # Öğrenme hızı (1e-5 ile 5e-4 arası)
    max_grad_norm=0.3,                  # Gradient clipping (0.3-1.0 arası)
    logging_steps=1,                    # Her kaç adımda log basılacak
    dataloader_num_workers=2,           # Veri yükleme thread sayısı
)
```

**Öneriler:**
- **Daha iyi öğrenme:** `num_train_epochs=50`, `learning_rate=2e-4`
- **Daha hızlı eğitim:** `num_train_epochs=10`, batch_size artırın (GPU belleği yeterse)
- **Kararsız eğitim:** `learning_rate=5e-5` düşürün, `max_grad_norm=0.5` artırın
- **Efektif batch size:** `batch_size * gradient_accumulation_steps` = 16-32 olmalı

### Veri Kalitesi

- **Daha fazla veri:** Daha iyi genelleme
- **Çeşitli örnekler:** Farklı soru tipleri ekleyin
- **Temiz veri:** Tutarsız veya hatalı örnekleri temizleyin
- **Dengeli dağılım:** Her kategoriden benzer sayıda örnek

### Performans vs Kalite Dengesi

| Ayar | Hız | Kalite | Bellek |
|------|-----|--------|--------|
| `r=8, epochs=10` | ⚡⚡⚡ | ⭐⭐ | 💾 |
| `r=16, epochs=30` | ⚡⚡ | ⭐⭐⭐ | 💾💾 |
| `r=32, epochs=50` | ⚡ | ⭐⭐⭐⭐ | 💾💾💾 |

---

## Yüklü Kütüphaneler

Aşağıda ortamda yüklü olan temel kütüphaneler ve versiyonları listelenmiştir:

| Kütüphane | Versiyon |
|-----------|----------|
| python | 3.12 |
| torch | 2.6.0+rocm6.2.4 |
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
```bash
python3.12 -m venv finetune
source finetune/bin/activate
```

2. **Gerekli kütüphaneleri yükleyin:**

PyTorch (CUDA 12.1 destekli) ve diğer temel yapay zeka kütüphaneleri için:

```bash
# Önce pip'i güncelleyin
pip install --upgrade pip

# PyTorch Kurulumu (AMD ROCm 6.2.4)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2.4

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

    *   **AMD (ROCm 6.2.4):**
        ```bash
        pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
        ```
    
    *   **CPU:**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```

2.  **Diğer kütüphaneleri yükleyin:**

    ```bash
    pip install -r requirements.txt
    ```




