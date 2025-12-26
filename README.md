# Finetune Environment Documentation

Bu doküman, `../finetune` konumunda bulunan Python sanal ortamının (virtual environment) nasıl aktive edileceği, içeriği, sıfırdan nasıl oluşturulacağı ve **bu projenin nasıl kullanılacağı** hakkında bilgiler içerir.

## Kurulum (Eğer environment yoksa)

Eğer `../finetune` klasörü mevcut değilse, aşağıdaki komutlarla oluşturun:


```bash
python3 -m venv ../finetune
source ../finetune/bin/activate
```

**Donanımınıza uygun komutu seçin:**

*   **NVIDIA (CUDA 12.1):**
    ```bash
    pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
    ```

*   **AMD (ROCm 6.1):**
    ```bash
    pip install -r requirements.txt --index-url https://download.pytorch.org/whl/rocm6.1
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

```text
accelerate==1.12.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.2
aiosignal==1.4.0
anyio==4.12.0
attrs==25.4.0
bitsandbytes==0.49.0
certifi==2025.11.12
charset-normalizer==3.4.4
datasets==4.4.2
dill==0.4.0
filelock==3.20.0
frozenlist==1.8.0
fsspec==2025.10.0
h11==0.16.0
hf-xet==1.2.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==0.36.0
idna==3.11
Jinja2==3.1.6
MarkupSafe==2.1.5
mpmath==1.3.0
multidict==6.7.0
multiprocess==0.70.18
networkx==3.6.1
numpy==2.3.5
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvtx-cu12==12.1.105
packaging==25.0
pandas==2.3.3
peft==0.18.0
pillow==12.0.0
propcache==0.4.1
psutil==7.2.0
pyarrow==22.0.0
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.3
regex==2025.11.3
requests==2.32.5
safetensors==0.7.0
setuptools==70.2.0
six==1.17.0
sympy==1.13.1
tokenizers==0.22.1
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
torchvision==0.20.1+cu121
tqdm==4.67.1
transformers==4.57.3
triton==3.1.0
trl==0.26.2
typing_extensions==4.15.0
tzdata==2025.3
urllib3==2.6.2
xxhash==3.6.0
yarl==1.22.0
```

Yükleme komutu:

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```
