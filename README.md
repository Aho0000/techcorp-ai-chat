# TechCorp AI Chat - Chatbot Medical avec Phi-3.5

Chatbot conversationnel spécialisé en médecine utilisant le modèle Phi-3.5-Financial fine-tuné avec LoRA.

## 🎯 Projet

Système de chat intelligent capable de répondre à des questions médicales en utilisant:
- **Modèle de base**: Phi-3.5-Financial (3.8B paramètres)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) sur données médicales
- **Dataset**: 191K paires instruction-input-output médicales

## 📁 Structure

```
techcorp-ai-chat/
├── models/                          # Modèles
│   ├── phi35-financial-optimized/   # Config d'inférence optimisée
│   └── phi35-medical-lora/          # Poids LoRA médical
├── medical_dataset/                 # Datasets
│   ├── medical_lora_train.json      # 191K samples d'entraînement
│   ├── medical_lora_val.json        # Validation
│   ├── medical_lora_test.json       # Test
│   └── medical_datase_clean.parquet # Dataset nettoyé
├── scripts/                         # Scripts d'entraînement
│   ├── CREATE_LORA_WEIGHTS.py       # Fine-tuning LoRA
│   ├── DATA-Phase*.py               # Nettoyage/validation données
├── tritton_server/                  # Configuration Triton (futur)
├── docs/                            # Documentation
└── requirements.txt
```

## 🚀 Démarrage Rapide

### 1. Installation

```bash
git clone https://github.com/YOUR_USERNAME/techcorp-ai-chat.git
cd techcorp-ai-chat
pip install -r requirements.txt
```

### 2. Charger le Modèle

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3.5-mini-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-3.5-mini-instruct")
model = PeftModel.from_pretrained(model, "./models/phi35-medical-lora")

# Charger config d'inférence
import json
with open("./models/phi35-financial-optimized/inference_config.json") as f:
    config = json.load(f)
```

### 3. Utiliser le Chatbot

```python
def chat(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=config["inference_parameters"]["max_tokens"],
        temperature=config["inference_parameters"]["temperature"],
        top_p=config["inference_parameters"]["top_p"]
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
response = chat("What is fever in medical terms?")
print(response)
```

## 📊 Datasets

### Medical LoRA Dataset
- **Train**: 191,543 exemples
- **Val**: 24,393 exemples
- **Test**: 24,393 exemples
- **Format**: {instruction, input, output}

### Dataset Clean
- Format Parquet nettoyé
- 122 MB
- Prêt pour analyse

## ⚙️ Configuration Optimisée

Paramètres d'inférence optimisés pour médical:
```json
{
  "temperature": 0.6,
  "top_p": 0.85,
  "max_tokens": 110,
  "top_k": 50
}
```

## 🔧 Fine-tuning LoRA

Configuration LoRA:
- **Rank (r)**: 8
- **Alpha**: 32
- **Target Modules**: q_proj, v_proj, k_proj, dense, fc1, fc2
- **Dropout**: 0.05
- **Trainable Parameters**: ~2.6M (0.069% du modèle total)

## 📚 Scripts Disponibles

- `CREATE_LORA_WEIGHTS.py` - Fine-tuning du modèle avec LoRA
- `DATA-Phase1-Data Exploration.py` - Exploration des données
- `DATA-Phase2-Data Cleaning.py` - Nettoyage des données
- `DATA-Phase3-Validation.py` - Validation de qualité
- `DATA-Phase4-LoRA Preparation.py` - Préparation pour LoRA

## 📖 Documentation

Voir le dossier `docs/` pour:
- `RAPPORT IA.html` - Résultats du fine-tuning
- `RAPPORT DATA.html` - Analyse des données
- Guides détaillés

## 🔄 Workflow Complet

```
1. Données brutes (medical_datase_clean.parquet)
   ↓ (DATA-Phase1-4)
2. Données nettoyées et validées
   ↓
3. Préparation pour LoRA (medical_lora_*.json)
   ↓
4. Fine-tuning avec LoRA (CREATE_LORA_WEIGHTS.py)
   ↓
5. Modèle spécialisé en médical (adapter_model.bin)
   ↓
6. Déploiement du chatbot
```

## 🎯 Cas d'Utilisation

- Questions médicales générales
- Symptomatologie
- Prévention et traitement
- Termes médicaux
- Conseils de santé

## ⚠️ Notes Importantes

- Ce modèle est **expérimental** (R&D)
- Les réponses doivent être validées par un professionnel de santé
- Ne remplace pas une consultation médicale

## 📝 Licence

À définir

## 👥 Équipes

- **IA**: Fine-tuning et optimisation
- **DATA**: Préparation et validation données
- **INFRA**: Déploiement serveur (Triton/Ollama)
- **DEV WEB**: Interface utilisateur

## 🔗 Ressources

- [Phi-3.5 Documentation](https://huggingface.co/microsoft/phi-3.5-mini-instruct)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

**Status**: MVP Phase (R&D)  
**Dernière mise à jour**: Juin 2026
