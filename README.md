## Mamba vs Transformer

### State Space Model

#### 1 etap
fine-tuning lub implementacja i trening od zera Transformera i Mamby
#### 2 etap Analiza skalowania
- porównanie architektur dla różnych długości sekwencji (128, 512, 1024 tokenów)
- czas trenowanie epoki
- czas inferencji
- accuracy, F1
#### Datasety
EdinburghNLP/xsum

[1808.08745](https://arxiv.org/pdf/1808.08745) 2018

### Metryka

- **ROUGE-1 (R1)**  
- **ROUGE-2 (R2)**  
- **ROUGE-L (RL)**  
  
All scores are reported as **F1 scores**.
#### F1 Score  
$$  
F_1 = 2 \cdot \frac{precision \cdot recall}{precision + recall}  
$$
#### Precision  
  
$$  
Precision = \frac{\text{overlapping n-grams}}{\text{generated n-grams}}  
$$  
#### Recall  
$$  
Recall = \frac{\text{overlapping n-grams}}{\text{reference n-grams}}  
$$
- ROUGE-2 → dla bigramów,
- ROUGE-L → dla longest common subsequence.
#### Zasoby
oficjalne repozytorium Mamba, pretrenowane model state-spaces
### Sprawozdanie 
- Przegląd literatury
- Opis rozwiązania
- Wyniki ewaluacji eksperymentalnej


### Planowane eksperymenty

- Na przykładzie mamby porównanie constatnst learning rate, a cosine.

-  Analiza skalowania
• Porównanie obu architektur dla różnych długości sekwencji (128, 512, 1024 tokenów).
• Czas trenowania epoki.
• Czas inferencji.


python3 main.py --model "state-spaces/mamba-130m-hf" --rank 16 --epoch 10 
