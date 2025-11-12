import time
import os

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_slide(title, content, wait=6):
    clear()
    print(f"\033[1;36m{title}\033[0m")
    print("=" * len(title))
    print(content)
    time.sleep(wait)

slides = [
    (
        "Machine Learning: An Overview",
        """
Machine Learning (ML) é o campo da Inteligência Artificial que
permite que sistemas aprendam padrões a partir de dados
para realizar previsões ou tomar decisões.

Aplicações:
- Recomendação de filmes e músicas
- Reconhecimento facial
- Modelos de linguagem (como GPT)
"""
    ),
    (
        "Supervised vs Unsupervised Learning",
        """
Supervised Learning:
  ➜ Treinado com dados rotulados (x, y)
  ➜ Exemplo: regressão linear, redes neurais

Unsupervised Learning:
  ➜ Descobre padrões sem rótulos
  ➜ Exemplo: k-means, PCA

Semi-supervised e Reinforcement Learning completam o quadro.
"""
    ),
    (
        "Pipeline de Treinamento",
        """
1. Coleta e limpeza de dados
2. Separação em treino, validação e teste
3. Escolha e configuração do modelo
4. Treinamento e ajuste de hiperparâmetros
5. Avaliação e deploy

Ferramentas comuns:
- Python, Scikit-learn, PyTorch, TensorFlow
"""
    ),
    (
        "Tokenization e Representações",
        """
Para modelos de linguagem, o texto precisa ser tokenizado:
convertido em números (tokens).

Tipos:
- Word-level (palavras)
- Subword (BPE, SentencePiece)
- Character-level

Essas representações são depois usadas em embeddings.
"""
    ),
    (
        "Transformers e Modelos Modernos",
        """
Modelos como GPT, BERT e T5 usam a arquitetura Transformer,
baseada em mecanismos de atenção.

Atenção permite ao modelo focar em partes relevantes do texto,
sem depender de estruturas sequenciais como RNNs.

Resultado: eficiência e escalabilidade em grandes corpora.
"""
    ),
    (
        "Conclusão",
        """
Machine Learning é uma das áreas mais dinâmicas da ciência moderna.

Revisamos:
- Tipos de aprendizado
- Pipeline de treinamento
- Tokenização e Transformers

Na próxima etapa: Implementar seu primeiro modelo supervisionado!
"""
    )
]

if __name__ == "__main__":
    for title, content in slides:
        show_slide(title, content, wait=7)
    clear()
    print("🎓 Apresentação concluída! Obrigado por assistir.\n")
