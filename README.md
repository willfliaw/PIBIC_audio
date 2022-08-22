# PIBIC_audio

## Tabela de conteúdos

- [PIBIC_audio](#pibic_audio)
  - [Tabela de conteúdos](#tabela-de-conteúdos)
  - [Sobre o projeto](#sobre-o-projeto)
  - [Estrutura dos arquivos deste repositório](#estrutura-dos-arquivos-deste-repositório)
  - [Preliminar](#preliminar)
    - [Pré-requisitos](#pré-requisitos)
    - [Instalação](#instalação)
  - [Execução e teste](#execução-e-teste)
  - [Autores](#autores)
  - [Agradecimentos](#agradecimentos)

## Sobre o projeto

O presente repositório destina-se ao armazenamento de arquivos referentes ao projeto Análise e Síntese de Sinais de Voz incorporando Aprendizagem Profunda, de código 943, do Programa Institucional de Bolsas de Iniciação Científica (PIBIC), 2021 - 2022, pela Escola Politécnica da Universidade de São Paulo (USP). Tal trabalho acadêmico tem como objetivo o estudo e pesquisa sobretudo no que tange ao processamento e reconhecimento por Aprendizagem de Máquina (ML, *Machine Learning*) da expressão emocional humana via sinais de voz, com especial interesse para métodos avaliativos.

## Estrutura dos arquivos deste repositório

```bash
PIBIC_audio
│   .gitignore
│   eval_preds.ipynb
│   README.md
│   finetune_models.py
│   preprocess_data.py
│   pretrained_models.py
│   select_features.py
│   test_eval_models.py
│   train_models.py
│   environment.yml
│
└───data
│   │
│   └───alumni
│   │   │   README.md
│   │
│   └───EmoDB
│   │   │   README.md
│   │
│   └───general
│   │   │   README.md
│   │
│   └───RAVDESS_songs
│   │   │   README.md
│   │
│   └───RAVDESS_speeches
│   │   │   README.md
│   │
│   └───Soundtracks
│       │   README.md
│
│
└───utils
    │   __init__.py
    │   config.py
    │   dataset_reading.py
    │   feature_extraction.py
    │   feature_selection.py
    │   model_evaluation.py
    │   models.py
```

## Preliminar

### Pré-requisitos

Para utilizar os conteúdos do presente repositório é necessário que se instale uma versão de [Python 3](https://www.python.org/), e suas dependências. Além disso, recomenda-se veementemente a instalação de um editor de texto ou Ambiente de Desenvolvimento Integrado (IDE, *Integrated Development Environment*), tal qual o [VSCodium](https://vscodium.com/), e, ainda, de um gerenciador de pacotes de ciência de dados para Python [Anaconda](https://anaconda.org/), como o [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### Instalação

```bash
# Clone este repositório
git clone https://github.com/willfliaw/PIBIC_audio.git

# Acesse o diretório do projeto na linha de comando
cd PIBIC_audio
```

Para executar, testar e contribuir com o material do projeto aqui presente, convém a criação de um ambiente virtual (*virtual environment*). Para tanto, existem diversas formas, dentre as quais se destacam:

1. Via arquivo `environment.yml`:

   ```bash
   conda env create --name PIBIC_audio --file ./environment.yml
   ```

2. Pela instalação individual das dependências necessárias:

   ```bash
    conda create -n PIBIC_audio -c conda-forge python datasets ffmpeg ipykernel ipython joblib jupyter keras librosa matplotlib numpy pandas pytorch SciencePlots scikit-learn seaborn tensorflow=2.6.0=gpu_py39he88c5ba_0 tensorflow-estimator=2.6.0 tqdm transformers[tf-cpu]
   ```

Vale frisar que as duas maneiras apresentadas foram concebidas, em 5 jan. 2022, com a intenção de se utilizar a GPU (*Graphics Processing Unit*) da máquina local. Recomenda-se, assim, que o usuário confira em [Install TensorFlow 2](https://www.tensorflow.org/install/) os pré-requisitos para a instalação. Ressalta-se, ainda, que é de importância sobremaneira que os pacotes (*packages*) `tensorflow` e `tensorflow-estimator` apresentem a mesma versão. É interessante que se utilize o comando `conda search -c conda-forge <package>` para averiguar as versões e *builds* disponíveis.

## Execução e teste

O material aqui presente foi testado no sistema operacional Windows 11 x64, apenas.

<!-- INCOMPLETO -->

## Autores

Orientador: Prof. Dr. Miguel Arjona Ramirez; Co-orientador: Prof. Dr. Wesley Beccaro; William Liaw.

## Agradecimentos

- À Universidade de São Paulo, pela dedicação e pelo financiamento no fomento à pesquisa científica.
- Aos professores, pelas correções e ensinamentos que me possibilitaram um melhor desenvolvimento acadêmico.
