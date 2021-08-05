{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Cópia de App_teste.py",
      "provenance": [],
      "collapsed_sections": [],
      "machine_shape": "hm",
      "authorship_tag": "ABX9TyM7GzHfprn/q8fISHXjKidO",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/belliga001/Dogs-vs-Cats/blob/master/C%C3%B3pia_de_App_teste.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YZqEDVnWEqYV",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1e9746e0-00fb-4bde-c436-80ffa4ee1ada"
      },
      "source": [
        "!pip install -q streamlit\n",
        "#!pip install streamlit\n",
        "#lembrar de entrar em ambiente de execução depois e reiniciar ambiente de execução"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\u001b[K     |████████████████████████████████| 7.9 MB 12.8 MB/s \n",
            "\u001b[K     |████████████████████████████████| 4.2 MB 55.1 MB/s \n",
            "\u001b[K     |████████████████████████████████| 170 kB 60.9 MB/s \n",
            "\u001b[K     |████████████████████████████████| 75 kB 4.1 MB/s \n",
            "\u001b[K     |████████████████████████████████| 111 kB 75.1 MB/s \n",
            "\u001b[K     |████████████████████████████████| 122 kB 69.8 MB/s \n",
            "\u001b[K     |████████████████████████████████| 786 kB 67.0 MB/s \n",
            "\u001b[K     |████████████████████████████████| 368 kB 70.6 MB/s \n",
            "\u001b[K     |████████████████████████████████| 63 kB 1.9 MB/s \n",
            "\u001b[?25h  Building wheel for blinker (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
            "jupyter-console 5.2.0 requires prompt-toolkit<2.0.0,>=1.0.0, but you have prompt-toolkit 3.0.19 which is incompatible.\n",
            "google-colab 1.0.0 requires ipykernel~=4.10, but you have ipykernel 6.0.3 which is incompatible.\n",
            "google-colab 1.0.0 requires ipython~=5.5.0, but you have ipython 7.26.0 which is incompatible.\u001b[0m\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vWJFxbkDD6Cc",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "73e1441f-799b-455a-8074-4b754f5a6623"
      },
      "source": [
        "import streamlit as st\n",
        "\n",
        "from PIL import Image, ImageOps\n",
        "\n",
        "import numpy as np\n",
        "import tensorflow as tf"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "2021-08-05 12:59:12.888645: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uWO1yirfEWE-"
      },
      "source": [
        "#st.set_option('deprecation.showfileUploaderEncoding',False)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pQc98NLzFin9"
      },
      "source": [
        "#%%time\n",
        "#from google.colab import drive\n",
        "#drive.mount(\"/content/gdrive\")"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Dh2q81F9EctW"
      },
      "source": [
        "#@st.cache(allow_output_mutation=True)\n",
        "#def load_model():\n",
        "#model = tf.keras.models.load_model(\"G:/Meu Drive/Bases_Colab/dogsandcats/model.h5\")\n",
        "#return model"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1HnIaiyCHpDI"
      },
      "source": [
        "#model = tf.keras.models.load_model(\"G:/Meu Drive/Bases_Colab/dogsandcats/model.h5\")\n",
        "#model=tf.keras.models.load_model(\"modeldemaged.h5\")\n",
        "#A unica maneira de funcionar foi de fazer o upload do modelo nos arquivos."
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ADWSzWVVFBMR",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "3ae77ebe-d809-40c2-8def-8922ddbc3b3d"
      },
      "source": [
        "st.markdown(\"# Aplicativo da web de predição de cães x gatos\", unsafe_allow_html = True)\n",
        "st.markdown(\"---\", unsafe_allow_html = True)\n",
        "st.markdown(\"* Este é um aplicativo da web simples que prevê se a imagem que o usuário carrega contém um carro quebrado.\", unsafe_allow_html = True)\n",
        "st.markdown(\"---\", unsafe_allow_html = True)\n",
        "\n",
        "st.markdown(\"# Predictor\", unsafe_allow_html = True)\n",
        "st.markdown(\"---\", unsafe_allow_html = True)\n",
        "st.markdown(\"* Faça upload de um arquivo de imagem abaixo e clique no botão 'Prever' que aparece abaixo da imagem enviada para fazer as previsões.\", unsafe_allow_html = True)\n",
        "uploaded_file = st.file_uploader(\"Imagem de carro quebrado / intacto a ser carregada.\", type=['png','jpeg','jpg'])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "2021-08-05 12:59:26.333 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py [ARGUMENTS]\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8HD_QF7KFF-K"
      },
      "source": [
        "if uploaded_file is not None:\n",
        "\n",
        "    st.write(\"File carregado! File type: \"+uploaded_file.type+\".\")\n",
        "    \n",
        "    image = Image.open(uploaded_file)\n",
        "    st.image(image, caption = 'File carregado.', use_column_width = True)\n",
        "    \n",
        "    bl = st.button(\"Prever\")\n",
        "    \n",
        "    if bl:\n",
        "        \n",
        "        size = (150, 150)\n",
        "        \n",
        "        image = np.asarray(image)\n",
        "        image = tf.image.resize(image, [150, 150])\n",
        "        image = np.asarray(image)\n",
        "        image = np.reshape(image, (1, 150, 150, 3))\n",
        "        image = image.copy()\n",
        "        \n",
        "        image /= 255\n",
        "        \n",
        "        label = model.predict_classes(image)\n",
        "        \n",
        "        label = label[0][0]\n",
        "                          \n",
        "        if label==1:\n",
        "            st.markdown(\"* ### A imagem é um carro quebrado.\", unsafe_allow_html = True)\n",
        "        else:\n",
        "            st.markdown(\"* ### A imagem é um carro intacto\", unsafe_allow_html = True)"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}