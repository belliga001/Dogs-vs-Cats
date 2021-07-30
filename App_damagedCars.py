{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "App_damagedCars.ipynb",
      "provenance": [],
      "machine_shape": "hm",
      "authorship_tag": "ABX9TyPe81uXyvu2dLjkA+KRNsex",
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
        "<a href=\"https://colab.research.google.com/github/belliga001/Dogs-vs-Cats/blob/master/App_damagedCars.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YZqEDVnWEqYV",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "51579ab7-6c6d-45a8-983b-ae7c3178bac9"
      },
      "source": [
        "!pip install -q streamlit\n",
        "#!pip install streamlit\n",
        "#lembrar de entrar em ambiente de execução depois e reiniciar ambiente de execução"
      ],
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\u001b[K     |████████████████████████████████| 7.9 MB 12.8 MB/s \n",
            "\u001b[K     |████████████████████████████████| 75 kB 3.8 MB/s \n",
            "\u001b[K     |████████████████████████████████| 111 kB 81.3 MB/s \n",
            "\u001b[K     |████████████████████████████████| 4.2 MB 63.4 MB/s \n",
            "\u001b[K     |████████████████████████████████| 170 kB 76.1 MB/s \n",
            "\u001b[K     |████████████████████████████████| 122 kB 72.7 MB/s \n",
            "\u001b[K     |████████████████████████████████| 786 kB 35.7 MB/s \n",
            "\u001b[K     |████████████████████████████████| 368 kB 74.6 MB/s \n",
            "\u001b[K     |████████████████████████████████| 63 kB 1.9 MB/s \n",
            "\u001b[?25h  Building wheel for blinker (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
            "jupyter-console 5.2.0 requires prompt-toolkit<2.0.0,>=1.0.0, but you have prompt-toolkit 3.0.19 which is incompatible.\n",
            "google-colab 1.0.0 requires ipykernel~=4.10, but you have ipykernel 6.0.3 which is incompatible.\n",
            "google-colab 1.0.0 requires ipython~=5.5.0, but you have ipython 7.25.0 which is incompatible.\u001b[0m\n"
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
        "outputId": "14211edd-0e56-4f57-fda5-503250edba3c"
      },
      "source": [
        "import streamlit as st\n",
        "\n",
        "from PIL import Image, ImageOps\n",
        "\n",
        "import numpy as np\n",
        "import tensorflow as tf"
      ],
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "2021-07-30 18:59:21.105815: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0\n"
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
      "execution_count": 2,
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
      "execution_count": 4,
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
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1HnIaiyCHpDI",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4ce2ddbe-128a-4cbd-e33c-f1d6886d0cdd"
      },
      "source": [
        "#model = tf.keras.models.load_model(\"G:/Meu Drive/Bases_Colab/dogsandcats/model.h5\")\n",
        "model=tf.keras.models.load_model(\"modeldemaged.h5\")\n",
        "#A unica maneira de funcionar foi de fazer o upload do modelo nos arquivos."
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "2021-07-30 19:03:02.616178: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcuda.so.1\n",
            "2021-07-30 19:03:02.715022: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
            "2021-07-30 19:03:02.715778: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties: \n",
            "pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0\n",
            "coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.78GiB deviceMemoryBandwidth: 836.37GiB/s\n",
            "2021-07-30 19:03:02.715825: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0\n",
            "2021-07-30 19:03:02.826328: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcublas.so.11\n",
            "2021-07-30 19:03:02.826464: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcublasLt.so.11\n",
            "2021-07-30 19:03:03.002312: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcufft.so.10\n",
            "2021-07-30 19:03:03.044180: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcurand.so.10\n",
            "2021-07-30 19:03:03.319336: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcusolver.so.10\n",
            "2021-07-30 19:03:03.350084: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcusparse.so.11\n",
            "2021-07-30 19:03:03.360537: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudnn.so.8\n",
            "2021-07-30 19:03:03.360700: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
            "2021-07-30 19:03:03.361373: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
            "2021-07-30 19:03:03.365864: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0\n",
            "2021-07-30 19:03:03.367058: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX512F\n",
            "To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.\n",
            "2021-07-30 19:03:03.367478: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
            "2021-07-30 19:03:03.368214: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties: \n",
            "pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0\n",
            "coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.78GiB deviceMemoryBandwidth: 836.37GiB/s\n",
            "2021-07-30 19:03:03.368335: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
            "2021-07-30 19:03:03.369031: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
            "2021-07-30 19:03:03.369657: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0\n",
            "2021-07-30 19:03:03.374498: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0\n",
            "2021-07-30 19:03:05.997170: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1258] Device interconnect StreamExecutor with strength 1 edge matrix:\n",
            "2021-07-30 19:03:05.997219: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1264]      0 \n",
            "2021-07-30 19:03:05.997229: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1277] 0:   N \n",
            "2021-07-30 19:03:05.997442: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
            "2021-07-30 19:03:05.998056: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
            "2021-07-30 19:03:05.998654: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
            "2021-07-30 19:03:05.999239: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.\n",
            "2021-07-30 19:03:05.999291: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1418] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14682 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0)\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ADWSzWVVFBMR",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a834accf-34d4-49c2-dc90-affd4ec52ebe"
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
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "2021-07-30 19:03:15.479 \n",
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
      "execution_count": 8,
      "outputs": []
    }
  ]
}