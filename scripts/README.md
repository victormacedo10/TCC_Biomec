# Dependencias:
    - Python3
    - Numpy
    - Matplotlib
    - Opencv 3
    - Pykalman
    - Pandas
    - Json
    - 

# Preparar repositorio

    sudo ./prepareRepo.sh   

# Colorir Imagem

Colocar o video na pasta Videos
Executar todos os passos:

    cd scripts
    python3 colorize_image.py nome_video.mp4

Caso os passos de detecção (D), preparação (P) ou filtragem (F) já tenham sido executados anteriormente, utilize um segundo argumento apenas com as letras desejadas.
Exemplo para o caso que deseja-se executar apenas a etapa de preparação e filtragem:

    cd scripts
    python3 colorize_image.py nome_video.mp4 PF

Os videos e dados processados serão salvos na pasta Data/nome_video/ , onde:

    - nome_video.mp4 -> é o vídeo processado somente com a rede neural
    - nome_video_I.mp4 -> é o vídeo processado com os pontos perdidos interpolados
    - nome_video_ISK.mp4 -> é o vídeo processado com os pontos perdidos interpolados e filtrado com um filtro de Kalman
    - nome_video_ISK_CI.png -> é a imagem colorida do vídeo processado com os pontos perdidos interpolados e filtrado com um filtro de Kalman
