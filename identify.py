import cv2 as cv
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torchvision
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.functional import to_pil_image
from imutils.video import VideoStream
from imutils.video import FPS
from PIL import Image
import numpy as np
from sort import *
import sys



def visualize_video(path):
    """
    Função para visualização do video
    Input:
        path [list or string]: Lista ou variavel contendo as 
            string de caminho para leitura do arquivo de video
    Output:
        void
    """

    #Definindo o modelo
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.95)
    
    #Definindo o préprocessamento
    preprocess = weights.transforms()


    model = model.eval()
    classes = ['truck', 'car']

    # caso a variavel path não seja uma lista
    # converta para uma lista
    if type(path) is not list:
        path = [path]
    # percorre todos os caminhos dos arquivos
    for archive in path:
        ids = []
        tracker = Sort()
        # lendo o arquivo
        video = cv.VideoCapture(archive)
        # Se nao abriu, retorne uma mensagem
        if (video.isOpened() == False):
            print("Erro de abertura do arquivo, verifique o path enviado ao programa")
            break
        
        # Se abriu, processe
        while (video.isOpened):
            # Obtendo o valor de FPS do video para exibição
            
            # Confirmando se a leitura esta certa e retornando a matriz
            ret, frame = video.read()

            # se o video tem um frame a mostrar
            if ret is True:
                # transformando BGR do opencv em RGB
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # Transformando em formato PIL
                frame = Image.fromarray(frame)

                # Transformando em formato tensor
                frame = pil_to_tensor(frame)

                # Aplicando Pré-processamento na imagem de frame
                frame_out = preprocess(frame).unsqueeze(0)
                
                # Obtendo resposta do modelo pré-treinado
                output = model(frame_out)
                boxes = output[0]['boxes']
                labels_ = output[0]['labels'].tolist()
                labels = []
                
                # Nomeando os labels de saida do modelo
                for label in labels_:
                    labels.append(weights.meta['categories'][label])



                # Fazendo o pós processamento para desenhar uma caixa e o label das caixas na imagem
                frame = torchvision.utils.draw_bounding_boxes(frame, boxes, labels, colors='red')
                
                # transformando RBG para BGR novamente
                frame = cv.cvtColor(np.array(to_pil_image(frame)), cv.COLOR_RGB2BGR)

                # Identificando os objetos de acordo com o tempo
                trackers = tracker.update(boxes.detach().numpy())
                

                # Percorrendo os objetos encontrados
                for i, box in enumerate(trackers):
                    # Se o id do objeto é novo, armazene
                    if box[-1] not in ids:
                        ids.append(box[-1])
                    # Desenhando sobre o objeto encontrado
                    box = box.astype(int)
                    cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
                    cv.putText(frame, f'track{box[-1]}', (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                # Mostrando a imagem
                cv.imshow(archive, frame)

                #remover depois
                print(len(ids))
                
                # Se ao meio da demonstração, o usuario quiser fechar, aperte 'q'
                if cv.waitKey(1) == ord('q'):
                    break
            # Se o ret nao for true, então o video acabou
            else:
                print("O video acabou")
                break
        # Liberando espaço de memoria
        cv.destroyAllWindows()
        video.release()
        # Demonstração dos objetos encontrados
        print("No video ", path, "passaram ", len(ids), "carros e caminhoes")




path = sys.argv[1:]

if len(path) == 0:
    print("Entrada incorreta, passe o caminho de ao menos um video")

else:
    visualize_video(path)

    # Para testes, um exemplo de como executar o script
    # python3 identify.py "Dataset/road_video001.mp4" "Dataset/road_video002.mp4" "Dataset/road_video003.mp4"

    print(path)
    print(type(path))
