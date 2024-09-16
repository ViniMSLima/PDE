import cv2
import numpy as np
import os

# Carregue o classificador em cascata treinado para placas de veículos
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Tamanho fixo da imagem final de cada placa
fixed_size = (110, 30)

# Crie a pasta plates se não existir
output_folder = 'plates'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Abra a câmera
cap = cv2.VideoCapture(0)  # 0 para a câmera padrão, altere se você tiver múltiplas câmeras

if not cap.isOpened():
    print('Erro ao abrir a câmera.')
    exit()

# Contador de frames processados
frame_count = 0

while True:
    # Capture o frame da câmera
    ret, frame = cap.read()

    if not ret:
        print('Erro ao capturar o frame.')
        break

    # Converta o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecte placas de veículos
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Crie uma cópia do frame para desenhar retângulos e exibir a placa
    display_frame = frame.copy()

    # Processar e exibir as placas detectadas
    for (x, y, w, h) in plates:
        # Recorte a placa
        plate_img = frame[y + 14:y+h-10, x + 22:x+w - 30]

        # Converta a imagem da placa para escala de cinza
        gray_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Aplique o desfoque Gaussian
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # Aplique o limiar de Otsu para binarização
        _, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Redimensione a imagem para 110x30
        resized_img = cv2.resize(binary_img, fixed_size)

        # Converta a imagem redimensionada para uma imagem binária
        resized_img = np.uint8(resized_img)

        # Crie uma máscara para a área preta que toca as bordas
        mask = np.zeros_like(resized_img)
        mask[0, :] = 255
        mask[-1, :] = 255
        mask[:, 0] = 255
        mask[:, -1] = 255

        # Preencha as áreas pretas tocando as bordas usando flood fill
        flood_fill_mask = np.zeros((resized_img.shape[0] + 2, resized_img.shape[1] + 2), np.uint8)
        cv2.floodFill(resized_img, flood_fill_mask, (0, 0), 255)
        cv2.floodFill(resized_img, flood_fill_mask, (resized_img.shape[1] - 1, resized_img.shape[0] - 1), 255)

        # Combine a imagem binária com a máscara
        filled_img = cv2.bitwise_or(resized_img, mask)

        # Exiba a imagem da placa processada
        cv2.imshow('Placa Processada', filled_img)

        # Salve a imagem processada no quinto frame
        frame_count += 1
        if frame_count == 5:
            # Crie um nome de arquivo para a imagem processada
            output_path = os.path.join(output_folder, 'plate_processed.jpg')
            cv2.imwrite(output_path, filled_img)
            print(f'Imagem processada da placa salva em: {output_path}')
            frame_count = 0  # Resetar o contador após salvar a imagem

        # Desenhe um retângulo verde ao redor da placa detectada na imagem original
        cv2.rectangle(display_frame, (x + 22, y + 17), (x+w - 30, y+h - 8), (0, 255, 0), 2)

    # Exiba o frame com as placas detectadas
    cv2.imshow('Vídeo em Tempo Real', display_frame)

    # Saia do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere os recursos
cap.release()
cv2.destroyAllWindows()
