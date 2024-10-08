import cv2
import pytesseract  # Importando o Tesseract para OCR
import os

# Adicionar o caminho para o executável do Tesseract se necessário
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Carregue o classificador em cascata treinado para placas de veículos
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Crie a pasta plates se não existir
output_folder = 'plates'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Abra a câmera
cap = cv2.VideoCapture(0)  # 0 para a câmera padrão, altere se você tiver múltiplas câmeras

if not cap.isOpened():
    print('Erro ao abrir a câmera.')
    exit()

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

    # Processar e exibir as placas detectadas
    for (x, y, w, h) in plates:
        # Recorte a placa diretamente da imagem colorida original
        plate_img = frame[y:y+h, x:x+w]

        # Exiba a imagem da placa original
        cv2.imshow('Placa Detectada', plate_img)

        # Realize OCR (reconhecimento de caracteres) usando Tesseract
        plate_text = pytesseract.image_to_string(plate_img, config='--psm 7')  # Configuração para melhor leitura de placas
        print(f'Texto da placa detectada: {plate_text.strip()}')

        # Desenhe um retângulo verde ao redor da placa detectada na imagem original
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Exiba o frame com as placas detectadas
    cv2.imshow('Vídeo em Tempo Real', frame)

    # Saia do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere os recursos
cap.release()
cv2.destroyAllWindows()
