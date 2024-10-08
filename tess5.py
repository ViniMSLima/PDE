import cv2
import pytesseract  # Importando o Tesseract para OCR
import os
import time
from collections import Counter
import re

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

# Função para validar formato da placa
def validar_placa(texto):
    # Regex para 3 letras seguidas de 4 números (ABC1234)
    padrao1 = r'^[A-Z]{3}\d{4}$'
    # Regex para 3 letras seguidas de 1 número, 1 letra e 2 números (ABC1D23)
    padrao2 = r'^[A-Z]{3}\d[A-Z]\d{2}$'
    
    return re.match(padrao1, texto) or re.match(padrao2, texto)

# Lista para armazenar os resultados detectados
leituras_detectadas = []

# Variável para armazenar o tempo de início
tempo_inicial = None
duracao = 5  # 5 segundos de execução

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

        # Realize OCR (reconhecimento de caracteres) usando Tesseract, permitindo apenas letras e números
        custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        plate_text = pytesseract.image_to_string(plate_img, config=custom_config).strip()

        # Verifique se o texto detectado contém exatamente 7 caracteres e se segue o formato esperado
        if len(plate_text) == 7 and validar_placa(plate_text):
            leituras_detectadas.append(plate_text)  # Adiciona o texto da placa à lista

            # Desenhe um retângulo verde ao redor da placa detectada na imagem original
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Inicie o temporizador apenas quando detectar a primeira placa válida
            if tempo_inicial is None:
                tempo_inicial = time.time()

    # Exiba o frame com as placas detectadas
    cv2.imshow('Vídeo em Tempo Real', frame)

    # Verifique se o temporizador foi iniciado e se os 5 segundos já passaram
    if tempo_inicial is not None and (time.time() - tempo_inicial > duracao):
        break

    # Saia do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere os recursos
cap.release()
cv2.destroyAllWindows()

# Verifique qual foi a leitura mais frequente entre as detectadas
if leituras_detectadas:
    resultado_final = Counter(leituras_detectadas).most_common(1)[0][0]  # A leitura mais comum
    print(f'Resultado final mais frequente: {resultado_final}')
else:
    print('Nenhuma placa válida foi detectada.')
