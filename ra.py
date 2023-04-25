import cv2
import numpy as np

# Definir los parámetros para la detección de arucos
parametros = cv2.aruco.DetectorParameters()
# Cargar el diccionario de aruco
diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# Cargar la imagen en vivo desde la cámara del celular
cap = cv2.VideoCapture(0)

while True:
    # Leer el fotograma actual
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar aruco en la imagen
    esquinas, ids, _ = cv2.aruco.detectMarkers(gray, diccionario, parameters=parametros)

    # Si se detecta un aruco, dibujar un rectángulo verde alrededor
    if np.all(ids != None):
        frame = cv2.aruco.drawDetectedMarkers(frame, esquinas, ids, (0,255,0))

        #Extraer los puntos de las esquinas en coordenadas separadas
        c1 = (esquinas[0][0][0][0], esquinas[0][0][0][1])
        c2 = (esquinas[0][0][1][0], esquinas[0][0][1][1])
        c3 = (esquinas[0][0][2][0], esquinas[0][0][2][1])
        c4 = (esquinas[0][0][3][0], esquinas[0][0][3][1])

        copy = frame
        # Cargar la imagen
        img = cv2.imread("menu.jpg")
        # Obtener el tamaño de la imagen
        img_size = img.shape
        #Coordenadas del aruco en una matriz
        puntos_aruco = np.array([c1,c2,c3,c4])

        #Coordenadas de la imagen en una matriz
        puntos_imagen = np.array([
            [0,0],
            [img_size[1] - 1, 0],
            [img_size[1] - 1, img_size[0] -1],
            [0, img_size[0] - 1]
        ], dtype = float)

        #Superposicion de la imagen
        h, estado = cv2.findHomography(puntos_imagen, puntos_aruco)

        #Transformacion de perspectiva
        perspectiva = cv2.warpPerspective(img, h, (copy.shape[1], copy.shape[0]))
        cv2.fillConvexPoly(copy, puntos_aruco.astype(int), 0, 16)
        copy = copy + perspectiva

        # Mostrar la imagen en vivo
        cv2.imshow("Realidad aumentada", copy)

    else:
        cv2.imshow("Realidad aumentada", frame)

# Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()