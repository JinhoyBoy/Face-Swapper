import cv2
import numpy as np

def extract_ellipse(image, center, axes, angle=0, start_angle=0, end_angle=360):
    # Erstelle eine leere Maske, die die gleiche Größe wie das Bild hat
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Zeichne eine Ellipse auf der Maske (weiß auf schwarzem Hintergrund)
    cv2.ellipse(mask, center, axes, angle, start_angle, end_angle, (255), -1)

    # Erstelle einen Alpha-Kanal aus der Maske (für Transparenz)
    alpha_mask = cv2.merge([mask, mask, mask, mask])

    # Wenn das Bild nur 3 Kanäle hat, konvertiere es in ein 4-Kanal-Bild (BGRA)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Wende die Maske auf das Bild an, um den Ellipsenbereich zu extrahieren
    result = cv2.bitwise_and(image, alpha_mask)

    return result

# Lade das Bild
image_path = 'test_images/image.png'  # Ersetze mit deinem Bildpfad
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Lade OpenCVs vortrainiertes Haar-Cascade für Gesichtsdetektion
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detectiere Gesichter im Bild
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

if len(faces) < 2:
    print("Es müssen mindestens zwei Gesichter im Bild sein!")
else:
    # Extrahiere die Regionen der Gesichter
    x1, y1, w1, h1 = faces[0]
    x2, y2, w2, h2 = faces[1]

    # Extrahiere die Gesichter aus den Rechtecken
    face1 = image[y1:y1 + h1, x1:x1 + w1]
    face2 = image[y2:y2 + h2, x2:x2 + w2]

    # Resize die Gesichter auf die Zielgröße
    face1_resized = cv2.resize(face1, (w2, h2))
    face2_resized = cv2.resize(face2, (w1, h1))

    # Berechne die Mittelpunkte und Achsen nach dem Resizing
    center1_resized = (w2 // 2, h2 // 2)  # Mittelpunkte nach Resize
    axes1_resized = (w2 // 3, h2 // 2)    # Achsen nach Resize
    center2_resized = (w1 // 2, h1 // 2)  # Mittelpunkte nach Resize
    axes2_resized = (w1 // 3, h1 // 2)    # Achsen nach Resize

    # Extrahiere die Ellipsen aus den resized Gesichtern
    face1_ellipse = extract_ellipse(face1_resized, center1_resized, axes1_resized)
    face2_ellipse = extract_ellipse(face2_resized, center2_resized, axes2_resized)

    # Erstelle ein Ergebnisbild, das das Originalbild kopiert
    result = image.copy()

    # Extrahiere nur die ersten 3 Kanäle aus den resized Gesichtern (ignoriere den Alpha-Kanal)
    face1_resized_bgr = face1_ellipse[:, :, :3]  # Entferne den Alpha-Kanal
    face2_resized_bgr = face2_ellipse[:, :, :3]  # Entferne den Alpha-Kanal

    # Extrahiere den Alpha-Kanal (Transparenz) des Ellipsen-Bildes
    alpha1 = face1_ellipse[:, :, 3]  # Alpha-Kanal von Gesicht 1
    alpha2 = face2_ellipse[:, :, 3]  # Alpha-Kanal von Gesicht 2

    # Beispiel: Nur die sichtbaren (nicht-transparenten) Pixel hinzufügen
    for i in range(h2):
        for j in range(w2):
            if face1_ellipse[i, j, 3] != 0:  # Nur nicht-transparente Pixel
                result[y2 + i, x2 + j, :3] = face1_ellipse[i, j, :3]  # Kopiere BGR-Werte

    for i in range(h1):
        for j in range(w1):
            if face2_ellipse[i, j, 3] != 0:  # Nur nicht-transparente Pixel
                result[y1 + i, x1 + j, :3] = face2_ellipse[i, j, :3]  # Kopiere BGR-Werte

    # Ergebnisbild anzeigen
    cv2.imshow("Face Swap with Ellipses", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
