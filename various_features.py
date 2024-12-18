import os

import numpy as np
from skimage.feature import canny, corner_peaks, corner_harris, graycomatrix, graycoprops
from skimage.filters import sobel, gabor
from skimage.measure import shannon_entropy
from skimage.transform import hough_circle, hough_circle_peaks, resize
from skimage.color import rgb2gray


def detect_hough_circles(image, hough_radii: np.ndarray, circle_number: int):
    # detect edges
    image = rgb2gray(image)
    edges = canny(image, sigma=2, low_threshold=10, high_threshold=100)

    # Detect two radii
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent circles
    accums, cx, cy, radii = hough_circle_peaks(
        hough_res, hough_radii, total_num_peaks=circle_number
    )

    # Feature vector: Mean and std deviation of radii, and number of detected circles
    if radii.size > 0:
        print(radii.mean(), radii.std())
        return [radii.mean(), radii.std()] #, len(radii)]
    else:
        return [0, 0, 0]  # No circles detected


def detect_fur(image):
    # detect edges
    image = rgb2gray(image)
    edges = canny(image, sigma=0.75, low_threshold=10, high_threshold=100)

    # Feature vector: Anzahl der Kantenpixel und der Anteil an allen Pixeln
    edge_pixels = edges.sum()  # Anzahl der Kantenpixel
    edge_density = edge_pixels / edges.size  # Anteil der Kantenpixel

    return [edge_pixels, edge_density]


def detect_corners(image):
    # Corner Detection
    image = rgb2gray(image)
    coords = corner_peaks(corner_harris(image), min_distance=5, threshold_rel=0.10)

    return [len(coords)]


def average_color(image):
    return [image.mean()]

def extract_features_histogram(image, bins=32):
    image = rgb2gray(image)
    hist, _ = np.histogram(image, bins=bins, range=(0, 1))  # Histogramm (normiert)
    return hist / hist.sum()  # Normalisierung

def extract_image_entropy(image):
    image = rgb2gray(image)  # Falls Farbbild, konvertiere in Graustufen.
    entropy_value = shannon_entropy(image)  # Shannon-Entropie berechnen.
    return [entropy_value]  # Einzelnes Entropie-Feature zurückgeben.

def extract_glcm_features(image):
    gray_image = rgb2gray(image)
    # Graustufen-Bild in diskrete Werte umwandeln (z. B. 16 Stufen)
    gray_image = (gray_image * 16).astype("uint8")
    glcm = graycomatrix(
        gray_image,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=17,
        symmetric=True,
        normed=True,
    )
    # Aus der GLCM wichtige Eigenschaften extrahieren
    contrast = graycoprops(glcm, "contrast").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    energy = graycoprops(glcm, "energy").mean()
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))  # Entropie manuell berechnet
    return [contrast, homogeneity, energy, entropy]

def extract_sobel_features(image):
    gray_image = rgb2gray(image)
    edges = sobel(gray_image)  # Sobel-Kantenfilter
    edge_mean = edges.mean()
    edge_std = edges.std()
    return [edge_mean, edge_std]


def compute_glrl_matrix(image, levels=256, direction=(0, 1)):
    gray_image = (rgb2gray(image) * (levels - 1)).astype(
        int
    )  # Normiere auf [0, levels - 1]
    max_run_length = max(image.shape)  # Maximale mögliche Run-Länge (größte Dimension)
    glrl = np.zeros(
        (levels, max_run_length + 1), dtype=int
    )  # Breite + 1 für Sicherheit

    # Richtung (dx, dy) extrahieren
    dx, dy = direction

    for i in range(gray_image.shape[0]):
        run_length = 0
        prev_value = -1
        for j in range(gray_image.shape[1]):
            # Sicherstellen, dass wir innerhalb der Bildgrenzen bleiben
            if dx == 0:  # Horizontal
                value = gray_image[i, j]
            elif dy == 0:  # Vertikal
                value = gray_image[j, i]
            else:
                continue  # Unterstütze komplexere Richtungen später

            # Gleiche Werte --> Verlängere Lauf
            if value == prev_value:
                run_length += 1
            else:
                # Neuer Wert, füge bisherigen Run zu Matrix hinzu
                if prev_value != -1 and run_length > 0:
                    glrl[prev_value, run_length] += 1
                prev_value = value
                run_length = 1

        # Vergiss nicht, die letzte Run-Länge zu aktualisieren
        if prev_value != -1 and run_length > 0:
            glrl[prev_value, run_length] += 1

    return glrl


def compute_glrl_features(image):
    glrl = compute_glrl_matrix(image)
    short_run_emphasis = np.sum(glrl / (1 + np.arange(glrl.shape[1]))[None, :])
    long_run_emphasis = np.sum(glrl * np.arange(glrl.shape[1])[None, :])
    return [short_run_emphasis, long_run_emphasis]


def extract_gabor_features(
    image,
    frequencies=[0.1, 0.3, 0.5],
    directions=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
):
    gray_image = rgb2gray(image)  # Konvertiere in Graustufen
    gabor_features = []

    # Iteriere über Frequenzen und Richtungen
    for freq in frequencies:
        for theta in directions:
            # Filter anwenden
            filt_real, filt_imag = gabor(gray_image, frequency=freq, theta=theta)
            # Extrahiere Mittelwert und Standardabweichung als Features
            gabor_features.append(np.mean(filt_real))
            gabor_features.append(np.std(filt_real))

    return gabor_features


def haar_wavelet_transform(image):
    # Zeilen-Transformation
    rows = image.shape[0]
    cols = image.shape[1]

    # Approximations-Koeffizienten (Mittelwerte)
    approx_rows = (image[:, 0:cols:2] + image[:, 1:cols:2]) / 2

    # Detail-Koeffizienten (Differenzen)
    detail_rows = (image[:, 0:cols:2] - image[:, 1:cols:2]) / 2

    # Spalten-Transformation auf Ergebnis anwenden
    LL = (
        approx_rows[0:rows:2, :] + approx_rows[1:rows:2, :]
    ) / 2  # Approximationsbereich
    LH = (approx_rows[0:rows:2, :] - approx_rows[1:rows:2, :]) / 2  # Horizontal-Details
    HL = (detail_rows[0:rows:2, :] + detail_rows[1:rows:2, :]) / 2  # Vertikal-Details
    HH = (detail_rows[0:rows:2, :] - detail_rows[1:rows:2, :]) / 2  # Diagonal-Details

    return [(LL, LH, HL, HH)]


def extract_wavelet_features_manual(image):
    transformed = haar_wavelet_transform(image)
    wavelet_features = []

    # Iteriere über Stufen und extrahiere Merkmale
    for LL, LH, HL, HH in transformed:
        wavelet_features.extend(
            [np.mean(LL), np.std(LL)]
        )  # Approximationskoeffizienten
        wavelet_features.extend([np.mean(LH), np.std(LH)])  # Horizontal
        wavelet_features.extend([np.mean(HL), np.std(HL)])  # Vertikal
        wavelet_features.extend([np.mean(HH), np.std(HH)])  # Diagonal

    return wavelet_features


def extract_rgb_histogram(image, bins=32):
    # Histogramme für jeden Farbkanal berechnen
    r_hist, _ = np.histogram(image[:, :, 0], bins=bins, range=(0, 256))
    g_hist, _ = np.histogram(image[:, :, 1], bins=bins, range=(0, 256))
    b_hist, _ = np.histogram(image[:, :, 2], bins=bins, range=(0, 256))

    # Histogramme normieren (optional, um die Bildunabhängigkeit zu gewährleisten)
    r_hist = r_hist / np.sum(r_hist)
    g_hist = g_hist / np.sum(g_hist)
    b_hist = b_hist / np.sum(b_hist)

    # Kombiniere die Histogramme in einen Feature-Vektor
    histogram_features = np.concatenate([r_hist, g_hist, b_hist])

    return histogram_features


def extract_features(image):
    fv = []
    #fv.extend(detect_hough_circles(image, np.arrange(10, 80, 2), 10))
    #fv.extend(detect_fur(image))
    #fv.extend(detect_corners(image))
    #fv.extend(average_color(image))
    fv.extend(extract_features_histogram(image))
    fv.extend(extract_image_entropy(image))
    #fv.extend(extract_glcm_features(image))
    #fv.extend(extract_sobel_features(image))
    #fv.extend(compute_glrl_features(image))
    #fv.extend(extract_gabor_features(image)) # Dauert ewig
    #fv.extend(extract_wavelet_features_manual(image))
    #fv.extend(extract_rgb_histogram(image))
    return fv