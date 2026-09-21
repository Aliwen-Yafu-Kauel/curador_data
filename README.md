# Detovision Annotator

Herramienta GUI (PyQt6) para etiquetar *Ground Truth* de voladuras mineras. Opera sobre el **residuo cinemático** (diferencia absoluta entre frames, `cv2.absdiff`) para aislar el movimiento y eliminar el fondo estático.

##  Instalación

```bash
git clone https://github.com/Aliwen-Yafu-Kauel/curador_data.git
cd detovision-annotator
uv add PyQt6 opencv-python-headless numpy
uv run python annotador.py

```

##  Guía Rápida

### Carga y Navegación

* **Cargar (MP4 / NPZ):** Busca automáticamente la detonación en el video y extrae los siguientes *N* frames. Si cargas un `.npz`, restaura tu sesión anterior.
* **Zoom:** Rueda del ratón (Scroll).
* **Paneo:** Mantener `Clic Derecho` + Arrastrar.
* **Deshacer:** Tecla `Z` (memoria de 20 pasos).

### Herramientas de Etiquetado (Clic Izquierdo)

Las brochas usan umbrales sobre el valor físico del píxel, evitando etiquetar ruido:

* 🟢 **Humo (Clase 1):** Usa un umbral bajo (ej. `15`) para capturar la pluma difusa.
* 🔵 **Metralla (Clase 2):** Usa un umbral alto (ej. `60`) para capturar solo las rocas brillantes en alta velocidad.
*  **Borrador (Clase 0):** Limpia cualquier etiqueta incondicionalmente.

##  Formato de Exportación (`.npz`)

Al guardar, se genera un archivo comprimido con dos tensores NumPy `uint8` de forma `(N_frames, Alto, Ancho)`:

* **`inputs`**: Matriz de la diferencia absoluta en escala de grises.
* **`targets`**: Matriz de etiquetas categóricas:
* `0`: Fondo
* `1`: Humo
* `2`: Metralla