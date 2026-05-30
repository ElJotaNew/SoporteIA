"""
IA Asistente — Wikipedia REST API + búsqueda inteligente
pip install scikit-learn speechrecognition pyttsx3 joblib numpy requests pyaudio
"""

import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
import json
import os
import re
import numpy as np
import requests

try:
    import speech_recognition as sr
    VOZ_ENTRADA_OK = True
except ImportError:
    VOZ_ENTRADA_OK = False

try:
    import pyttsx3
    VOZ_SALIDA_OK = True
except ImportError:
    VOZ_SALIDA_OK = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    import joblib
    ML_OK = True
except ImportError:
    ML_OK = False


# =============================================
# VOZ
# =============================================

_cola_voz = queue.Queue()

def _worker_voz():
    motor = None
    while True:
        texto = _cola_voz.get()
        if not VOZ_SALIDA_OK:
            _cola_voz.task_done()
            continue
        try:
            if motor is None:
                motor = pyttsx3.init()
                motor.setProperty("rate", 150)
                for v in motor.getProperty("voices"):
                    if any(x in v.id.lower() for x in ["spanish","es_","es-"]):
                        motor.setProperty("voice", v.id)
                        break
            motor.say(texto[:350])
            motor.runAndWait()
        except Exception:
            motor = None
        finally:
            _cola_voz.task_done()

threading.Thread(target=_worker_voz, daemon=True).start()

def hablar(texto):
    if VOZ_SALIDA_OK:
        _cola_voz.put(texto)


# =============================================
# WIKIPEDIA REST API — sin librerías externas
# =============================================

HEADERS_WIKI = {
    "User-Agent": "IAAsistente/2.0 (Python; educativo)",
    "Accept": "application/json",
    "Accept-Language": "es",
}

def wiki_resumen(titulo):
    """Obtiene resumen de Wikipedia por título exacto."""
    titulo_url = titulo.replace(" ", "_")
    url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{titulo_url}"
    try:
        r = requests.get(url, headers=HEADERS_WIKI, timeout=8)
        if r.status_code == 200:
            data = r.json()
            texto = data.get("extract", "")
            if texto and len(texto) > 30:
                return recortar(texto)
    except Exception:
        pass
    return None

def wiki_buscar(termino):
    """Busca páginas en Wikipedia por término."""
    url = "https://es.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": termino,
        "format": "json",
        "utf8": 1,
        "srlimit": 5,
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS_WIKI, timeout=8)
        if r.status_code == 200:
            resultados = r.json().get("query", {}).get("search", [])
            return [x["title"] for x in resultados]
    except Exception:
        pass
    return []

def recortar(texto, chars=480):
    if not texto or len(texto) < 15:
        return None
    texto = texto.strip()
    # Quitar referencias [1], [2]
    texto = re.sub(r"\[\d+\]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    corte = texto.find(". ", 200)
    if 0 < corte < chars:
        return texto[:corte + 1]
    return texto[:chars].rsplit(" ", 1)[0] + "..." if len(texto) > chars else texto

def limpiar_consulta(texto):
    prefijos = [
        "que es ","qué es ","quien es ","quién es ",
        "como funciona ","cómo funciona ",
        "cuentame sobre ","cuéntame sobre ",
        "hablame de ","háblame de ",
        "explicame ","explícame ",
        "dime sobre ","acerca de ",
        "que sabes de ","qué sabes de ",
        "definicion de ","definición de ",
        "donde queda ","dónde queda ",
        "para que sirve ","para qué sirve ",
        "que significa ","qué significa ",
        "cuando nació ","cuándo nació ",
        "quien fue ","quién fue ",
        "cual es ","cuál es ",
        "busca ","buscar ",
        "dime ","información sobre ",
        "informacion sobre ",
    ]
    t = texto.lower().strip().rstrip("?¿.!,")
    for p in prefijos:
        if t.startswith(p):
            t = t[len(p):]
            break
    return t.strip()

def buscar_wikipedia(consulta):
    """
    Estrategia de búsqueda en 3 pasos:
    1. Intento directo con el término
    2. Intento con variantes (mayúsculas, etc.)
    3. Búsqueda libre y tomar el primer resultado
    """
    termino = limpiar_consulta(consulta)
    if not termino or len(termino) < 2:
        return None

    # Paso 1: intentos directos
    variantes = [
        termino.title(),
        termino.capitalize(),
        termino,
        termino.upper(),
    ]
    for v in variantes:
        r = wiki_resumen(v)
        if r:
            return r

    # Paso 2: búsqueda libre
    titulos = wiki_buscar(termino)
    for titulo in titulos:
        r = wiki_resumen(titulo)
        if r:
            return r

    return None


# =============================================
# INTENCIONES LOCALES
# =============================================

DATOS_PATH = "historial_ia.json"
MODELO_PATH = "modelo_ia.joblib"

DATOS_INICIALES = {
    "textos": [
        "hola","buenos dias","buenas tardes","buenas noches","hey","holi","ey",
        "adios","hasta luego","nos vemos","chao","bye","hasta pronto","chau",
        "como te llamas","quien eres","que eres","tu nombre","presentate",
        "como estas","todo bien","como te va","estas bien","bien y tu",
        "gracias","muchas gracias","te lo agradezco","gracias por todo",
        "ayuda","que puedes hacer","para que sirves","en que me ayudas",
        "chiste","cuentame un chiste","dime un chiste","hazme reir",
    ],
    "etiquetas": [
        "saludo","saludo","saludo","saludo","saludo","saludo","saludo",
        "despedida","despedida","despedida","despedida","despedida","despedida","despedida",
        "identidad","identidad","identidad","identidad","identidad",
        "estado","estado","estado","estado","estado",
        "agradecimiento","agradecimiento","agradecimiento","agradecimiento",
        "ayuda","ayuda","ayuda","ayuda",
        "chiste","chiste","chiste","chiste",
    ]
}

RESPUESTAS = {
    "saludo":        "¡Hola! Pregúntame lo que quieras, busco en Wikipedia.",
    "despedida":     "¡Hasta luego!",
    "identidad":     "Soy una IA hecha en Python que busca en Wikipedia para responder.",
    "estado":        "¡Todo bien! Lista para buscar lo que necesites.",
    "agradecimiento":"¡De nada!",
    "ayuda":         "Pregúntame sobre cualquier persona, lugar, concepto o evento.",
    "chiste":        None,
}

CHISTES = [
    "¿Por qué los programadores confunden Halloween con Navidad? Porque OCT 31 = DEC 25.",
    "Un SQL entra a un bar y pregunta: ¿Puedo hacer un JOIN con ustedes?",
    "¿Qué le dijo el cero al ocho? ¡Bonito cinturón!",
    "¿Cómo se llama el campeón de buceo japonés? Tokofondo.",
    "¿Por qué los esqueletos no pelean? Porque no tienen agallas.",
]
_ci = [0]
def get_chiste():
    c = CHISTES[_ci[0] % len(CHISTES)]
    _ci[0] += 1
    return c

def cargar_datos():
    if os.path.exists(DATOS_PATH):
        with open(DATOS_PATH,"r",encoding="utf-8") as f:
            return json.load(f)
    return dict(DATOS_INICIALES)

def guardar_datos(d):
    with open(DATOS_PATH,"w",encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def entrenar_modelo(datos):
    if not ML_OK:
        return None
    m = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))),
        ("clf",   SGDClassifier(loss="modified_huber", random_state=42, max_iter=1000)),
    ])
    m.fit(datos["textos"], datos["etiquetas"])
    joblib.dump(m, MODELO_PATH)
    return m

def cargar_o_entrenar():
    datos = cargar_datos()
    if ML_OK and os.path.exists(MODELO_PATH):
        try:
            return joblib.load(MODELO_PATH), datos
        except Exception:
            pass
    return entrenar_modelo(datos), datos

def es_intencion_local(modelo, texto):
    if modelo is None or not ML_OK:
        return None
    t = texto.strip()
    palabras = t.split()
    # Si hay nombre propio en el texto → buscar en Wikipedia
    if any(w[0].isupper() for w in palabras[1:]):
        return None
    if len(palabras) > 4:
        return None
    proba = modelo.predict_proba([t.lower()])[0]
    confianza = float(np.max(proba))
    clase = modelo.classes_[np.argmax(proba)]
    return clase if confianza >= 0.88 else None

def agregar_ejemplo(datos, texto, etiqueta):
    datos["textos"].append(texto.lower().strip())
    datos["etiquetas"].append(etiqueta)
    guardar_datos(datos)
    return entrenar_modelo(datos)


# =============================================
# MICRÓFONO
# =============================================

def escuchar_microfono():
    if not VOZ_ENTRADA_OK:
        return None, "SpeechRecognition no instalado"
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        return r.recognize_google(audio, language="es-ES"), None
    except sr.WaitTimeoutError:
        return None, "No se detectó voz"
    except sr.UnknownValueError:
        return None, "No se entendió"
    except sr.RequestError as e:
        return None, f"Error de red: {e}"
    except Exception as e:
        return None, str(e)


# =============================================
# INTERFAZ
# =============================================

class AppIA:
    def __init__(self, root):
        self.root = root
        self.root.title("IA Asistente")
        self.root.geometry("720x620")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(True, True)

        self.modelo, self.datos = cargar_o_entrenar()
        self.escuchando = False
        self.procesando = False
        self.voz_activa = tk.BooleanVar(value=VOZ_SALIDA_OK)
        self._ultimo_texto = ""

        self._ui()
        self._log("sistema", "Lista. Pregúntame lo que quieras.")

    def _ui(self):
        h = tk.Frame(self.root, bg="#181825", pady=10)
        h.pack(fill="x")
        tk.Label(h, text="IA Asistente + Wikipedia",
                 font=("Segoe UI", 15, "bold"),
                 bg="#181825", fg="#cdd6f4").pack(side="left", padx=16)
        tk.Checkbutton(h, text="Voz", variable=self.voz_activa,
                       bg="#181825", fg="#a6adc8", selectcolor="#313244",
                       activebackground="#181825",
                       font=("Segoe UI", 10)).pack(side="right", padx=16)

        self.lbl_estado = tk.Label(self.root, text="",
                                   font=("Segoe UI", 9, "italic"),
                                   bg="#1e1e2e", fg="#fab387")
        self.lbl_estado.pack(anchor="w", padx=16)

        self.chat = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, font=("Segoe UI", 11),
            bg="#181825", fg="#cdd6f4", insertbackground="#cdd6f4",
            relief="flat", padx=14, pady=10, state="disabled"
        )
        self.chat.pack(fill="both", expand=True, padx=10, pady=(0,6))
        self.chat.tag_config("usuario",  foreground="#89b4fa")
        self.chat.tag_config("ia",       foreground="#a6e3a1")
        self.chat.tag_config("sistema",  foreground="#f9e2af")
        self.chat.tag_config("aviso",    foreground="#f38ba8")

        inf = tk.Frame(self.root, bg="#1e1e2e")
        inf.pack(fill="x", padx=10, pady=(0,12))
        self.entrada = tk.Entry(inf, font=("Segoe UI", 12),
                                bg="#313244", fg="#cdd6f4",
                                insertbackground="#cdd6f4", relief="flat")
        self.entrada.pack(side="left", fill="x", expand=True, ipady=9, padx=(0,8))
        self.entrada.bind("<Return>", lambda e: self._enviar())
        self.btn_enviar = tk.Button(inf, text="Enviar", command=self._enviar,
                                    bg="#89b4fa", fg="#1e1e2e",
                                    font=("Segoe UI", 11, "bold"),
                                    relief="flat", padx=14)
        self.btn_enviar.pack(side="left")
        self.btn_mic = tk.Button(inf, text="Mic", command=self._mic,
                                 bg="#a6e3a1", fg="#1e1e2e",
                                 font=("Segoe UI", 11, "bold"),
                                 relief="flat", padx=12,
                                 state="normal" if VOZ_ENTRADA_OK else "disabled")
        self.btn_mic.pack(side="left", padx=(6,0))

    def _log(self, tipo, texto):
        self.chat.config(state="normal")
        pref = {
            "usuario": "Tú:    ",
            "ia":      "IA:    ",
            "sistema": "[ sistema ] ",
            "aviso":   "[ aviso ]   ",
        }
        self.chat.insert("end", pref.get(tipo,"") + texto + "\n\n", tipo)
        self.chat.config(state="disabled")
        self.chat.see("end")

    def _estado(self, txt):
        self.root.after(0, self.lbl_estado.config, {"text": txt})

    def _ui_ocupada(self, ocupada):
        s = "disabled" if ocupada else "normal"
        self.root.after(0, self.btn_enviar.config, {"state": s})
        self.root.after(0, self.entrada.config,    {"state": s})

    def _procesar(self, texto):
        self._ultimo_texto = texto
        self._log("usuario", texto)
        self._ui_ocupada(True)

        try:
            # 1. Intención local
            clase = es_intencion_local(self.modelo, texto)
            if clase:
                resp = RESPUESTAS.get(clase) or get_chiste()
                self._log("ia", resp)
                if self.voz_activa.get():
                    hablar(resp)
                return

            # 2. Wikipedia REST API
            termino = limpiar_consulta(texto)
            self._estado(f'Buscando "{termino}"...')
            resultado = buscar_wikipedia(texto)
            self._estado("")

            if resultado:
                self._log("ia", resultado)
                if self.voz_activa.get():
                    hablar(resultado)
            else:
                msg = f'No encontré información sobre "{termino}". ¿Puedes reformular la pregunta?'
                self._log("ia", msg)
                if self.voz_activa.get():
                    hablar(msg)

        finally:
            self.procesando = False
            self._ui_ocupada(False)
            self._estado("")

    def _enviar(self):
        texto = self.entrada.get().strip()
        if not texto or self.procesando:
            return
        self.entrada.delete(0, "end")
        self.procesando = True
        threading.Thread(target=self._procesar, args=(texto,), daemon=True).start()

    def _mic(self):
        if self.escuchando or self.procesando:
            return
        self.escuchando = True
        self.btn_mic.config(text="Escuchando...", state="disabled", bg="#f38ba8")
        self._estado("Escuchando micrófono...")
        def tarea():
            t, e = escuchar_microfono()
            self.root.after(0, self._fin_mic, t, e)
        threading.Thread(target=tarea, daemon=True).start()

    def _fin_mic(self, texto, error):
        self.escuchando = False
        self.btn_mic.config(text="Mic", state="normal", bg="#a6e3a1")
        self._estado("")
        if error:
            self._log("aviso", error)
        elif texto:
            self._procesar(texto)


if __name__ == "__main__":
    root = tk.Tk()
    AppIA(root)
    root.mainloop()