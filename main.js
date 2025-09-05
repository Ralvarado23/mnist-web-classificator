// Variables globales
let canvas;
let previewCanvas = document.getElementById("previewCanvas");
let previewCtx = previewCanvas.getContext("2d");
let models = {
    denso: null,
    cnn: null,
    cnn_aug: null
};
let modelsLoaded = {
    denso: false,
    cnn: false,
    cnn_aug: false
};

// Configuración del pincel
let brushWidth = 15;
let brushColor = "#000000";

// Rutas de los modelos
const MODEL_PATHS = {
    denso: './modelos_mnist/modelo_denso/model.json',
    cnn: './modelos_mnist/modelo_cnn/model.json',
    cnn_aug: './modelos_mnist/modelo_cnn_aug/model.json'
};

// Inicializar aplicación
function init() {
    // Configurar Fabric.js canvas
    canvas = new fabric.Canvas('canvas', {
    isDrawingMode: true,
    backgroundColor: 'white'
    });

    canvas.freeDrawingBrush.color = brushColor;
    canvas.freeDrawingBrush.width = brushWidth;
    fabric.Object.prototype.transparentCorners = false;

    // Cargar modelos
    loadModels();

    // Event listeners
    document.getElementById("clearBtn").addEventListener("click", clearCanvas);
    document.getElementById("predictBtn").addEventListener("click", predict);
}

// Cargar todos los modelos
async function loadModels() {
    const statusSection = document.getElementById("statusSection");
    const modelNames = ['denso', 'cnn', 'cnn_aug'];
    let loadedCount = 0;
    let totalModels = modelNames.length;

    statusSection.innerHTML = '<span class="loading-spinner"></span> Cargando modelos neuronales...';
    statusSection.className = 'status-section loading';

    // Cargar cada modelo
    for (let i = 0; i < modelNames.length; i++) {
    const modelName = modelNames[i];
    const statusElement = document.getElementById(`status${i + 1}`);

    try {
        statusElement.innerHTML = '<span class="loading-spinner"></span> Cargando...';
        statusElement.className = 'model-status status-loading';

        models[modelName] = await tf.loadLayersModel(MODEL_PATHS[modelName]);
        modelsLoaded[modelName] = true;
        loadedCount++;

        statusElement.innerHTML = '✅ Modelo cargado';
        statusElement.className = 'model-status status-ready';

        console.log(`Modelo ${modelName} cargado exitosamente`);

    } catch (error) {
        console.error(`Error cargando modelo ${modelName}:`, error);

        statusElement.innerHTML = '❌ Error al cargar';
        statusElement.className = 'model-status status-error';
    }
    }

    // Actualizar estado global
    if (loadedCount === totalModels) {
    statusSection.innerHTML = `✅ ${loadedCount}/${totalModels} modelos cargados correctamente`;
    statusSection.className = 'status-section ready';
    document.getElementById("predictBtn").disabled = false;
    } else if (loadedCount > 0) {
    statusSection.innerHTML = `⚠️ ${loadedCount}/${totalModels} modelos cargados. Algunos fallos detectados.`;
    statusSection.className = 'status-section ready';
    document.getElementById("predictBtn").disabled = false;
    } else {
    statusSection.innerHTML = '❌ No se pudieron cargar los modelos. Revisa las rutas de archivo.';
    statusSection.className = 'status-section error';
    }
}

// Limpiar canvas
function clearCanvas() {
    // Limpiar canvas principal
    canvas.clear();
    canvas.backgroundColor = 'white';

    // Limpiar preview
    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);

    // Limpiar todas las predicciones
    for (let i = 1; i <= 3; i++) {
    document.getElementById(`pred${i}`).textContent = '?';
    document.getElementById(`conf${i}`).textContent = '0%';
    document.getElementById(`confBar${i}`).style.width = '0%';
    document.getElementById(`pred${i}`).classList.remove('animate');
    }
}

// Realizar predicciones con todos los modelos
async function predict() {
    try {
    // Preparar imagen
    let tempCanvas = document.createElement("canvas");
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    let tempCtx = tempCanvas.getContext("2d");

    // Copiar la imagen del canvas de Fabric
    tempCtx.drawImage(canvas.lowerCanvasEl, 0, 0, 28, 28);

    // Mostrar la preview
    previewCtx.imageSmoothingEnabled = false;
    previewCtx.drawImage(tempCanvas, 0, 0, previewCanvas.width, previewCanvas.height);

    // Extraer datos en escala de grises
    let imgData = tempCtx.getImageData(0, 0, 28, 28);
    let gray = [];

    for (let i = 0; i < imgData.data.length; i += 4) {
        let r = imgData.data[i];
        let g = imgData.data[i + 1];
        let b = imgData.data[i + 2];
        let v = (r + g + b) / 3;
        v = 255 - v;  // invertir colores para MNIST
        gray.push(v / 255.0);  // normalizar a 0-1
    }

    // Crear tensor de entrada
    let input = tf.tensor(gray, [1, 28, 28, 1]);

    // Predicciones con cada modelo
    const modelNames = ['denso', 'cnn', 'cnn_aug'];

    for (let i = 0; i < modelNames.length; i++) {
        const modelName = modelNames[i];
        const model = models[modelName];

        if (model && modelsLoaded[modelName]) {
        try {
            let prediction = model.predict(input);
            let probs = await prediction.data();
            let predClass = probs.indexOf(Math.max(...probs));
            let confidence = Math.max(...probs);

            // Mostrar resultado con animación
            animatePrediction(i + 1, predClass, confidence);

            // Limpiar tensor de predicción
            prediction.dispose();

        } catch (error) {
            console.error(`Error en predicción del modelo ${modelName}:`, error);
        }
        }
    }

    // Limpiar tensor de entrada
    input.dispose();

    } catch (error) {
    console.error("Error en predicción:", error);
    alert("Error realizando predicción");
    }
}

// Animar resultado de predicción
function animatePrediction(modelIndex, prediction, confidence) {
    const predElement = document.getElementById(`pred${modelIndex}`);
    const confElement = document.getElementById(`conf${modelIndex}`);
    const confBarElement = document.getElementById(`confBar${modelIndex}`);

    // Animar número con efecto de escala
    predElement.classList.add('animate');
    predElement.textContent = prediction;

    // Animar barra de confianza y porcentaje
    setTimeout(() => {
    confBarElement.style.width = `${confidence * 100}%`;
    confElement.textContent = `${(confidence * 100).toFixed(1)}%`;
    }, 200);

    // Quitar animación después de un tiempo
    setTimeout(() => {
    predElement.classList.remove('animate');
    }, 600);
}

// Inicializar cuando la página cargue
window.addEventListener('load', init);
