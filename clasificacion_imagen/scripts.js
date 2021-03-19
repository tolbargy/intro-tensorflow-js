let net;

const imgEl = document.getElementById('img');
const descEl = document.getElementById('descripcion_imagen');
const webcamElement = document.getElementById('webcam');

async function app() {
    net = await mobilenet.load();
    displayImagePrediction();

    webcam = await tf.data.webcam(webcamElement);
    while (true) {
        const img = await webcam.capture();
        const result = await net.classify(img);
        document.getElementById('console').innerHTML = JSON.stringify(result);
    }
}


imgEl.onload = async function () {
    displayImagePrediction();
}

async function displayImagePrediction() {
    try {
        result = await net.classify(imgEl);
        descEl.innerHTML = JSON.stringify(result);
    } catch {

    }
}

async function cambiarImagen() {
    imgEl.crossOrigin = 'anonymous';
    imgEl.src = "https://picsum.photos/200/300?random=1"
}

app();