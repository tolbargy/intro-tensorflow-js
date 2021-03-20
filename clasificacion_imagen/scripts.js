let net;

const imgEl = document.getElementById('img');
const descEl = document.getElementById('descripcion_imagen');
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();

async function app() {
    net = await mobilenet.load();
    displayImagePrediction();

    webcam = await tf.data.webcam(webcamElement);
    while (true) {
        const img = await webcam.capture();
        const result = await net.classify(img);
        const activation = net.infer(img, 'conv_preds');
        var resultTransferencia;
        try {
            resultTransferencia = await classifier.predictClass(activation);
            const classes = ["Untrained", "Tarro", "Jorge", "OK", "Tijeras"];
            document.getElementById('console2').innerHTML = classes[resultTransferencia.label];
        } catch (error) {
            console.log("modelo no configurado aun");
        }

        document.getElementById('console').innerHTML = JSON.stringify(result);
        img.dispose();
        await tf.nextFrame();
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

async function addExample(classId) {
    console.log('added example');
    const img = await webcam.capture();
    const activation = net.infer(img, true);
    classifier.addExample(activation, classId);

    img.dispose();
}

app();