async function getData() {
    const datosCasasR = await fetch("datos.json");
    const datosCasas = await datosCasasR.json();
    var datosLimpios = datosCasas.map(casa => ({
        precio: casa.Precio,
        cuartos: casa.NumeroDeCuartosPromedio
    }))
    datosLimpios = datosLimpios.filter(casa => (
        casa.precio != null && casa.cuartos != null
    ));

    return datosLimpios;
}


function visualizarDatos(data) {
    const valores = data.map(d => ({
        x: d.cuartos,
        y: d.precio
    }));

    tfvis.render.scatterplot(
        {
            name: 'Cuartos vs Precio'
        },
        {
            values: valores
        },
        {
            xLabel: 'Cuartos',
            yLabel: 'Precio',
            height: 300
        }
    );
}


function crearModelo() {
    const modelo = tf.sequential();

    modelo.add(
        tf.layers.dense({
            inputShape: [1],
            units: 1,
            useBias: true
        })
    );

    modelo.add(
        tf.layers.dense({
            units: 1,
            useBias: true
        })
    );

    return modelo;
}

function convertDatosATensores(data) {
    return tf.tidy(() => {
        tf.util.shuffle(data);

        const entradas = data.map(d => d.cuartos);
        const etiquetas = data.map(d => d.precio);
        const tensorEntradas = tf.tensor2d(entradas, [entradas.length, 1]);
        const tensorEtiquetas = tf.tensor2d(etiquetas, [etiquetas.length, 1]);

        const entradasMax = tensorEntradas.max();
        const entradasMin = tensorEntradas.min();

        const etiquetasMax = tensorEtiquetas.max();
        const etiquetasMin = tensorEtiquetas.min();

        const entradasNormalizadas = tensorEntradas.sub(entradasMin)
                                    .div(entradasMax.sub(entradasMin));
        const etiquetasNormalizadas = tensorEtiquetas.sub(etiquetasMin)
                                    .div(etiquetasMax.sub(etiquetasMin));

        return {
            entradas: entradasNormalizadas,
            etiquetas: etiquetasNormalizadas,
            entradasMax,
            entradasMin,
            etiquetasMax,
            etiquetasMin
        }
    });
}

async function run() {
    // body
    const data = await getData();
    visualizarDatos(data);

    console.log(crearModelo());
    const tensorData = convertDatosATensores(data);
    const {entradas,etiquetas} = tensorData;
}

run();