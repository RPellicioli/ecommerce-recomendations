import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = null;

const WEIGHTS = {
    age: 0.1,
    price: 0.2,
    color: 0.3,
    category: 0.4
}

const normalize = (value, min, max) => (value - min) / (max - min) || 1;
const oneHotWeighted = (index, length, weight) => tf.oneHot(index, length).cast('float32').mul(weight);

function makeContext(products, users) {
    const ages = users.map(u => u.age);
    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const prices = products.map(p => p.price);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colors = [...new Set(products.map(p => p.color))];
    const categories = [...new Set(products.map(p => p.category))];

    const colorsIndex = Object.fromEntries(colors.map((c, i) => [c, i]));
    const categoryIndex = Object.fromEntries(categories.map((c, i) => [c, i]));

    // Computar a media de idade dos usuários para usar como baseline
    const midAge = (minAge + maxAge) / 2;
    const ageSums = {}
    const ageCounts = {}

    users.forEach(u => {
        u.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + u.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        })
    })

    const productAvgAgeNorm = Object.fromEntries(products.map(p => {
        const avg = ageCounts[p.name] ? ageSums[p.name] / ageCounts[p.name] : midAge;
        return [p.name, normalize(avg, minAge, maxAge)];
    }));

    return {
        products,
        users,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        colors,
        colorsIndex,
        numColors: colors.length,
        categories,
        numCategories: categories.length,
        categoryIndex,
        dimesions: 2 + colors.length + categories.length, // idade + preço + one-hot de cor + one-hot de categoria  
        productAvgAgeNorm
    }
}

function encodeProduct(product, ctx) {
    // Normalizando dados para ficar de 0 a 1 e aplicando pesos
    const price = tf.tensor1d([normalize(product.price, ctx.minPrice, ctx.maxPrice) * WEIGHTS.price])

    const age = tf.tensor1d([(ctx.productAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age])

    const category = oneHotWeighted(ctx.categoryIndex[product.category], ctx.numCategories, WEIGHTS.category);

    const color = oneHotWeighted(ctx.colorsIndex[product.color], ctx.numColors, WEIGHTS.color);

    return tf.concat1d([age, price, category, color]);
}

function encodeUser(user, context) {
    if (user.purchases.length) {
        return tf.stack(user.purchases.map(p => encodeProduct(p, context))).min(0).reshape([1, context.dimesions]);
    }

    return tf.concat1d(
        [
            tf.tensor1d([normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age]),
            tf.tensor1d([0]), // preço médio dos produtos comprados, 0 se não tiver compras
            tf.zeros([context.numColors]).cast('float32'), // one-hot de cores, tudo 0 se não tiver compras
            tf.zeros([context.numCategories]).cast('float32') // one-hot de categorias, tudo 0 se não tiver compras
        ]
    ).reshape([1, context.dimesions]);
}

function createTrainingModel(context) {
    const inputs = [];
    const labels = [];

    context.users.filter(u => u.purchases.length).forEach(u => {
        const userVector = encodeUser(u, context).dataSync();
        context.products.forEach(p => {
            const productVector = encodeProduct(p, context).dataSync()
            const label = u.purchases.some(pur => pur.name === p.name ? 1 : 0);

            inputs.push([...userVector, ...productVector]);
            labels.push(label);
        })
    })

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimentions: context.dimesions * 2
        // tamanho do vetor de usuário + tamanho do vetor de produto
    }
}

async function configureNeuralNetworkAndTrain(traningData) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [traningData.inputDimentions], units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    await model.fit(traningData.xs, traningData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    });

    return model;
}

async function trainModel({ users }) {
    console.log('Training model with users:', users);

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    const products = await fetch('/data/products.json').then(res => res.json());
    const context = makeContext(products, users);

    context.productVectors = products.map(p => {
        return {
            name: p.name,
            meta: { ...p },
            vector: encodeProduct(p, context).dataSync()
        }
    });

    _globalCtx = context;

    const trainingData = createTrainingModel(context);
    _model = await configureNeuralNetworkAndTrain(trainingData);

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

function recommend({ user }) {
    if (!_model) return;

    const userVector = encodeUser(user, _globalCtx).dataSync();
    const inputs = _globalCtx.productVectors.map(p => [...userVector, ...p.vector]);

    // inputs é uma matriz onde cada linha é a concatenação do vetor do usuário com o vetor de um produto
    // formato: [[userVector, product1Vector], [userVector, product2Vector], ...]
    const inputTensor = tf.tensor2d(inputs);

    const predictions = _model.predict(inputTensor);
    const scores = predictions.dataSync();

    const recommendations = _globalCtx.productVectors
        .map((p, i) => ({ ...p.meta, name: p.name, score: scores[i] }))
        .sort((a, b) => b.score - a.score)

    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations
    });
}

const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: recommend,
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
