import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';

const FILE_NAME = 'iris.csv';
const mapping = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'];
const csvPath = path.join(__dirname, '..', 'datasets', FILE_NAME);
const csvString = fs.readFileSync(csvPath, 'utf-8');
const { features, target } = csvString
  .split('\r\n')
  .filter((_, index) => index !== 0)
  .sort(() => Math.random() - 0.5)
  .map((v) => v.trim())
  .reduce<{ features: number[][]; target: number[] }>(
    (pre, curr) => {
      const row: any[] = curr.split(',').map((col) => (!Number.isNaN(Number(col)) ? Number(col) : col));
      const arrX = row.slice(0, -1);
      const y = row.pop();
      const arrY = mapping.findIndex((value) => value === y);
      return {
        ...pre,
        features: [...pre.features, arrX],
        target: [...pre.target, arrY],
      };
    },
    { features: [], target: [] },
  );

const trainData = {
  features: features.slice(0, features.length / 2),
  target: target.slice(0, target.length / 2),
};
const testData = {
  features: features.slice(features.length / 2),
  target: target.slice(target.length / 2),
};

const trainXs = tf.tensor2d(trainData.features, [trainData.features.length, trainData.features[0].length]);
const trainYs = tf.oneHot(tf.tensor1d(trainData.target).toInt(), mapping.length);

const testXs = tf.tensor2d(testData.features, [testData.features.length, testData.features[0].length]);
const testYs = tf.oneHot(tf.tensor1d(testData.target).toInt(), mapping.length);

const model = tf.sequential();

model.add(
  tf.layers.dense({
    inputShape: [trainData.features[0].length],
    units: 3,
    activation: 'softmax',
  }),
);

model.compile({ optimizer: tf.train.adam(0.01), loss: tf.losses.softmaxCrossEntropy, metrics: 'accuracy' });

(async () => {
  await model.fit(trainXs, trainYs, {
    epochs: 100,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log?.loss} acc = ${log?.acc}`),
    },
  });

  console.log(model.evaluate(testXs, testYs).toString());
  const predictOut = (model.predict(tf.tensor2d([[4.9, 3.1, 1.5, 0.1]])) as tf.Tensor<tf.Rank>).argMax(-1).dataSync();
  console.log(mapping[predictOut[0]]);
})();
