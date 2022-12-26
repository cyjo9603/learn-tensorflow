import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';

const FILE_NAME = 'fish.csv';
const mapping = ['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch', 'Pike', 'Smelt'];
const csvPath = path.join(__dirname, '..', 'datasets', FILE_NAME);
const csvString = fs.readFileSync(csvPath, 'utf-8');

const { features, target } = csvString
  .split('\n')
  .filter((_, index) => index !== 0)
  .sort(() => Math.random() - 0.5)
  .map((v) => v.trim())
  .reduce<{ features: number[][]; target: number[] }>(
    (pre, curr) => {
      const [y, ...arrX]: any[] = curr.split(',').map((col) => (!Number.isNaN(Number(col)) ? Number(col) : col));
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
  features: features.slice(0, ~~(features.length / 2)),
  target: target.slice(0, ~~(target.length / 2)),
};
const testData = {
  features: features.slice(~~(features.length / 2)),
  target: target.slice(~~(target.length / 2)),
};

const trainXs = tf.tensor2d(trainData.features, [trainData.features.length, trainData.features[0].length]);
const trainYs = tf.oneHot(tf.tensor1d(trainData.target).toInt(), mapping.length);

const testXs = tf.tensor2d(testData.features, [testData.features.length, testData.features[0].length]);
const testYs = tf.oneHot(tf.tensor1d(testData.target).toInt(), mapping.length);

const model = tf.sequential();

model.add(
  tf.layers.dense({
    inputShape: [trainData.features[0].length],
    units: mapping.length,
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
  const predictOut = (model.predict(tf.tensor2d([[200, 30, 32.3, 34.8, 5.568, 3.3756]])) as tf.Tensor<tf.Rank>)
    .argMax(-1)
    .dataSync();
  console.log(mapping[predictOut[0]]);
})();
