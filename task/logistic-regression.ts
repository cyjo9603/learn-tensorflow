import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';

const FILE_NAME = 'student-mat-pass-or-fail.csv';
const csvPath = path.join(__dirname, '..', 'datasets', FILE_NAME);
const csvString = fs.readFileSync(csvPath, 'utf-8');

const { head, features, target } = csvString
  .split('\n')
  .reduce<{ head: string[]; features: number[][]; target: number[] }>(
    (pre, curr, i) => {
      if (i === 0) {
        return { ...pre, head: curr.split(',') };
      }

      const row = curr.split(',').map((col) => Number(col));
      const arrX = row.slice(0, -1);
      const arrY = row.pop();

      return {
        ...pre,
        features: [...pre.features, arrX],
        target: [...pre.target, arrY],
      };
    },
    { head: [], features: [], target: [] },
  );

const trainData = {
  features: features.slice(0, features.length / 2),
  target: target.slice(0, target.length / 2),
};

const testData = {
  features: features.slice(0, features.length / 2),
  target: target.slice(0, target.length / 2),
};

const trainXs = tf.tensor2d(trainData.features, [trainData.features.length, trainData.features[0].length]);
const trainYs = tf.tensor2d(trainData.target, [trainData.target.length, 1]);

const testXs = tf.tensor2d(testData.features, [testData.features.length, testData.features[0].length]);
const testYs = tf.tensor2d(testData.target, [testData.target.length, 1]);

const model = tf.sequential();

model.add(
  tf.layers.dense({
    inputShape: [head.length - 1],
    units: 1,
    activation: 'sigmoid',
  }),
);

model.compile({ optimizer: 'sgd', loss: tf.losses.sigmoidCrossEntropy, metrics: 'acc' });

(async () => {
  await model.fit(trainXs, trainYs, {
    batchSize: 40,
    epochs: 100,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log?.loss}`),
      onTrainEnd: () => {
        model.layers[0]
          .getWeights()[0]
          .data()
          .then((kernelAsArr) => {
            const weightArr = [];
            kernelAsArr.forEach((value, index) => weightArr.push({ key: head[index], value }));

            const sortedWeightArr = weightArr.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
            console.log({ sortedWeightArr });
          });
      },
    },
  });

  console.log(model.evaluate(testXs, testYs).toString());
  (
    model.predict(
      tf.tensor2d([[0, 0, 18, 0, 1, 1, 3, 2, 3, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 4, 4, 1, 3, 4, 5, 0, 11, 12, 10]]),
    ) as tf.Tensor<tf.Rank>
  ).print();
})();
