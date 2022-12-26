import * as tf from '@tensorflow/tfjs';
import { Scalar, Tensor } from '@tensorflow/tfjs';

import '@tensorflow/tfjs-node';

import { trainData, testData } from './data';

const trainTensors = {
  sizeMB: tf.tensor2d(trainData.sizeMB, [20, 1]),
  timeSec: tf.tensor2d(trainData.timeSec, [20, 1]),
};

const testTensors = {
  sizeMB: tf.tensor2d(testData.sizeMB, [20, 1]),
  timeSec: tf.tensor2d(testData.timeSec, [20, 1]),
};

const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [1], units: 1, activation: 'linear' }));
model.compile({ optimizer: 'sgd', loss: 'meanAbsoluteError' });

(async () => {
  await model.fit(trainTensors.sizeMB, trainTensors.timeSec, {
    epochs: 10,
    callbacks: {
      onEpochEnd: (epoch, log) => {
        console.log(`Epoch: ${epoch}: loss = ${log.loss}`);
      },
    },
  });

  (model.evaluate(trainTensors.sizeMB, trainTensors.timeSec) as Scalar).print();

  (
    model.predict(
      tf.tensor2d([
        [1, 100, 10000],
        [3, 1],
      ]),
    ) as Tensor
  ).print();
})();
