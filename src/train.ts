import fs from 'node:fs';
import sharp from 'sharp';
import prettyMilliseconds from 'pretty-ms';
import {core as mx, optimizers as optim, nn} from '@frost-beta/mlx';

import {VAE} from './model';

const imageShape = [ 64, 64, 1 ];

const latentDims = 8;
const maxFilters = 64;

const epochs = 1;
const batchSize = 128;
const learningRate = 1e-3;

const reportPerIter = 1;

train(process.argv[2] ?? 'assets');

async function train(assets: string) {
  // Get the images to train with.
  const files = fs.readdirSync(assets).filter(f => f.endsWith('.png'))
                                      .map(f => `${assets}/${f}`);
  if (files.length < batchSize)
    throw new Error('Too few images.');

  // Reorder files randomly.
  shuffle(files);

  const model = new VAE(latentDims, imageShape, maxFilters);

  const lossAndGradFunction = nn.valueAndGrad(model, lossFunction);
  const optimizer = new optim.AdamW(learningRate);

  let trainedFiles = 0;
  let losses: number[] = [];
  for (let e = 0, iterations = 0, start = Date.now(); e < epochs; ++e) {
    for await (const batch of iterateBatches(files)) {
      // Use mx.tidy to free all the intermediate tensors immediately.
      mx.tidy(() => {
        const x = mx.array(batch).reshape([ batchSize, ...imageShape ]);
        // Compute loss and gradients, then update the model.
        const [loss, grads] = lossAndGradFunction(model, x);
        optimizer.update(model, grads);
        mx.eval(model.state, optimizer.state);
        losses.push(loss.item() as number);
        // Keep the states of model and optimizer from getting freed.
        return [model.state, optimizer.state];
      })
      // Report updates.
      if (++iterations % reportPerIter === 0) {
        const stop = Date.now();
        const trainLoss = mean(losses);
        const eta = ((files.length - trainedFiles - batchSize) / batchSize) * (stop - start);
        console.log(`Iter ${iterations}`,
                    `(${(100 * (trainedFiles + batchSize) / files.length).toFixed(1)}%):`,
                    `Train loss ${trainLoss.toFixed(2)},`,
                    `It/sec ${(reportPerIter / (stop - start) * 1000).toFixed(2)},`,
                    `ETA ${prettyMilliseconds(eta, {compact: true})}.`);
        start = Date.now();
        losses = [];
        trainedFiles += batchSize;
      }
    }
  }
}

function lossFunction(model: VAE, x: mx.array) {
  const [ z, mean, logvar ] = model.forward(x)
  // Reconstruction loss.
  const loss = nn.losses.mseLoss(z, x, 'sum');
  // KL divergence between encoder distribution and standard normal.
  const klDiv = mx.multiply(-0.5,
                            mx.sum(mx.subtract(mx.subtract(mx.add(1, logvar),
                                                           mean.square()),
                                               logvar.exp())))
  return mx.add(loss, klDiv);
}

async function* iterateBatches(files: string[]) {
  // Read files into batches of bitmaps.
  for (let i = 0; i < files.length - batchSize; i += batchSize) {
    const data = await Promise.all(files.slice(i, i + batchSize).map(file => {
      return sharp(file).extractChannel('red')  // greyscale image
                        .raw()  // to bitmap
                        .toBuffer();
    }));
    yield data.map(b => Array.from(b));
  }
}

function shuffle(array: unknown[]) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

function mean(array: number[]) {
  if (array.length == 0)
    return 0;
  return array.reduce((a, b) => a + b) / array.length;
}