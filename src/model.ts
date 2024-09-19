import {core as mx, nn} from '@frost-beta/mlx';

export class VAE extends nn.Module {
  encoder: nn.Sequential;
  fc1: nn.Linear;
  fc2: nn.Linear;
  fc3: nn.Linear;

  decoder: Decoder;

  constructor(latentDims: number, imageShape: number[], maxFilters: number) {
    super();
    this.encoder = new nn.Sequential(
      new nn.Conv2d(imageShape.at(-1), maxFilters / 4, 3, 2, 1),
      new nn.BatchNorm(maxFilters / 4),
      nn.leakyRelu,
      new nn.Conv2d(maxFilters / 4, maxFilters / 2, 3, 2, 1),
      new nn.BatchNorm(maxFilters / 2),
      nn.leakyRelu,
      new nn.Conv2d(maxFilters / 2, maxFilters, 3, 2, 1),
      new nn.BatchNorm(maxFilters),
      nn.leakyRelu,
      (x) => mx.flatten(x, 1));

    const outputShape = imageShape.slice(0, -1).map(n => n / 8);
    const flattenedDim = maxFilters * outputShape.reduce((product, v) => product * v);

    this.fc1 = new nn.Linear(flattenedDim, latentDims);
    this.fc2 = new nn.Linear(flattenedDim, latentDims);
    this.fc3 = new nn.Linear(latentDims, flattenedDim);

    this.decoder = new nn.Sequential(
      (x) => x.reshape(-1, outputShape[0], outputShape[1], maxFilters),
      new nn.ConvTranspose2d(maxFilters, maxFilters / 2, 4, 2, 1),
      new nn.BatchNorm(maxFilters / 2),
      nn.leakyRelu,
      new nn.ConvTranspose2d(maxFilters / 2, maxFilters / 4, 4, 2, 1),
      new nn.BatchNorm(maxFilters / 4),
      nn.leakyRelu,
      new nn.ConvTranspose2d(maxFilters / 4, imageShape.at(-1), 4, 2, 1),
      nn.sigmoid);
  }

  forward(x: mx.array) {
    let h = this.encoder.forward(x);
    let [ z, mu, logvar ] = this.bottleneck(h);
    z = this.fc3.forward(z);
    return [ this.decoder.forward(z), mu, logvar ];
  }

  decode(z: mx.array) {
    return this.decoder.forward(z);
  }

  private bottleneck(h: mx.array) {
    const mu = this.fc1.forward(h);
    const logvar = this.fc2.forward(h);
    const z = this.reparameterize(mu, logvar);
    return [ z, mu, logvar ];
  }

  private reparameterize(mu: mx.array, logvar: mx.array) {
    const std = mx.exp(mx.multiply(logvar, 0.5));
    const eps = mx.random.normal(std.shape);
    const z = mx.add(mu, mx.multiply(std, eps));
    return z;
  }
}
