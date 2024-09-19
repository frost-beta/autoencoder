// A script that converts all svg files to png files in a directory.

import fs from 'node:fs';
import sharp from 'sharp';

if (process.argv.length != 4) {
  console.error('Usage: convert.ts sourceDir targetDir');
  process.exit(1);
}

main(process.argv[2], process.argv[3]);

async function main(dir: string, target: string) {
  const filenames = fs.readdirSync(dir);
  await Promise.all(filenames.map(async (file) => {
    if (!file.endsWith('.svg'))
      return;
    const image = await resizedSvgToSharp(`${dir}/${file}`, {width: 64});
    await image.flatten({background: '#FFF'})
               .greyscale()
               .png()
               .toFile(`${target}/${file.replace(/\.svg$/, '.png')}`);
  }));
}

async function resizedSvgToSharp(input: string | Buffer,
                                 {width, height}: {width?: number; height?:number }) {
  const instance = sharp(input);
  const metadata = await instance.metadata();
  if (metadata.format != 'svg')
    return instance;

  const initDensity = metadata.density ?? 72;
  let wDensity = 0;
  let hDensity = 0;
  if (width && metadata.width)
    wDensity = (initDensity * width) / metadata.width;
  if (height && metadata.height)
    hDensity = (initDensity * height) / metadata.height;

  // Both width & height are not present and/or
  // can't detect both metadata.width & metadata.height.
  if (!wDensity && !hDensity)
    return instance;

  return sharp(input, {density: Math.max(wDensity, hDensity)}).resize(width, height);
}
