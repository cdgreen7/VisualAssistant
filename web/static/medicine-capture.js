// Freeze the existing phone preview once. Derive detection and OCR images from
// the same pixels, so coordinate scaling never assumes a later frame is aligned.
(function(root) {
  "use strict";
  const DEFAULTS = {maxSide: 2560, maxBytes: 8*1024*1024, quality: 0.92};
  function dimensions(width, height, maxSide) {
    if (!(width > 0 && height > 0)) throw new Error('Camera frame is not ready.');
    const scale = Math.min(1, maxSide/Math.max(width,height));
    return {width: Math.max(1,Math.round(width*scale)), height: Math.max(1,Math.round(height*scale))};
  }
  function mapBox(box, sourceWidth, sourceHeight, width, height) {
    // Both images are unmirrored, full frames, with no CSS cover/crop transform.
    if (![sourceWidth,sourceHeight,width,height].every(v => Number.isFinite(v) && v > 0) ||
        Math.abs(sourceWidth/sourceHeight-width/height) > 0.01*width/height ||
        !Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite))
      throw new Error('Frame orientation or bounding box is invalid.');
    const mapped = [Math.floor(box[0]*width/sourceWidth), Math.floor(box[1]*height/sourceHeight),
      Math.ceil(box[2]*width/sourceWidth), Math.ceil(box[3]*height/sourceHeight)];
    mapped[0] = Math.max(0,Math.min(width,mapped[0]));
    mapped[2] = Math.max(0,Math.min(width,mapped[2]));
    mapped[1] = Math.max(0,Math.min(height,mapped[1]));
    mapped[3] = Math.max(0,Math.min(height,mapped[3]));
    if (mapped[2] <= mapped[0] || mapped[3] <= mapped[1]) throw new Error('Empty candidate crop.');
    return mapped;
  }
  function jpeg(canvas, quality) {
    return new Promise((resolve,reject) => canvas.toBlob(blob => blob ? resolve(blob) :
      reject(new Error('Could not encode candidate image.')), 'image/jpeg', quality));
  }
  class MedicineCapture {
    constructor(createCanvas, options) { this.createCanvas = createCanvas; this.options = {...DEFAULTS,...options}; }
    async freeze(video) {
      const original = {width: video.videoWidth, height: video.videoHeight};
      const size = dimensions(original.width, original.height, this.options.maxSide);
      const frame = this.createCanvas();
      Object.assign(frame, size);
      frame.getContext('2d').drawImage(video,0,0,size.width,size.height);
      const low = this.createCanvas();
      const lowSize = dimensions(size.width,size.height,640);
      Object.assign(low, lowSize);
      low.getContext('2d').drawImage(frame,0,0,low.width,low.height);
      return {frame, original, capturedAt: Date.now(), detectionWidth: low.width,
        detectionHeight: low.height, detectionBlob: await jpeg(low,0.7)};
    }
    async prepare(snapshot, candidate) {
      if (candidate.image_width !== snapshot.detectionWidth || candidate.image_height !== snapshot.detectionHeight)
        throw new Error('Detection does not match the frozen frame.');
      const frame = snapshot.frame;
      const box = mapBox(candidate.bounding_box,candidate.image_width,candidate.image_height,frame.width,frame.height);
      const crop = this.createCanvas();
      crop.width = box[2]-box[0]; crop.height = box[3]-box[1];
      crop.getContext('2d').drawImage(frame,...box.slice(0,2),crop.width,crop.height,0,0,crop.width,crop.height);
      const [image, roi] = await Promise.all([jpeg(frame,this.options.quality),jpeg(crop,this.options.quality)]);
      if (image.size > this.options.maxBytes || roi.size > this.options.maxBytes)
        throw new Error('Candidate image exceeds 8 MiB. Move back or use manual capture.');
      return {image, crop: roi, metadata: {...candidate, captured_at: snapshot.capturedAt,
        original_width: snapshot.original.width, original_height: snapshot.original.height,
        capture_width: frame.width, capture_height: frame.height, capture_box: box,
        crop_width: crop.width, crop_height: crop.height, source: 'same_frozen_frame',
        product_identity_verified: false}};
    }
  }
  const api = {MedicineCapture, dimensions, mapBox, DEFAULTS};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MedicineCapture = api;
})(typeof self !== 'undefined' ? self : this);
