/**
 * Much of the code below is adapted from the official BirdNET-Analyzer GitHub repository.
 */

importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

// Define custom layer for computing mel spectrograms
class MelSpecLayerSimple extends tf.layers.Layer {
    constructor(config) {
        super(config);

        // Initialize parameters
        this.sampleRate = config.sampleRate;
        this.specShape = config.specShape;
        this.frameStep = config.frameStep;
        this.frameLength = config.frameLength;
        this.fmin = config.fmin;
        this.fmax = config.fmax;
        this.melFilterbank = tf.tensor2d(config.melFilterbank);
    }

    build(inputShape) {
        // Initialize trainable weights, for example:
        this.magScale = this.addWeight(
            'magnitude_scaling',
            [],
            'float32',
            tf.initializers.constant({ value: 1.23 })
        );

        super.build(inputShape);
    }

    // Compute the output shape of the layer
    computeOutputShape(inputShape) {
        return [inputShape[0], this.specShape[0], this.specShape[1], 1];
    }

    // Define the layer's forward pass
    call(inputs) {
        return tf.tidy(() => {
            // inputs is a tensor representing the input data
            inputs = inputs[0];
            // Split 'inputs' along batch dimension into array of tensors with length == batch size
            const inputList = tf.split(inputs, inputs.shape[0])
            // Perform STFT on each tensor in the array
            const specBatch = inputList.map(input =>{
                input = input.squeeze();
                // Normalize values between -1 and 1
                input = tf.sub(input, tf.min(input, -1, true));
                input = tf.div(input, tf.max(input, -1, true).add(0.000001));
                input = tf.sub(input, 0.5);
                input = tf.mul(input, 2.0);

                // Perform STFT
                let spec = tf.signal.stft(
                    input,
                    this.frameLength,
                    this.frameStep,
                    this.frameLength,
                    tf.signal.hannWindow,
                );

                // Cast from complex to float
                spec = tf.cast(spec, 'float32');

                // Apply mel filter bank
                spec = tf.matMul(spec, this.melFilterbank);

                // Convert to power spectrogram
                spec = spec.pow(2.0);

                // Apply nonlinearity
                spec = spec.pow(tf.div(1.0, tf.add(1.0, tf.exp(this.magScale.read()))));

                // Flip the spectrogram
                spec = tf.reverse(spec, -1);

                // Swap axes to fit input shape
                spec = tf.transpose(spec)

                // Adding the channel dimension
                spec = spec.expandDims(-1);

                return spec;
            })
            // Convert tensor array into batch tensor
            return tf.stack(specBatch)
        });
    }

    // Optionally, include the `className` method to provide a machine-readable name for the layer
    static get className() {
        return 'MelSpecLayerSimple';
    }
}

// Register the custom layer with TensorFlow.js
tf.serialization.registerClass(MelSpecLayerSimple);

let model;
let labels;
let isInitialized = false;

async function init() {
    await tf.ready();

    [model, labels] = await Promise.all([
        tf.loadLayersModel('../static/birdnet/model.json', { custom_objects: { MelSpecLayerSimple } }),
        fetch('../static/birdnet/labels.json').then(res => res.json())
    ]);

    isInitialized = true;
    self.postMessage({ type: 'ready' });
}

async function chunkAudio(audioData) {
    const { sampleRate, length, channelData } = audioData;
    const chunkSize = 3 * sampleRate;
    const numChunks = Math.ceil(length / chunkSize);

    return Array.from({ length: numChunks }, (_, i) => {
        const start = i * chunkSize;
        const end = Math.min((i + 1) * chunkSize, length);

        const audioChunk = new Float32Array(chunkSize);
        audioChunk.set(channelData.subarray(start, end));

        return audioChunk;
    });
}

async function predict(audioData) {
    if (!isInitialized) {
        self.postMessage({ type: 'error', message: 'Model not initialized. Please wait for initialization to complete.' });
        return;
    }

    self.postMessage({ type: 'status', message: 'Running predictions...' });

    const chunks = await chunkAudio(audioData);
    for (let i = 0; i < chunks.length; i++) {
        const input = tf.tensor(chunks[i]).reshape([1, 144000]);
        const prediction = model.predict(input);
        const probs = await prediction.data();

        // TODO: cache predictions
        
        const top3Results = Array.from(probs)
            .map((prob, index) => ({ species: labels[index], confidence: prob }))
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, 3);
        
        self.postMessage({ type: 'result', chunkIndex: i, results: top3Results });
        
        tf.dispose([input, prediction]); // Dispose of tensors to free memory
    }
    
    self.postMessage({ type: 'status', message: 'All predictions complete!' });
}

self.addEventListener('message', async ({ data }) => {
    if (data.type === 'init') {
        await init();
    } else if (data.type === 'predict') {
        predict(data.audioData);
    }
});