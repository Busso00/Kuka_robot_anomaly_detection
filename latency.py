from models.MTGFLOW import MTGFLOW #TODO: add your import
import tensorflow as tf
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='name of model to measure latency')
model = parser.parse_args().model

if model == "MTGFLOW":
    #change model definition
    model = MTGFLOW(1,#n_blocks 
                  1, #input_size=1
                  47, #hidden_size
                  1, #n_hidden=1 
                  window_size=100, 
                  n_sensor=50,
                  dropout=0.0, 
                  batch_norm = False)

    #change input shape
    input_tensor = tf.random.normal(shape=(1, 50, 100, 1))

    #250 ms on Colab CPU, batch size=1
    #30 ms on Colab T4 GPU, batch size=1
    #NOTE: with build you can avoid warmup iterations

elif model == "TADGAN":
    pass #TODO

elif model == "VAE-LSTM":
    pass #TODO

elif model == "MSCRED":
    pass #TODO


# Warm-up to ensure accurate timing
for _ in range(10):
    _ = model(input_tensor)

# Measure latency (bs=1)
def measure_latency(model, input_tensor, num_runs=1000):
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(input_tensor)
    end_time = time.time()

    avg_latency = (end_time - start_time) / num_runs
    return avg_latency

# Run the latency measurement
latency = measure_latency(model, input_tensor)
print(f"Average Latency: {latency * 1000:.2f} ms per forward pass")
