import subprocess
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, mean_squared_error
import cv2

# 1. Test

path_to_origin = "assets/examples/bird_canny_blue.png"
path_to_test = "outputs/bird.png"

args = ['--model_name',
        'edge_to_image',
        '--input_image',
        'assets/examples/bird.png',
        '--prompt',
        'a blue bird',
        '--output_dir',
        './outputs']
command_line = " ". join(['python',
                            'src/inference_paired.py'] + args)
process = subprocess.Popen(['python',
                            'src/inference_paired.py'] + args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
if process.returncode == 0:
    print("1. Test results:")

    image_origin = cv2.imread(path_to_origin)
    image_origin = cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY)
    image_test = cv2.imread(path_to_origin)
    image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY) 
    (ssim, diff) = structural_similarity(image_origin, image_test, full=True)
    print("SSIM: {}".format(ssim))
    mse = mean_squared_error(image_origin, image_test)
    print("MSE: {}".format(mse))

else:
    print("1. Test NOT passed.")
    print(stdout.decode())
    print(stderr.decode())

# 2. Test

path_to_origin = "assets/examples/sketch_output.png"
path_to_test = "outputs/bird.png"

args = ['--model_name',
        'sketch_to_image_stochastic',
        '--input_image',
        './assets/examples/sketch_input.png',
        '--gamma',
        '0.4',
        '--prompt',
        'ethereal fantasy concept art of an asteroid. magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy',
        '--output_dir',
        './outputs']
command_line = " ". join(['python',
                            'src/inference_paired.py'] + args)
process = subprocess.Popen(['python',
                            'src/inference_paired.py'] + args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
if process.returncode == 0:
    print("2. Test results:")

    image_origin = cv2.imread(path_to_origin)
    image_origin = cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY)
    image_test = cv2.imread(path_to_origin)
    image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY) 
    (ssim, diff) = structural_similarity(image_origin, image_test, full=True)
    print("SSIM: {}".format(ssim))
    mse = mean_squared_error(image_origin, image_test)
    print("MSE: {}".format(mse))
else:
    print("2. Test NOT passed.")
    print(stdout.decode())
    print(stderr.decode())