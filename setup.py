from setuptools import setup, find_packages

setup(
    name='yolo_onnx_runner',
    version='0.1.0',
    description='YOLO ONNX segmentation wrapper',
    author='William Engel',
    packages=find_packages(),
    install_requires=[
        'onnxruntime',
        'numpy',
        'opencv-python',
        'matplotlib'
    ],
    python_requires='>=3.7',
)
