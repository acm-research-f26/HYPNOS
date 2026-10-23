"""Test parsing and forward pass with fabricated data; NOT a performance test."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from preprocess import extract_subject
from model import ActigraphyCNN
import torch

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    m, l = root/'123_acceleration.txt', root/'123_labeled_sleep.txt'
    t = np.arange(-1., 61., .02)
    acceleration = np.column_stack([t, np.sin(t), np.cos(t), np.sin(2*t)])
    np.savetxt(m, acceleration)
    np.savetxt(l, np.array([[0.,0.],[30.,2.]]))
    x,y,times = extract_subject(m,l)
    print('Shapes:', x.shape, y.shape, times.shape, 'labels:', y.tolist())
    assert x.shape == (2,3,300) and y.tolist() == [0,1]
    out = ActigraphyCNN()(torch.from_numpy(x))
    assert out.shape == (2,1)
    print('PASS: synthetic data parsed, windows aligned, CNN executed.')
