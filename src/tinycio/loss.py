import torch

def feature_moments_calculation(feat, eps=1e-5):
    # https://github.com/semchan/NLUT/blob/main/LICENSE
    # MIT License
    # <!-- Copyright (c) 2010, 2011 the Friendika Project -->
    # All rights reserved.

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    # THE SOFTWARE
    size = feat.size()
    assert (len(size) == 3)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    # feat_std = feat_var.sqrt().view(N, C, 1, 1)
    # the first order
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)

    # the second order
    feat_size = 2
    # N, C = size[:2]
    feat_p2 = torch.abs(feat-feat_mean).pow(feat_size).view(N, C, -1)
    N, C,L = feat_p2.shape
    feat_p2 = feat_p2.sum(dim=2)/L
    feat_p2 = feat_p2.pow(1/feat_size).view(N, C, 1)
    # the third order
    feat_size = 3
    # N, C = size[:2]
    feat_p3 = torch.abs(feat-feat_mean).pow(feat_size).view(N, C, -1)
    # N, C,L = feat_p3.shape
    feat_p3 = feat_p3.sum(dim=2)/L
    feat_p3 = feat_p3.pow(1/feat_size).view(N, C, 1)

    return feat_mean.view(N, C) , feat_p2.view(N, C), feat_p3.view(N, C)