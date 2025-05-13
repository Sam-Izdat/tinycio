import torch

def fitted_polynomial_curve_6th_order(x: torch.Tensor, fit: torch.Tensor) -> torch.Tensor:
    """
    Evaluate 6th-order polynomial curve.

    :param x: Input tensor
    :param fit: Coefficient tensor of shape [7]
    :return: Evaluated tensor
    """
    x2 = x * x
    x4 = x2 * x2
    return + (fit[0] * x4 * x2 +
        fit[1] * x4 * x +
        fit[2] * x4 +
        fit[3] * x2 * x +
        fit[4] * x2 +
        fit[5] * x +
        fit[6])

def fitted_polynomial_curve_7th_order(x: torch.Tensor, fit: torch.Tensor) -> torch.Tensor:
    """
    Evaluate 7th-order polynomial curve.

    :param x: Input tensor
    :param fit: Coefficient tensor of shape [8]
    :return: Evaluated tensor
    """
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    return (fit[0] * x6 * x +
            fit[1] * x6 +
            fit[2] * x4 * x +
            fit[3] * x4 +
            fit[4] * x2 * x +
            fit[5] * x2 +
            fit[6] * x +
            fit[7])