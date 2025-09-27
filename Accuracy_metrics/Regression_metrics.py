import numpy as np


def MSE(y_act, y_pred):
    """
    This estimates the mean squared error of the model.

    Args:
    y_act (array-like): Actual values.
    y_pred (array-like): Predicted values.

    Returns:
    float: Mean Squared Error.
    """
    if len(y_act) != len(y_pred):
        raise ValueError("The lengths of y_act and y_pred must be equal.")
    
    y_act = np.array(y_act)
    y_pred = np.array(y_pred)
    
    err = np.mean((y_act - y_pred) ** 2)
    return err

def MAE(y_act, y_pred) -> float:
    """
    This estimates the mean absolute error of the model.

    Args:
    y_act (array-like): Actual values.
    y_pred (array-like): Predicted values.

    Returns:
    float: Mean Absolute Error.
    """
    if len(y_act) != len(y_pred):
        raise ValueError("The lengths of y_act and y_pred must be equal.")
    
    y_act = np.array(y_act)
    y_pred = np.array(y_pred)
    
    err = np.mean(np.abs((y_act - y_pred)))
    return err

def RMSE(y_act, y_pred) -> float:
    """
    This estimates the root mean squared error of the model.

    Args:
    y_act (array-like): Actual values.
    y_pred (array-like): Predicted values.

    Returns:
    float: Root  Mean Squared Error.
    """
    if len(y_act) != len(y_pred):
        raise ValueError("The lengths of y_act and y_pred must be equal.")
    
    y_act = np.array(y_act)
    y_pred = np.array(y_pred)
    
    err = np.sqrt(np.mean(np.abs((y_act - y_pred))))
    return err

def R_square(y_act, y_pred)  -> float:
    """
    This estimates the Root mean squared error of the model.

    Args:
    y_act (array-like): Actual values.
    y_pred (array-like): Predicted values.

    Returns:
    float: Root Mean Squared Error.
    """
    if len(y_act) != len(y_pred):
        raise ValueError("The lengths of y_act and y_pred must be equal.")
    
    y_act = np.array(y_act)
    y_pred = np.array(y_pred)
    
    ss_total = np.sum((y_act - np.mean(y_act)) ** 2)
    ss_residual = np.sum((y_act - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2
    
    
def MAPE(y_act, y_pred) -> float:
    """
    This estimates the  mean absolute percentage error of the model.

    Args:
    y_act (array-like): Actual values.
    y_pred (array-like): Predicted values.

    Returns:
    float: Mean Absolute Percentage Error.
    """
    if len(y_act) != len(y_pred):
        raise ValueError("The lengths of y_act and y_pred must be equal.")
    
    y_act = np.array(y_act)
    y_pred = np.array(y_pred)
    
    err = np.mean(np.abs((y_act - y_pred) / y_act)) * 100
    return err

def MBD(y_act, y_pred) -> float:
    """
    This estimates the mean bias deviation of the model.

    Args:
    y_act (array-like): Actual values.
    y_pred (array-like): Predicted values.

    Returns:
    float: Mean Bias Deviation.
    """
    if len(y_act) != len(y_pred):
        raise ValueError("The lengths of y_act and y_pred must be equal.")
    
    y_act = np.array(y_act)
    y_pred = np.array(y_pred)
    
    err = np.mean(y_pred - y_act)
    return err

def HuberLoss(y_act, y_pred, delta=1.0) -> float:
    """
    This estimates the Huber loss of the model.

    Args:
    y_act (array-like): Actual values.
    y_pred (array-like): Predicted values.
    delta (float): The point where the Huber loss function changes from a quadratic to linear.

    Returns:
    float: Huber Loss.
    """
    if len(y_act) != len(y_pred):
        raise ValueError("The lengths of y_act and y_pred must be equal.")
    
    y_act = np.array(y_act)
    y_pred = np.array(y_pred)
    
    error = y_act - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = np.square(error) / 2
    linear_loss = delta * (np.abs(error) - delta / 2)
    
    return np.where(is_small_error, squared_loss, linear_loss).mean()