import numpy as np
import pandas as pd
import os
import numba as nb
import tensorflow as tf
from data_utils import divide_into_windows, divide_into_overlapping_pieces, train_test_split, preprocessing_message_df, compute_accuracy

def test_divide_into_overlapping_pieces():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Test 1: Divide a vector into 3 pieces with overlap size of 1
    overlap_size = 1
    num_pieces = 3
    expected_output = [np.array([1, 2, 3, 4]), np.array([4, 5, 6, 7]), np.array([7, 8, 9, 10])]
    for i in range(len(expected_output)):
        assert np.array_equal(divide_into_overlapping_pieces(data, overlap_size, num_pieces)[i], expected_output[i])

    # Test 2: Divide a vector into 4 pieces with overlap size of 2
    overlap_size = 2
    num_pieces = 4
    expected_output = [np.array([1, 2, 3, 4]), np.array([3, 4, 5, 6]), np.array([5, 6, 7, 8]), np.array([7, 8, 9, 10])]
    for i in range(len(expected_output)):
        assert np.array_equal(divide_into_overlapping_pieces(data, overlap_size, num_pieces)[i], expected_output[i])

    # Test 3: Divide a vector into 2 pieces with overlap size of 0
    overlap_size = 0
    num_pieces = 2
    expected_output = [np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10])]
    for i in range(len(expected_output)):
        assert np.array_equal(divide_into_overlapping_pieces(data, overlap_size, num_pieces)[i], expected_output[i])

    # Test 4: Divide a vector into 1 piece with overlap size of 0
    overlap_size = 0
    num_pieces = 1
    expected_output = [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
    for i in range(len(expected_output)):
        assert np.array_equal(divide_into_overlapping_pieces(data, overlap_size, num_pieces)[i], expected_output[i])

    # Test 5: Divide a vector into 5 pieces with overlap size of 3
    overlap_size = 3
    num_pieces = 5
    expected_output = [np.array([1, 2, 3, 4]), np.array([2, 3, 4, 5]), np.array([3, 4, 5, 6]), np.array([4, 5, 6, 7]), np.array([5, 6, 7, 8, 9, 10])]
    for i in range(len(expected_output)):
        assert np.array_equal(divide_into_overlapping_pieces(data, overlap_size, num_pieces)[i], expected_output[i])

def test_divide_into_windows():
    # create some test data
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    window_size = 2
    # divide the data into windows
    windows = divide_into_windows(data, window_size)
    # check if the windows have been computed correctly
    assert np.allclose(windows[0], np.array([[1, 2, 3], [4, 5, 6]]))
    assert np.allclose(windows[1], np.array([[4, 5, 6], [7, 8, 9]]))

def test_train_test_split():
    # Define a sample data array.
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Test 1: Standard case with default train_size.
    train_data, test_data = train_test_split(data)
    assert len(train_data) == 8  # 75% of 10 elements (rounded up).
    assert len(test_data) == 2
    assert np.array_equal(train_data, np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    assert np.array_equal(test_data, np.array([9, 10]))

    # Test 2: Edge case with train_size = 1 (all data for training).
    train_data, test_data = train_test_split(data, train_size=1)
    assert len(train_data) == 10  # 100% of 10 elements.
    assert len(test_data) == 0
    assert np.array_equal(train_data, data)
    assert np.array_equal(test_data, np.array([]))

    # Test 3: Edge case with train_size = 0 (all data for testing).
    train_data, test_data = train_test_split(data, train_size=0)
    assert len(train_data) == 0
    assert len(test_data) == 10  # 100% of 10 elements.
    assert np.array_equal(train_data, np.array([]))
    assert np.array_equal(test_data, data)

    # Test 4: Check non-default train_size.
    train_data, test_data = train_test_split(data, train_size=0.6)
    assert len(train_data) == 6  # 60% of 10 elements.
    assert len(test_data) == 4
    assert np.array_equal(train_data, np.array([1, 2, 3, 4, 5, 6]))
    assert np.array_equal(test_data, np.array([7, 8, 9, 10]))

    # Test 5: Check with non-integer elements.
    data_float = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    train_data, test_data = train_test_split(data_float, train_size=0.6)
    assert len(train_data) == 3
    assert len(test_data) == 2
    assert np.array_equal(train_data, np.array([1.1, 2.2, 3.3]))
    assert np.array_equal(test_data, np.array([4.4, 5.5]))

    # Test 6: Check with a 3D array.
    data_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    train_data, test_data = train_test_split(data_3d, train_size=0.66)  # Approximately 2/3 for training.
    assert train_data.shape == (2, 2, 2)  # Expecting 2 'sheets' in the training data.
    assert test_data.shape == (1, 2, 2)  # Expecting 1 'sheet' in the test data.
    assert np.array_equal(train_data, np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    assert np.array_equal(test_data, np.array([[[9, 10], [11, 12]]]))

def test_preprocessing_message_df():
    # Sample data frame.
    data = {
        'Order ID': [1, 2, 3, 4, 5, 6, 7, 8],
        'Time': [10, 20, 30, 40, 50, 60, 70, 80],
        'Event type': [1, 2, 3, 4, 5, 6, 1, 2],
        'Other Data': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    }
    message_df = pd.DataFrame(data)
    
    # Test 1: Basic functionality.
    processed_df, index = preprocessing_message_df(message_df)
    
    # Check if 'Order ID' and 'Time' columns are dropped.
    assert 'Order ID' not in processed_df.columns
    assert 'Time' not in processed_df.columns
    
    # Check if only rows with event type 1,2,3, or 4 are retained.
    assert processed_df['Event type'].isin([1, 2, 3, 4]).all()
    
    # Check if m=20000 rows are discarded from the start and end.
    # Note: In this test case, the dataframe should be empty, as it has fewer than 40000 rows.
    assert processed_df.empty
    
    # Test 2: Verify functionality with different event types and m value.
    data['Event type'] = [1, 2, 1, 2, 3, 3, 4, 4]  # Updating event types for different test scenario.
    message_df = pd.DataFrame(data)
    m_val = 2
    processed_df, index = preprocessing_message_df(message_df, m=m_val)
    
    # Check the m value truncation and other functionality.
    assert len(processed_df) == len(message_df) - 2 * m_val  # m rows should be discarded from both start and end.
    assert processed_df['Event type'].isin([1, 2, 3, 4]).all()
    assert 'Order ID' not in processed_df.columns
    assert 'Time' not in processed_df.columns

def test_compute_accuracy():
    # Test with two outputs
    real_output = tf.constant([0.7, 0.9, 0.1, 0.4], dtype=tf.float32)
    fake_output = tf.constant([0.1, 0.2, 0.8, 0.4], dtype=tf.float32)

    accuracy = compute_accuracy([real_output, fake_output])
    assert abs(accuracy.numpy() - 0.625) < 1e-7, f"Unexpected value: {accuracy.numpy()}"

    # Test with one output
    fake_output = tf.constant([0.1, 0.2, 0.8, 0.4], dtype=tf.float32)
    accuracy = compute_accuracy(fake_output)
    assert abs(accuracy.numpy() - 0.75) < 1e-7, f"Unexpected value: {accuracy.numpy()}"

    # Test with empty output
    try:
        compute_accuracy([])
    except ValueError as e:
        assert str(e) == "not enough values to unpack (expected 2, got 0)"

    # Test with invalid type
    fake_output = tf.constant([0.7, 0.9, 0.1, 0.4], dtype=tf.float32)

    try:
        compute_accuracy([fake_output, 'invalid'])
    except TypeError as e:
        assert "Cannot convert 0.5 to EagerTensor of dtype string" in str(e)

# Run the tests
test_train_test_split()
test_divide_into_windows()
test_divide_into_overlapping_pieces()
test_preprocessing_message_df()