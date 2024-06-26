#!/bin/bash

# Data generation
for fake_type in style_transfer inpainting
do
    python generate_train_data.py --fake_type $fake_type
    python generate_test_data.py --fake_type $fake_type
done

# Feature selection (based on only training data)
for fake_type in style_transfer inpainting
do
    python feature_selection.py --fake_type $fake_type
done

# Train & test
for fake_type in style_transfer inpainting
do
    python main.py --fake_type $fake_type
done
