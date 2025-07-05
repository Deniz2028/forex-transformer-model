#!/usr/bin/env python3
"""
Backward compatibility wrapper for the modular LSTM system.

This file maintains compatibility with existing calls while 
using the new modular architecture under the hood.
"""

# Import the modular CLI
from src.cli import main

if __name__ == '__main__':
    main()